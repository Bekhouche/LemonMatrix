"""Tests for router runs, trials sidecar, SweepStore, and CSV export.

These tests cover the areas that were absent after the router integration:
- Router runs produce valid schema results (not marked invalid by backend check)
- route_trace data is captured in raw trials
- config.router_default_model is populated when a trial has default_used=True
- Trials sidecar: save_trials / load_trials round-trip
- list_results() never returns trials sidecar files
- SweepStore: persist / rehydrate / interrupt_running_batches
- CSV: gpu column uses dgpu/igpu (not the absent "gpu" key); run_type column present
"""

import json
import time

import pytest
from conftest import FAKE_ROUTER_MODEL_ID, FAKE_ROUTE_TRACE, get_fake_server

from lemonmatrix.bench import SweepConfig, run_sweep
from lemonmatrix.capabilities import available_routers, is_router_model
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.results_store import (
    list_results,
    load_trials,
    results_to_csv,
    save_trials,
)
from lemonmatrix.sweep_store import SweepStore
from lemonmatrix.validate import validate_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _router_cfg(**overrides) -> SweepConfig:
    defaults = dict(
        model_name=FAKE_ROUTER_MODEL_ID,
        model_class="router",
        quantization="none",
        context_length=4096,
        compute_engine="router",
        backend="collection.router",
        os="linux",
        power_state="plugged",
        warmup_trials=1,
        measured_trials=2,
        run_type="router",
    )
    defaults.update(overrides)
    return SweepConfig(**defaults)


# ---------------------------------------------------------------------------
# capabilities helpers
# ---------------------------------------------------------------------------

def test_is_router_model_detects_collection_recipe():
    assert is_router_model({"id": "my-router", "recipe": "collection.router"})


def test_is_router_model_detects_collection_id_prefix():
    assert is_router_model({"id": "collection.my-policy"})


def test_is_router_model_rejects_normal_model():
    assert not is_router_model({"id": "Llama-3.1-8B-Instruct-GGUF", "recipe": "llamacpp"})


def test_available_routers_filters_correctly(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    models = client.models()
    routers = available_routers(models)
    assert len(routers) >= 1
    assert all(is_router_model(m) for m in routers)
    router_ids = [m["id"] for m in routers]
    assert FAKE_ROUTER_MODEL_ID in router_ids


# ---------------------------------------------------------------------------
# Router bench run
# ---------------------------------------------------------------------------

def test_router_run_produces_schema_conformant_result(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_sweep(client, environment, _router_cfg())

    validate_result(result)  # must not raise


def test_router_run_is_marked_valid(fake_lemonade):
    """Backend resolution must be skipped for routers so valid=True."""
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _router_cfg())

    assert result["validity"]["valid"] is True, (
        f"Router run must not fail the backend-resolution validity gate. "
        f"validity={result['validity']}"
    )


def test_router_run_skips_device_verification(fake_lemonade, monkeypatch):
    # A router's compute_engine is "router" -- there is no single physical
    # device to verify (the downstream model's engine varies per routed
    # request, per idea.md), so a device report that would fail a model
    # run's check must not affect a router run's validity.
    import conftest

    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": FAKE_ROUTER_MODEL_ID, "is_busy": False, "is_streaming": False, "device": "cpu"}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _router_cfg())

    assert result["validity"]["valid"] is True
    assert "compute_engine claimed" not in result["validity"].get("notes", "")


def test_router_run_sets_run_type_field(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _router_cfg())

    assert result.get("run_type") == "router"


def test_router_run_captures_route_trace_in_trials(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    _, trials = run_sweep(client, environment, _router_cfg())

    for trial in trials:
        assert "route_to" in trial, "route_to missing from trial"
        assert trial["route_to"] == FAKE_ROUTE_TRACE["route_to"]
        assert "matched_rule" in trial
        assert "default_used" in trial


def test_router_run_no_backend_version(fake_lemonade):
    """collection.router is not a recipe — no backend_version should be set."""
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _router_cfg())

    # backend_version is only set when a real recipe version was found
    assert "backend_version" not in result.get("environment", {})


def test_router_run_router_default_model_populated_when_default_used(fake_lemonade, monkeypatch):
    """When a trial has default_used=True, router_default_model should be set."""
    import lemonmatrix.bench as bench_mod

    original_run_one = bench_mod._run_one

    call_count = [0]

    def patched_run_one(client, cfg):
        result = original_run_one(client, cfg)
        call_count[0] += 1
        # Patch the second trial (first measured) to have default_used=True
        if call_count[0] == 2:
            result["route_to"] = "fallback-model"
            result["matched_rule"] = None
            result["default_used"] = True
        return result

    monkeypatch.setattr(bench_mod, "_run_one", patched_run_one)

    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _router_cfg())

    assert result["config"].get("router_default_model") == "fallback-model"


# ---------------------------------------------------------------------------
# Trials sidecar
# ---------------------------------------------------------------------------

def test_save_and_load_trials_round_trip(tmp_path):
    run_id = "abc-123"
    profile = "demo"
    data = [
        {"trial": 0, "ttft_ms": 180.0, "decode_tokens_per_sec": 42.0},
        {"trial": 1, "ttft_ms": 175.0, "decode_tokens_per_sec": 44.0},
    ]

    save_trials(tmp_path, profile, run_id, data)
    loaded = load_trials(tmp_path, profile, run_id)

    assert loaded == data


def test_trials_do_not_appear_in_list_results(tmp_path):
    """save_trials must never cause the trials sidecar to show up in list_results."""
    run_id = "xyz-999"
    profile = "test"
    save_trials(tmp_path, profile, run_id, [{"trial": 0}])

    results = list_results(tmp_path)
    assert results == [], (
        "trials sidecar files must not be returned by list_results()"
    )


def test_load_trials_returns_none_when_missing(tmp_path):
    """load_trials signals an absent sidecar by returning None (not [])."""
    loaded = load_trials(tmp_path, "no-such-profile", "no-such-run")
    assert loaded is None


# ---------------------------------------------------------------------------
# SweepStore
# ---------------------------------------------------------------------------

class _FakeBatch:
    """Minimal batch-like object accepted by SweepStore.save_batch."""
    def __init__(self, batch_id, profile="demo", status="pending"):
        self.id = batch_id
        self.profile_name = profile
        self.created_at = "2026-01-01T00:00:00"
        self.status = status
        self.items = [
            {"cfg": {"model_name": "A"}, "status": "pending"},
            {"cfg": {"model_name": "B"}, "status": "pending"},
        ]


def test_sweep_store_round_trip(tmp_path):
    store = SweepStore(tmp_path / ".sweeps.db")
    batch = _FakeBatch("batch-1")

    store.save_batch(batch)
    all_batches = store.load_all_batches()

    assert len(all_batches) == 1
    loaded = all_batches[0]
    assert loaded["id"] == "batch-1"
    assert loaded["profile_name"] == "demo"
    assert len(loaded["items"]) == 2


def test_sweep_store_update_item(tmp_path):
    store = SweepStore(tmp_path / ".sweeps.db")
    batch = _FakeBatch("batch-2")
    store.save_batch(batch)

    store.update_item("batch-2", 0, {"cfg": {"model_name": "A"}, "status": "completed"})

    all_batches = store.load_all_batches()
    assert all_batches[0]["items"][0]["status"] == "completed"


def test_sweep_store_finish_batch(tmp_path):
    store = SweepStore(tmp_path / ".sweeps.db")
    batch = _FakeBatch("batch-3", status="running")
    store.save_batch(batch)
    store.finish_batch("batch-3", "done")

    loaded = store.load_all_batches()
    assert loaded[0]["status"] == "done"


def test_sweep_store_interrupt_running_batches(tmp_path):
    db = tmp_path / ".sweeps.db"
    store = SweepStore(db)

    b1 = _FakeBatch("b-run", status="running")
    b2 = _FakeBatch("b-done", status="done")
    store.save_batch(b1)
    store.save_batch(b2)

    # Simulate restart: new store instance over same DB
    store2 = SweepStore(db)
    store2.interrupt_running_batches()

    all_batches = {b["id"]: b for b in store2.load_all_batches()}
    assert all_batches["b-run"]["status"] == "interrupted"
    assert all_batches["b-done"]["status"] == "done"


def test_sweep_store_empty_load(tmp_path):
    store = SweepStore(tmp_path / ".sweeps.db")
    assert store.load_all_batches() == []


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _make_result(run_id="r1", profile="demo", run_type=None, dgpu="AMD Radeon 780M"):
    r = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "timestamp": "2026-01-01T00:00:00",
        "_profile": profile,
        "model": {
            "name": "TestModel",
            "class": "dense",
            "quantization": "Q4_K_M",
            "context_length": 4096,
        },
        "config": {
            "compute_engine": "igpu",
            "backend": "llama.cpp-vulkan",
            "os": "linux",
            "power_state": "plugged",
        },
        "environment": {
            "device_model": "Framework 16",
            "igpu": "AMD Radeon 780M",
            "os_version": "Ubuntu 24.04",
        },
        "metrics": {
            "decode": {"tokens_per_sec": 42.0},
            "prefill": {"tokens_per_sec": 222.0},
            "ttft_ms": 180.0,
            "peak_memory_gb": 9.5,
            "trial_count": 2,
        },
        "validity": {"valid": True, "notes": ""},
    }
    if dgpu:
        r["environment"]["dgpu"] = dgpu
    if run_type:
        r["run_type"] = run_type
    return r


def test_csv_gpu_column_uses_dgpu_when_present():
    r = _make_result(dgpu="AMD Radeon RX 7900 XTX")
    csv_text = results_to_csv([r])
    lines = csv_text.strip().splitlines()
    assert len(lines) == 2
    assert "AMD Radeon RX 7900 XTX" in lines[1]


def test_csv_gpu_column_falls_back_to_igpu_when_no_dgpu():
    r = _make_result(dgpu=None)
    r["environment"]["igpu"] = "AMD Radeon 780M"
    csv_text = results_to_csv([r])
    assert "AMD Radeon 780M" in csv_text


def test_csv_gpu_column_not_empty_for_typical_igpu_result():
    """Regression: environment.gpu does not exist — column must not be blank."""
    r = _make_result(dgpu=None)
    csv_text = results_to_csv([r])
    import csv, io
    reader = csv.DictReader(io.StringIO(csv_text))
    row = next(reader)
    assert row["gpu"] != "", "gpu CSV column should be populated from igpu/dgpu"


def test_csv_run_type_column_model_default():
    r = _make_result()  # no run_type key → defaults to "model"
    csv_text = results_to_csv([r])
    import csv, io
    reader = csv.DictReader(io.StringIO(csv_text))
    row = next(reader)
    assert row["run_type"] == "model"


def test_csv_run_type_column_router():
    r = _make_result(run_type="router")
    r["model"]["class"] = "router"
    r["config"]["compute_engine"] = "router"
    r["config"]["backend"] = "collection.router"
    r["config"]["router_default_model"] = "Llama-3.1-8B-Instruct-GGUF"
    csv_text = results_to_csv([r])
    import csv, io
    reader = csv.DictReader(io.StringIO(csv_text))
    row = next(reader)
    assert row["run_type"] == "router"
    assert row["router_default_model"] == "Llama-3.1-8B-Instruct-GGUF"


def test_router_run_recognizes_its_own_routed_candidate_as_not_competing(fake_lemonade, monkeypatch):
    """A router is virtual and never appears in /api/v1/health itself --
    confirmed live that only whichever candidate it actually dispatched to
    shows up there. Before this was fixed, the exclusivity monitor compared
    health entries against the router's own collection name and could never
    match, so it misidentified the router's own selected candidate
    (FAKE_ROUTE_TRACE's "Llama-3.1-8B-Instruct-GGUF", declared as the
    router's sole "components" entry in conftest.py) as an unrelated
    competing workload on every single trial.
    """
    import conftest

    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "Llama-3.1-8B-Instruct-GGUF", "is_busy": True, "is_streaming": False, "device": "cpu"}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _router_cfg())

    assert result["validity"]["valid"] is True
    assert "competing workload" not in result["validity"].get("notes", "")
