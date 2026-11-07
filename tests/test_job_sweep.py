"""Tests for run_sweep_via_job() -- delegates a sweep's execution and
durability to Lemonade's own job engine (POST /v1/jobs) instead of this
process making N direct sequential HTTP calls. See bench.py's
run_sweep_via_job docstring for the live-verified contract this rests on.
"""

import conftest
import pytest
from conftest import get_fake_server

from lemonmatrix.bench import JobFailedError, SweepConfig, run_sweep_via_job
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.validate import validate_result


def _cfg(**overrides) -> SweepConfig:
    defaults = dict(
        model_name="Llama-3.1-8B-Instruct-GGUF",
        model_class="dense",
        quantization="Q4_K_M",
        context_length=4096,
        compute_engine="igpu",
        backend="llama.cpp-vulkan",
        os="windows",
        power_state="plugged",
        warmup_trials=1,
        measured_trials=2,
    )
    defaults.update(overrides)
    return SweepConfig(**defaults)


def test_run_sweep_via_job_produces_schema_conformant_result(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_sweep_via_job(client, environment, _cfg())

    validate_result(result)  # raises on any schema violation
    assert result["metrics"]["decode"]["tokens_per_sec"] == 42.0
    assert result["metrics"]["ttft_ms"] == 180.0
    assert result["metrics"]["trial_count"] == 2
    assert len(trials) == 2
    assert all("ttft_ms" in t and "decode_tokens_per_sec" in t for t in trials)


def test_run_sweep_via_job_matches_direct_http_aggregation(fake_lemonade):
    # The whole point of sharing _aggregate_sweep_result: identical raw
    # measurements (the fake server reports the same FAKE_STATS either way)
    # must produce identical aggregated metrics regardless of execution path.
    from lemonmatrix.bench import run_sweep

    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    direct_result, _ = run_sweep(client, environment, _cfg())
    job_result, _ = run_sweep_via_job(client, environment, _cfg())

    assert job_result["metrics"]["decode"]["tokens_per_sec"] == direct_result["metrics"]["decode"]["tokens_per_sec"]
    assert job_result["metrics"]["prefill"]["tokens_per_sec"] == pytest.approx(
        direct_result["metrics"]["prefill"]["tokens_per_sec"]
    )
    assert job_result["metrics"]["ttft_ms"] == direct_result["metrics"]["ttft_ms"]


def test_run_sweep_via_job_tells_lemonade_which_backend_to_load(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_sweep_via_job(client, environment, _cfg(backend="llama.cpp-cuda"))

    payload = get_fake_server(fake_lemonade).last_load_payload
    assert payload["llamacpp_backend"] == "cuda"
    # Confirmed against Lemonade's own job-engine source: load's job-op
    # params use "model", not "model_name" like the direct /api/v1/load body.
    assert payload["model"] == "Llama-3.1-8B-Instruct-GGUF"


def test_run_sweep_via_job_rejects_router_runs(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    with pytest.raises(ValueError, match="does not support router runs"):
        run_sweep_via_job(client, environment, _cfg(run_type="router"))


def test_run_sweep_via_job_deletes_the_job_after_completion(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_sweep_via_job(client, environment, _cfg())

    server = get_fake_server(fake_lemonade)
    assert server.fake_jobs == {}


def test_run_sweep_via_job_raises_job_failed_error_on_failure(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    with pytest.raises(JobFailedError, match="failed"):
        run_sweep_via_job(client, environment, _cfg(model_name="trigger-load-failure"))


def test_run_sweep_via_job_marks_invalid_when_engine_backend_incompatible(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep_via_job(client, environment, _cfg(compute_engine="cpu", backend="llamacpp-rocm"))

    validate_result(result)
    assert result["validity"]["valid"] is False
    assert "engine/backend mismatch" in result["validity"]["notes"]


def test_run_sweep_via_job_detects_competing_model_via_health_polling(fake_lemonade, monkeypatch):
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "someone-elses-model", "is_busy": True, "is_streaming": False}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep_via_job(client, environment, _cfg(exclusive_run=True))

    validate_result(result)
    assert result["validity"]["exclusive_run"] is False
    assert result["validity"]["valid"] is False
    assert "competing workload detected during measurement: someone-elses-model" in result["validity"]["notes"]
