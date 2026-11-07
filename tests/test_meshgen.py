import conftest
import pytest
from conftest import get_fake_server

from lemonmatrix.bench import MeshGenConfig, run_meshgen
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.results_store import list_meshgen_results, list_results, save_meshgen_result
from lemonmatrix.validate import validate_meshgen_result

FAKE_INPUT_IMAGE_BYTES = b"\x89PNG\r\n\x1a\nnot a real png but the fake server never decodes it"


def _cfg(**overrides) -> MeshGenConfig:
    defaults = dict(
        model_name=conftest.FAKE_MESHGEN_MODEL_ID,
        compute_engine="dgpu",
        backend="trellis-cuda",
        os="windows",
        power_state="plugged",
        input_image_bytes=FAKE_INPUT_IMAGE_BYTES,
        warmup_trials=1,
        measured_trials=2,
    )
    defaults.update(overrides)
    return MeshGenConfig(**defaults)


def test_run_meshgen_produces_schema_conformant_result(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_meshgen(client, environment, _cfg())

    validate_meshgen_result(result)  # raises on any schema violation
    assert result["run_type"] == "meshgen"
    assert result["model"]["name"] == conftest.FAKE_MESHGEN_MODEL_ID
    assert result["model"]["checkpoint"] == conftest.FAKE_MESHGEN_CHECKPOINT
    assert result["metrics"]["trial_count"] == 2
    assert result["metrics"]["generation_time_ms"] > 0
    assert result["metrics"]["meshes_per_sec"] == pytest.approx(
        1000 / result["metrics"]["generation_time_ms"]
    )
    assert result["validity"]["model_reload_free"] is True
    assert result["validity"]["warmup_discarded"] is True
    # trellis/cuda is "installed" in the fake server's recipes tree.
    assert result["environment"]["backend_version"] == "2.0.0"
    assert len(trials) == 2
    assert all("generation_time_ms" in t for t in trials)


def test_run_meshgen_records_optional_params(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_meshgen(client, environment, _cfg(resolution="1024", bg_removal="birefnet", uv="xatlas", seed=42))

    validate_meshgen_result(result)
    assert result["metrics"]["resolution"] == "1024"
    assert result["metrics"]["bg_removal"] == "birefnet"
    assert result["metrics"]["uv"] == "xatlas"
    assert result["metrics"]["seed"] == 42


def test_run_meshgen_tells_lemonade_which_backend_to_load(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_meshgen(client, environment, _cfg())

    payload = get_fake_server(fake_lemonade).last_load_payload
    assert payload["trellis_backend"] == "cuda"


def test_run_meshgen_marks_invalid_when_engine_backend_incompatible(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_meshgen(client, environment, _cfg(compute_engine="npu"))

    validate_meshgen_result(result)
    assert result["validity"]["valid"] is False
    assert "engine/backend mismatch" in result["validity"]["notes"]


def test_run_meshgen_detects_competing_model_via_health_polling(fake_lemonade, monkeypatch):
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "someone-elses-model", "is_busy": True, "is_streaming": False}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_meshgen(client, environment, _cfg(exclusive_run=True))

    validate_meshgen_result(result)
    assert result["validity"]["exclusive_run"] is False
    assert result["validity"]["valid"] is False
    assert "competing workload detected during measurement: someone-elses-model" in result["validity"]["notes"]


def test_save_and_list_meshgen_results_is_isolated_from_list_results(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    result, _ = run_meshgen(client, environment, _cfg())

    save_meshgen_result(tmp_path, "demo-profile", result)

    meshgen_results = list_meshgen_results(tmp_path)
    assert len(meshgen_results) == 1
    assert meshgen_results[0]["run_id"] == result["run_id"]
    assert meshgen_results[0]["_profile"] == "demo-profile"

    # The main leaderboard's one-level-deep glob must never see this file --
    # it doesn't conform to result.schema.json.
    assert list_results(tmp_path) == []
