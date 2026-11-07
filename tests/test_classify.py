import conftest
import pytest

from lemonmatrix.bench import ClassifyConfig, run_classify
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.results_store import list_classify_results, list_results, save_classify_result
from lemonmatrix.validate import validate_classify_result


def _cfg(**overrides) -> ClassifyConfig:
    defaults = dict(
        model_name=conftest.FAKE_CLASSIFY_MODEL_ID,
        compute_engine="cpu",
        backend="onnxruntime-cpu",
        os="windows",
        power_state="plugged",
        input_text="Please verify your account at http://secure-login.example now.",
        warmup_trials=1,
        measured_trials=2,
    )
    defaults.update(overrides)
    return ClassifyConfig(**defaults)


def test_run_classify_produces_schema_conformant_result(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_classify(client, environment, _cfg())

    validate_classify_result(result)  # raises on any schema violation
    assert result["run_type"] == "classify"
    assert result["model"]["name"] == conftest.FAKE_CLASSIFY_MODEL_ID
    assert result["model"]["checkpoint"] == conftest.FAKE_CLASSIFY_CHECKPOINT
    assert result["metrics"]["trial_count"] == 2
    assert result["metrics"]["latency_ms"] > 0
    assert result["metrics"]["classifications_per_sec"] == pytest.approx(
        1000 / result["metrics"]["latency_ms"]
    )
    assert result["metrics"]["input_chars"] == len(_cfg().input_text)
    assert result["validity"]["model_reload_free"] is True
    assert result["validity"]["warmup_discarded"] is True
    # onnxruntime/cpu is "installed" in the fake server's recipes tree.
    assert result["environment"]["backend_version"] == "0.3.7"
    assert len(trials) == 2
    assert all("latency_ms" in t for t in trials)


def test_run_classify_respects_top_k(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_classify(client, environment, _cfg(top_k=1))

    validate_classify_result(result)
    assert result["metrics"]["top_k"] == 1


def test_run_classify_marks_invalid_when_engine_backend_incompatible(fake_lemonade):
    # npu engine + onnxruntime-cpu backend is a physical contradiction: the
    # onnxruntime recipe on this instance only has a cpu backend.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_classify(client, environment, _cfg(compute_engine="npu"))

    validate_classify_result(result)
    assert result["validity"]["valid"] is False
    assert "engine/backend mismatch" in result["validity"]["notes"]


def test_run_classify_detects_competing_model_via_health_polling(fake_lemonade, monkeypatch):
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "someone-elses-model", "is_busy": True, "is_streaming": False}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_classify(client, environment, _cfg(exclusive_run=True))

    validate_classify_result(result)
    assert result["validity"]["exclusive_run"] is False
    assert result["validity"]["valid"] is False
    assert "competing workload detected during measurement: someone-elses-model" in result["validity"]["notes"]


def test_save_and_list_classify_results_is_isolated_from_list_results(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    result, _ = run_classify(client, environment, _cfg())

    save_classify_result(tmp_path, "demo-profile", result)

    classify_results = list_classify_results(tmp_path)
    assert len(classify_results) == 1
    assert classify_results[0]["run_id"] == result["run_id"]
    assert classify_results[0]["_profile"] == "demo-profile"

    # The main leaderboard's one-level-deep glob must never see this file --
    # it doesn't conform to result.schema.json.
    assert list_results(tmp_path) == []


def test_list_classify_results_scoped_to_one_profile(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    result_a, _ = run_classify(client, environment, _cfg())
    result_b, _ = run_classify(client, environment, _cfg())

    save_classify_result(tmp_path, "profile-a", result_a)
    save_classify_result(tmp_path, "profile-b", result_b)

    assert len(list_classify_results(tmp_path)) == 2
    assert len(list_classify_results(tmp_path, profile="profile-a")) == 1
    assert list_classify_results(tmp_path, profile="profile-a")[0]["run_id"] == result_a["run_id"]


def test_run_classify_marks_invalid_when_device_contradicts_engine_claim(fake_lemonade, monkeypatch):
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": conftest.FAKE_CLASSIFY_MODEL_ID, "is_busy": False, "is_streaming": False, "device": "npu"}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_classify(client, environment, _cfg(compute_engine="cpu"))

    validate_classify_result(result)
    assert result["validity"]["valid"] is False
    assert "compute_engine claimed 'cpu' but Lemonade's own /api/v1/health reported device 'npu'" in result["validity"]["notes"]


def test_run_classify_marks_invalid_when_watchdog_reset_detected(fake_lemonade, monkeypatch):
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": conftest.FAKE_CLASSIFY_MODEL_ID, "is_busy": False, "is_streaming": False, "watchdog_reset": True}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_classify(client, environment, _cfg())

    validate_classify_result(result)
    assert result["validity"]["valid"] is False
    assert result["validity"]["model_reload_free"] is False
    assert "backend watchdog force-restarted the model process" in result["validity"]["notes"]
