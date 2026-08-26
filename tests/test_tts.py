import conftest
import pytest

from lemonmatrix.bench import TTSConfig, run_tts
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.results_store import list_results, list_tts_results, save_tts_result
from lemonmatrix.validate import validate_tts_result


def _cfg(**overrides) -> TTSConfig:
    defaults = dict(
        model_name=conftest.FAKE_TTS_MODEL_ID,
        compute_engine="cpu",
        backend="kokoro-cpu",
        os="windows",
        power_state="plugged",
        input_text="Lemonade can speak",
        warmup_trials=1,
        measured_trials=2,
    )
    defaults.update(overrides)
    return TTSConfig(**defaults)


def test_run_tts_produces_schema_conformant_result(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_tts(client, environment, _cfg())

    validate_tts_result(result)  # raises on any schema violation
    assert result["run_type"] == "tts"
    assert result["model"]["name"] == conftest.FAKE_TTS_MODEL_ID
    assert result["model"]["checkpoint"] == conftest.FAKE_TTS_CHECKPOINT
    assert result["metrics"]["trial_count"] == 2
    assert result["metrics"]["audio_duration_s"] == pytest.approx(conftest.FAKE_TTS_WAV_SECONDS)
    assert result["metrics"]["generation_time_ms"] > 0
    assert result["metrics"]["real_time_factor"] == pytest.approx(
        conftest.FAKE_TTS_WAV_SECONDS / (result["metrics"]["generation_time_ms"] / 1000)
    )
    assert result["metrics"]["input_chars"] == len(_cfg().input_text)
    assert result["validity"]["model_reload_free"] is True
    assert result["validity"]["warmup_discarded"] is True
    # kokoro/cpu is "installed" in the fake server's recipes tree.
    assert result["environment"]["backend_version"] == "b17"
    assert len(trials) == 2
    assert all("generation_time_ms" in t and "audio_duration_s" in t for t in trials)


def test_run_tts_records_voice_and_speed(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_tts(client, environment, _cfg(voice="onyx", speed=0.75))

    validate_tts_result(result)
    assert result["metrics"]["voice"] == "onyx"
    assert result["metrics"]["speed"] == 0.75


def test_run_tts_marks_invalid_when_engine_backend_incompatible(fake_lemonade):
    # npu engine + kokoro-cpu backend is a physical contradiction: kokoro on
    # this instance only has a cpu backend.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_tts(client, environment, _cfg(compute_engine="npu"))

    validate_tts_result(result)
    assert result["validity"]["valid"] is False
    assert "engine/backend mismatch" in result["validity"]["notes"]


def test_run_tts_detects_competing_model_via_health_polling(fake_lemonade, monkeypatch):
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "someone-elses-model", "is_busy": True, "is_streaming": False}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_tts(client, environment, _cfg(exclusive_run=True))

    validate_tts_result(result)
    assert result["validity"]["exclusive_run"] is False
    assert result["validity"]["valid"] is False
    assert "competing workload detected during measurement: someone-elses-model" in result["validity"]["notes"]


def test_save_and_list_tts_results_is_isolated_from_list_results(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    result, _ = run_tts(client, environment, _cfg())

    save_tts_result(tmp_path, "demo-profile", result)

    tts_results = list_tts_results(tmp_path)
    assert len(tts_results) == 1
    assert tts_results[0]["run_id"] == result["run_id"]
    assert tts_results[0]["_profile"] == "demo-profile"

    # The main leaderboard's one-level-deep glob must never see this file --
    # it doesn't conform to result.schema.json.
    assert list_results(tmp_path) == []
