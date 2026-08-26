import conftest
import pytest

from conftest import get_fake_server

from lemonmatrix.bench import STTConfig, run_stt
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.results_store import list_results, list_stt_results, save_stt_result
from lemonmatrix.validate import validate_stt_result

FAKE_AUDIO_SECONDS = 3.0
FAKE_AUDIO_BYTES = conftest._make_wav_bytes(FAKE_AUDIO_SECONDS)


def _cfg(**overrides) -> STTConfig:
    defaults = dict(
        model_name=conftest.FAKE_STT_MODEL_ID,
        compute_engine="cpu",
        backend="whispercpp-cpu",
        os="windows",
        power_state="plugged",
        audio_bytes=FAKE_AUDIO_BYTES,
        warmup_trials=1,
        measured_trials=2,
    )
    defaults.update(overrides)
    return STTConfig(**defaults)


def test_run_stt_produces_schema_conformant_result(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_stt(client, environment, _cfg())

    validate_stt_result(result)  # raises on any schema violation
    assert result["run_type"] == "stt"
    assert result["model"]["name"] == conftest.FAKE_STT_MODEL_ID
    assert result["model"]["checkpoint"] == conftest.FAKE_STT_CHECKPOINT
    assert result["metrics"]["trial_count"] == 2
    assert result["metrics"]["audio_duration_s"] == pytest.approx(FAKE_AUDIO_SECONDS)
    assert result["metrics"]["transcription_time_ms"] > 0
    assert result["metrics"]["real_time_factor"] == pytest.approx(
        FAKE_AUDIO_SECONDS / (result["metrics"]["transcription_time_ms"] / 1000)
    )
    assert result["validity"]["model_reload_free"] is True
    assert result["validity"]["warmup_discarded"] is True
    # whispercpp/cpu is "installed" in the fake server's recipes tree.
    assert result["environment"]["backend_version"] == "1.8.4"
    assert len(trials) == 2
    assert all("transcription_time_ms" in t for t in trials)


def test_run_stt_records_language(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_stt(client, environment, _cfg(language="en"))

    validate_stt_result(result)
    assert result["metrics"]["language"] == "en"


def test_run_stt_tells_lemonade_which_backend_to_load(fake_lemonade):
    # Regression test: whispercpp reports "selectable_backend": true with
    # multiple backends (confirmed live: cpu/metal/npu/rocm/vulkan) and a
    # "cpu" default -- without threading cfg.backend through to
    # /api/v1/load, Lemonade would silently fall back to cpu regardless of
    # what backend was actually requested.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_stt(client, environment, _cfg(compute_engine="igpu", backend="whispercpp-vulkan"))

    payload = get_fake_server(fake_lemonade).last_load_payload
    assert payload["whispercpp_backend"] == "vulkan"


def test_run_stt_notes_and_invalidates_when_backend_unresolvable(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_stt(client, environment, _cfg(backend="totally-not-a-real-backend-string"))

    validate_stt_result(result)
    assert result["validity"]["valid"] is False
    assert "could not resolve backend" in result["validity"]["notes"]


def test_run_stt_marks_invalid_when_engine_backend_incompatible(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_stt(client, environment, _cfg(compute_engine="npu"))

    validate_stt_result(result)
    assert result["validity"]["valid"] is False
    assert "engine/backend mismatch" in result["validity"]["notes"]


def test_run_stt_detects_competing_model_via_health_polling(fake_lemonade, monkeypatch):
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "someone-elses-model", "is_busy": True, "is_streaming": False}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_stt(client, environment, _cfg(exclusive_run=True))

    validate_stt_result(result)
    assert result["validity"]["exclusive_run"] is False
    assert result["validity"]["valid"] is False
    assert "competing workload detected during measurement: someone-elses-model" in result["validity"]["notes"]


def test_save_and_list_stt_results_is_isolated_from_list_results(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    result, _ = run_stt(client, environment, _cfg())

    save_stt_result(tmp_path, "demo-profile", result)

    stt_results = list_stt_results(tmp_path)
    assert len(stt_results) == 1
    assert stt_results[0]["run_id"] == result["run_id"]
    assert stt_results[0]["_profile"] == "demo-profile"

    # The main leaderboard's one-level-deep glob must never see this file --
    # it doesn't conform to result.schema.json.
    assert list_results(tmp_path) == []
