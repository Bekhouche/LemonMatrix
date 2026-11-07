import conftest
import pytest
from conftest import get_fake_server

from lemonmatrix.bench import AudioGenConfig, run_audiogen
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.results_store import list_audiogen_results, list_results, save_audiogen_result
from lemonmatrix.validate import validate_audiogen_result


def _cfg(**overrides) -> AudioGenConfig:
    defaults = dict(
        model_name=conftest.FAKE_AUDIOGEN_MODEL_ID,
        compute_engine="dgpu",
        backend="acestep-cuda",
        os="windows",
        power_state="plugged",
        prompt="An upbeat acoustic guitar riff",
        warmup_trials=1,
        measured_trials=2,
    )
    defaults.update(overrides)
    return AudioGenConfig(**defaults)


def test_run_audiogen_produces_schema_conformant_result(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_audiogen(client, environment, _cfg())

    validate_audiogen_result(result)  # raises on any schema violation
    assert result["run_type"] == "audiogen"
    assert result["model"]["name"] == conftest.FAKE_AUDIOGEN_MODEL_ID
    assert result["model"]["checkpoint"] == conftest.FAKE_AUDIOGEN_CHECKPOINT
    assert result["metrics"]["trial_count"] == 2
    assert result["metrics"]["audio_duration_s"] == pytest.approx(conftest.FAKE_AUDIOGEN_WAV_SECONDS)
    assert result["metrics"]["generation_time_ms"] > 0
    assert result["metrics"]["real_time_factor"] == pytest.approx(
        conftest.FAKE_AUDIOGEN_WAV_SECONDS / (result["metrics"]["generation_time_ms"] / 1000)
    )
    assert result["metrics"]["prompt_chars"] == len(_cfg().prompt)
    assert result["validity"]["model_reload_free"] is True
    assert result["validity"]["warmup_discarded"] is True
    # acestep/cuda is "installed" in the fake server's recipes tree.
    assert result["environment"]["backend_version"] == "1.0.0"
    assert len(trials) == 2
    assert all("generation_time_ms" in t and "audio_duration_s" in t for t in trials)


def test_run_audiogen_records_lyrics_and_vocal_language(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_audiogen(client, environment, _cfg(lyrics="la la la", vocal_language="en"))

    validate_audiogen_result(result)
    assert result["metrics"]["lyrics_chars"] == len("la la la")
    assert result["metrics"]["vocal_language"] == "en"


def test_run_audiogen_tells_lemonade_which_backend_to_load(fake_lemonade):
    # Regression test: acestep reports "selectable_backend": true with
    # multiple backends -- without threading cfg.backend through to
    # /api/v1/load, Lemonade would silently fall back to its recipe's
    # default backend regardless of what was actually requested (the same
    # bug class already fixed for run_sweep/run_stt/run_imagegen).
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_audiogen(client, environment, _cfg(backend="acestep-vulkan"))

    payload = get_fake_server(fake_lemonade).last_load_payload
    assert payload["acestep_backend"] == "vulkan"


def test_run_audiogen_notes_and_invalidates_when_backend_unresolvable(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_audiogen(client, environment, _cfg(backend="totally-not-a-real-backend-string"))

    validate_audiogen_result(result)
    assert result["validity"]["valid"] is False
    assert "could not resolve backend" in result["validity"]["notes"]


def test_run_audiogen_marks_invalid_when_engine_backend_incompatible(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_audiogen(client, environment, _cfg(compute_engine="npu"))

    validate_audiogen_result(result)
    assert result["validity"]["valid"] is False
    assert "engine/backend mismatch" in result["validity"]["notes"]


def test_run_audiogen_detects_competing_model_via_health_polling(fake_lemonade, monkeypatch):
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "someone-elses-model", "is_busy": True, "is_streaming": False}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_audiogen(client, environment, _cfg(exclusive_run=True))

    validate_audiogen_result(result)
    assert result["validity"]["exclusive_run"] is False
    assert result["validity"]["valid"] is False
    assert "competing workload detected during measurement: someone-elses-model" in result["validity"]["notes"]


def test_save_and_list_audiogen_results_is_isolated_from_list_results(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    result, _ = run_audiogen(client, environment, _cfg())

    save_audiogen_result(tmp_path, "demo-profile", result)

    audiogen_results = list_audiogen_results(tmp_path)
    assert len(audiogen_results) == 1
    assert audiogen_results[0]["run_id"] == result["run_id"]
    assert audiogen_results[0]["_profile"] == "demo-profile"

    # The main leaderboard's one-level-deep glob must never see this file --
    # it doesn't conform to result.schema.json.
    assert list_results(tmp_path) == []
