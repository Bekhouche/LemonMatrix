import conftest
import pytest

from conftest import get_fake_server

from lemonmatrix.bench import ImageGenConfig, run_imagegen
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.results_store import list_imagegen_results, list_results, save_imagegen_result
from lemonmatrix.validate import validate_imagegen_result


def _cfg(**overrides) -> ImageGenConfig:
    defaults = dict(
        model_name=conftest.FAKE_IMAGEGEN_MODEL_ID,
        compute_engine="cpu",
        backend="sd-cpp-cpu",
        os="windows",
        power_state="plugged",
        prompt="A red circle",
        warmup_trials=1,
        measured_trials=2,
    )
    defaults.update(overrides)
    return ImageGenConfig(**defaults)


def test_run_imagegen_produces_schema_conformant_result(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_imagegen(client, environment, _cfg())

    validate_imagegen_result(result)  # raises on any schema violation
    assert result["run_type"] == "imagegen"
    assert result["model"]["name"] == conftest.FAKE_IMAGEGEN_MODEL_ID
    assert result["model"]["checkpoint"] == conftest.FAKE_IMAGEGEN_CHECKPOINT
    assert result["metrics"]["trial_count"] == 2
    assert result["metrics"]["image_size"] == "512x512"
    assert result["metrics"]["steps"] == 4
    assert result["metrics"]["generation_time_ms"] > 0
    assert result["metrics"]["images_per_sec"] == pytest.approx(
        1000 / result["metrics"]["generation_time_ms"]
    )
    assert result["metrics"]["prompt_chars"] == len("A red circle")
    assert result["validity"]["model_reload_free"] is True
    assert result["validity"]["warmup_discarded"] is True
    # sd-cpp/cpu is "installed" in the fake server's recipes tree.
    assert result["environment"]["backend_version"] == "0.2.0"
    assert len(trials) == 2
    assert all("generation_time_ms" in t for t in trials)


def test_run_imagegen_records_cfg_scale_and_seed(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_imagegen(client, environment, _cfg(cfg_scale=1.5, seed=42))

    validate_imagegen_result(result)
    assert result["metrics"]["cfg_scale"] == 1.5
    assert result["metrics"]["seed"] == 42


def test_run_imagegen_tells_lemonade_which_backend_to_load(fake_lemonade):
    # Regression test: sd-cpp reports "selectable_backend": true with
    # multiple backends (confirmed live: cpu/cuda/metal/rocm/vulkan) and a
    # "cpu" default -- without threading cfg.backend through to
    # /api/v1/load, Lemonade would silently fall back to cpu regardless of
    # what backend was actually requested.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_imagegen(client, environment, _cfg(compute_engine="dgpu", backend="sd-cpp-cuda"))

    payload = get_fake_server(fake_lemonade).last_load_payload
    assert payload["sd-cpp_backend"] == "cuda"


def test_run_imagegen_notes_and_invalidates_when_backend_unresolvable(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_imagegen(client, environment, _cfg(backend="totally-not-a-real-backend-string"))

    validate_imagegen_result(result)
    assert result["validity"]["valid"] is False
    assert "could not resolve backend" in result["validity"]["notes"]


def test_run_imagegen_marks_invalid_when_engine_backend_incompatible(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_imagegen(client, environment, _cfg(compute_engine="npu"))

    validate_imagegen_result(result)
    assert result["validity"]["valid"] is False
    assert "engine/backend mismatch" in result["validity"]["notes"]


def test_run_imagegen_detects_competing_model_via_health_polling(fake_lemonade, monkeypatch):
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "someone-elses-model", "is_busy": True, "is_streaming": False}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_imagegen(client, environment, _cfg(exclusive_run=True))

    validate_imagegen_result(result)
    assert result["validity"]["exclusive_run"] is False
    assert result["validity"]["valid"] is False
    assert "competing workload detected during measurement: someone-elses-model" in result["validity"]["notes"]


def test_save_and_list_imagegen_results_is_isolated_from_list_results(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    result, _ = run_imagegen(client, environment, _cfg())

    save_imagegen_result(tmp_path, "demo-profile", result)

    imagegen_results = list_imagegen_results(tmp_path)
    assert len(imagegen_results) == 1
    assert imagegen_results[0]["run_id"] == result["run_id"]
    assert imagegen_results[0]["_profile"] == "demo-profile"

    # The main leaderboard's one-level-deep glob must never see this file --
    # it doesn't conform to result.schema.json.
    assert list_results(tmp_path) == []


FAKE_INPUT_IMAGE_BYTES = b"\x89PNG\r\n\x1a\nnot a real png but the fake server never decodes it"


def test_run_imagegen_edit_operation(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_imagegen(
        client, environment,
        _cfg(operation="edit", prompt="add a hat", input_image_bytes=FAKE_INPUT_IMAGE_BYTES),
    )

    validate_imagegen_result(result)
    assert result["metrics"]["operation"] == "edit"
    assert result["metrics"]["has_mask"] is False
    assert result["metrics"]["prompt_chars"] == len("add a hat")
    assert "steps" in result["metrics"]
    assert len(trials) == 2


def test_run_imagegen_edit_with_mask(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_imagegen(
        client, environment,
        _cfg(
            operation="edit", prompt="add a hat", input_image_bytes=FAKE_INPUT_IMAGE_BYTES,
            mask_bytes=FAKE_INPUT_IMAGE_BYTES,
        ),
    )

    validate_imagegen_result(result)
    assert result["metrics"]["has_mask"] is True


def test_run_imagegen_variation_operation_omits_prompt_and_steps(fake_lemonade):
    # Confirmed against Lemonade's own server source: /v1/images/variations
    # doesn't accept a prompt, steps, cfg_scale, or seed at all -- reporting
    # them would misrepresent what was actually requested.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_imagegen(
        client, environment,
        _cfg(operation="variation", input_image_bytes=FAKE_INPUT_IMAGE_BYTES),
    )

    validate_imagegen_result(result)
    assert result["metrics"]["operation"] == "variation"
    assert "prompt_chars" not in result["metrics"]
    assert "steps" not in result["metrics"]
    assert "cfg_scale" not in result["metrics"]
    assert "seed" not in result["metrics"]
    assert "has_mask" not in result["metrics"]
    assert len(trials) == 2
