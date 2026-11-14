"""Omni workflow benchmark against the fake server's collection.omni model."""

import json

import conftest
import pytest
from click.testing import CliRunner

from lemonmatrix.cli import cli
from lemonmatrix.client import LemonadeClient
from lemonmatrix.omni import OMNI_TASK_SETS, OmniConfig, media_counts, run_omni, task_succeeded
from lemonmatrix.profile import discover_environment
from lemonmatrix.validate import validate_omni_result

MODEL = conftest.FAKE_OMNI_MODEL_ID


def _cfg(**overrides) -> OmniConfig:
    values = dict(model_name=MODEL, os="windows", power_state="plugged", warmup_passes=1, measured_passes=2)
    values.update(overrides)
    return OmniConfig(**values)


def _env(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    return client, environment


def test_media_counts_reads_markdown_images_and_audio_tags():
    content = "Done.\n\n![x](data:image/png;base64,AAAA)\n\n<audio>\ndata:audio/wav;base64,BBBB\n</audio>"
    assert media_counts(content) == {"image": 1, "audio": 1}
    assert media_counts("no media here") == {"image": 0, "audio": 0}


def test_task_success_needs_every_expected_medium():
    assert task_succeeded({"image": 1, "audio": 1}, {"image": 1, "audio": 1}, "x") is True
    assert task_succeeded({"image": 1, "audio": 1}, {"image": 1, "audio": 0}, "x") is False
    assert task_succeeded({}, {"image": 0, "audio": 0}, "Ottawa.") is True
    assert task_succeeded({}, {"image": 0, "audio": 0}, "   ") is False


def test_omni_run_reports_every_category(fake_lemonade):
    client, environment = _env(fake_lemonade)

    result, trials = run_omni(client, environment, _cfg())

    validate_omni_result(result)
    tasks = OMNI_TASK_SETS["omni-basic"]
    m = result["metrics"]
    assert m["trial_count"] == len(trials) == 2 * len(tasks)
    assert m["success_rate"] == 1.0
    assert set(m["by_category"]) == {"text", "image", "speech", "image+speech"}
    assert m["by_category"]["image+speech"]["images"] == 4 and m["by_category"]["image+speech"]["audio"] == 4
    assert m["tasks_per_minute"] > 0
    assert result["model"]["components"] == conftest.FAKE_OMNI_COMPONENTS
    assert result["run_type"] == "omni"


def test_a_task_that_does_not_deliver_its_medium_fails(fake_lemonade, monkeypatch):
    monkeypatch.setattr(conftest, "FAKE_OMNI_DROP", {"audio"})
    client, environment = _env(fake_lemonade)

    result, _ = run_omni(client, environment, _cfg())

    by = result["metrics"]["by_category"]
    assert by["speech"]["success_rate"] == 0.0
    assert by["image+speech"]["success_rate"] == 0.0
    assert by["image"]["success_rate"] == 1.0
    assert result["metrics"]["success_rate"] == 0.5


def test_components_are_the_runs_own_models_but_others_are_not(fake_lemonade, monkeypatch):
    import time

    import lemonmatrix.omni as omni_mod
    original = omni_mod._run_task

    def slow(client, cfg, task):
        time.sleep(0.05)
        return original(client, cfg, task)

    monkeypatch.setattr(omni_mod, "_run_task", slow)
    loaded = [{"model_name": c, "is_busy": False, "is_streaming": False, "device": "gpu"} for c in conftest.FAKE_OMNI_COMPONENTS]
    monkeypatch.setitem(conftest.FAKE_HEALTH, "all_models_loaded", loaded)
    client, environment = _env(fake_lemonade)

    clean, _ = run_omni(client, environment, _cfg())
    monkeypatch.setitem(conftest.FAKE_HEALTH, "all_models_loaded",
                        loaded + [{"model_name": "Qwen3.5-35B-A3B-GGUF", "is_busy": False, "is_streaming": False}])
    shared, _ = run_omni(client, environment, _cfg())

    assert clean["validity"]["valid"] is True
    assert shared["validity"]["valid"] is False
    assert "Qwen3.5-35B-A3B-GGUF" in shared["validity"]["notes"]


def test_a_model_that_is_not_omni_is_refused(fake_lemonade):
    client, environment = _env(fake_lemonade)
    with pytest.raises(ValueError, match="collection.omni"):
        run_omni(client, environment, _cfg(model_name="Llama-3.1-8B-Instruct-GGUF"))


def test_cli_omni_writes_a_result_that_passes_submission_validation(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(cli, ["profile", "add", "box", "--url", fake_lemonade]).exit_code == 0

    ran = runner.invoke(cli, ["omni", "--profile", "box", "--model", MODEL, "--power-state", "plugged",
                              "--passes", "1", "--out", str(tmp_path / "results")])
    listed = runner.invoke(cli, ["omni-results", "--results-dir", str(tmp_path / "results")])
    checked = runner.invoke(cli, ["validate-submission", "--results-dir", str(tmp_path / "results")])

    assert ran.exit_code == 0, ran.output
    assert "success: 100%" in ran.output
    assert MODEL in listed.output
    assert checked.exit_code == 0, checked.output
    files = list((tmp_path / "results" / "box" / "omni").glob("*.json"))
    assert len(files) == 1 and json.loads(files[0].read_text())["run_type"] == "omni"
