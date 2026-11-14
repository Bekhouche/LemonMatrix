"""Idle models left loaded on the instance share the device's memory: the run
records them (validity.other_models_resident and a note) and is not ranked, and
--unload-others unloads everything before loading the model under test."""

import conftest
from click.testing import CliRunner

from lemonmatrix.bench import SweepConfig, run_sweep
from lemonmatrix.cli import cli
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.validate import validate_result

OWN = "Llama-3.1-8B-Instruct-GGUF"


def _cfg(**overrides) -> SweepConfig:
    values = dict(
        model_name=OWN, model_class="dense", quantization="Q4_K_M", context_length=4096,
        compute_engine="igpu", backend="llama.cpp-vulkan", os="windows", power_state="plugged",
        warmup_trials=1, measured_trials=2,
    )
    values.update(overrides)
    return SweepConfig(**values)


def _loaded(monkeypatch, *others: str) -> None:
    entries = [{"model_name": OWN, "is_busy": False, "is_streaming": False, "device": "gpu"}]
    entries += [{"model_name": name, "is_busy": False, "is_streaming": False, "device": "gpu"} for name in others]
    monkeypatch.setitem(conftest.FAKE_HEALTH, "all_models_loaded", entries)


def _slow_trials(monkeypatch):
    """Give the health monitor time to poll at least once."""
    import time

    import lemonmatrix.bench as bench_mod
    original = bench_mod._run_one

    def slow(client, cfg):
        time.sleep(0.4)
        return original(client, cfg)

    monkeypatch.setattr(bench_mod, "_run_one", slow)


def test_an_idle_model_left_loaded_makes_the_run_invalid(fake_lemonade, monkeypatch):
    _loaded(monkeypatch, "Qwen3.5-35B-A3B-GGUF")
    _slow_trials(monkeypatch)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg())

    validate_result(result)
    assert result["validity"]["other_models_resident"] is True
    assert result["validity"]["valid"] is False
    assert "other models stayed loaded during measurement" in result["validity"]["notes"]
    assert "--unload-others" in result["validity"]["notes"]
    assert "Qwen3.5-35B-A3B-GGUF" in result["validity"]["notes"]


def test_a_run_alone_on_the_instance_says_so(fake_lemonade, monkeypatch):
    _loaded(monkeypatch)
    _slow_trials(monkeypatch)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg())

    assert result["validity"]["other_models_resident"] is False
    assert result["validity"]["valid"] is True
    assert "stayed loaded" not in result["validity"].get("notes", "")


def test_a_busy_model_is_reported_as_competing_not_as_merely_resident(fake_lemonade, monkeypatch):
    monkeypatch.setitem(conftest.FAKE_HEALTH, "all_models_loaded", [
        {"model_name": OWN, "is_busy": False, "is_streaming": False, "device": "gpu"},
        {"model_name": "Other-GGUF", "is_busy": True, "is_streaming": False, "device": "gpu"},
    ])
    _slow_trials(monkeypatch)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg())

    assert result["validity"]["exclusive_run"] is False
    assert "stayed loaded" not in result["validity"]["notes"]


def _spy_unload(monkeypatch) -> list:
    calls = []
    original = LemonadeClient.unload

    def unload(self, model_name=None):
        calls.append(model_name)
        return original(self, model_name)

    monkeypatch.setattr(LemonadeClient, "unload", unload)
    return calls


def test_unload_others_unloads_everything_before_loading(fake_lemonade, monkeypatch):
    calls = _spy_unload(monkeypatch)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_sweep(client, environment, _cfg(unload_others=True))

    assert calls == [None]


def test_without_the_flag_nothing_is_unloaded(fake_lemonade, monkeypatch):
    calls = _spy_unload(monkeypatch)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_sweep(client, environment, _cfg())

    assert calls == []


def test_cli_run_passes_unload_others(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = _spy_unload(monkeypatch)
    runner = CliRunner()
    assert runner.invoke(cli, ["profile", "add", "box", "--url", fake_lemonade]).exit_code == 0

    result = runner.invoke(cli, [
        "run", "--profile", "box", "--model", OWN, "--model-class", "dense", "--quant", "Q4_K_M",
        "--context-length", "4096", "--engine", "igpu", "--backend", "llama.cpp-vulkan",
        "--power-state", "plugged", "--warmup", "1", "--trials", "2", "--unload-others",
    ])

    assert result.exit_code == 0, result.output
    assert calls == [None]
