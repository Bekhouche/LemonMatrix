"""Ryzen AI NPU and hybrid runs: ryzenai-llm fixes the engine by model name
(-Hybrid / -NPU / -CPU, all on its one "npu" backend), flm always runs on the
NPU, and Lemonade reports device "npu" for every ryzenai-llm model."""

import json

import pytest
from click.testing import CliRunner

import conftest
from lemonmatrix.bench import SweepConfig, _device_matches_engine, run_sweep
from lemonmatrix.capabilities import (
    compatible_engines_for_backend,
    engine_backend_compatible,
    expected_engine_for_model,
    validate_combo_against_profile,
)
from lemonmatrix.cli import cli
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.validate import validate_result

HYBRID = "Llama-3.2-1B-Instruct-Hybrid"
NPU = "Llama-3.2-1B-Instruct-NPU"
CPU = "Llama-3.2-1B-Instruct-CPU"
BACKEND = "ryzenai-llm-npu"


@pytest.mark.parametrize(
    "model,backend,expected",
    [
        (HYBRID, BACKEND, "hybrid"),
        (NPU, BACKEND, "npu"),
        (CPU, BACKEND, "cpu"),
        ("user.Phi-4-mini-instruct-Hybrid", BACKEND, "hybrid"),
        ("Some-Other-Model", BACKEND, None),  # unknown deployment: the caller decides
        ("Llama-3.2-1B-FLM", "flm-npu", "npu"),
        ("Qwen3-1.7B-GGUF", "llamacpp-vulkan", None),
        ("Llama-3.2-1B-CPU", "llamacpp-cpu", None),  # suffix only counts for ryzenai-llm
        (None, BACKEND, None),
    ],
)
def test_expected_engine_for_model(model, backend, expected):
    assert expected_engine_for_model(model, backend) == expected


def test_ryzenai_engine_must_match_the_model_suffix():
    assert engine_backend_compatible("hybrid", BACKEND, HYBRID) == (True, "")
    assert engine_backend_compatible("cpu", BACKEND, CPU) == (True, "")

    ok, reason = engine_backend_compatible("npu", BACKEND, CPU)
    assert ok is False
    assert "runs this model on 'cpu'" in reason
    assert "--engine cpu" in reason
    assert engine_backend_compatible("npu", BACKEND, HYBRID)[0] is False


def test_ryzenai_without_a_model_name_allows_its_three_deployments():
    for engine in ("cpu", "npu", "hybrid"):
        assert engine_backend_compatible(engine, BACKEND)[0] is True
    assert engine_backend_compatible("dgpu", BACKEND)[0] is False
    assert compatible_engines_for_backend(BACKEND) == ["cpu", "hybrid", "npu"]
    assert compatible_engines_for_backend(BACKEND, CPU) == ["cpu"]


def test_other_backends_keep_the_backend_table():
    assert engine_backend_compatible("dgpu", "llamacpp-vulkan", "Qwen3-1.7B-GGUF")[0] is True
    assert engine_backend_compatible("cpu", "llamacpp-vulkan", "Qwen3-1.7B-GGUF")[0] is False
    assert engine_backend_compatible("npu", "flm-npu", "Llama-3.2-1B-FLM")[0] is True
    assert engine_backend_compatible("cpu", "flm-npu", "Llama-3.2-1B-FLM")[0] is False


@pytest.mark.parametrize(
    "device,engine,backend,expected",
    [
        ("npu", "hybrid", BACKEND, None),  # ryzenai-llm reports npu for every model
        ("npu", "cpu", BACKEND, None),
        ("npu", "npu", BACKEND, True),
        ("gpu", "npu", BACKEND, False),
        ("npu", "npu", "flm-npu", True),
        ("cpu", "npu", "flm-npu", False),
        ("gpu", "hybrid", "llamacpp-vulkan", None),  # other recipes: hybrid stays unverified
        ("gpu", "dgpu", "llamacpp-vulkan", True),
    ],
)
def test_device_check_knows_ryzenai_always_reports_npu(device, engine, backend, expected):
    assert _device_matches_engine(device, engine, backend) is expected


def _system_info(npu_available: bool) -> dict:
    info = json.loads(json.dumps(conftest.FAKE_SYSTEM_INFO))
    info["amd_npu"]["available"] = npu_available
    info["recipes"]["ryzenai-llm"] = _ryzenai_recipe()
    return info


def _ryzenai_recipe() -> dict:
    return {
        "display_name": "Ryzen AI LLM",
        "modality": "Text generation",
        "backends": {"npu": {"state": "installed", "version": "v1.7.0"}},
    }


def test_ryzenai_hybrid_needs_an_npu_on_the_host():
    env = {"os_version": "Windows 11 Pro 24H2"}

    without = validate_combo_against_profile("hybrid", BACKEND, _system_info(False), env, model=HYBRID)
    with_npu = validate_combo_against_profile("hybrid", BACKEND, _system_info(True), env, model=HYBRID)

    assert any("need an NPU" in issue for issue in without)
    assert with_npu == []


@pytest.fixture
def ryzenai_server(fake_lemonade, monkeypatch):
    """The fake server with ryzenai-llm installed and its three models pulled."""
    monkeypatch.setitem(conftest.FAKE_SYSTEM_INFO["recipes"], "ryzenai-llm", _ryzenai_recipe())
    models = [
        {"id": name, "object": "model", "recipe": "ryzenai-llm", "checkpoint": f"amd/{name}-onnx",
         "max_context_window": 4096}
        for name in (HYBRID, NPU, CPU)
    ]
    monkeypatch.setitem(conftest.FAKE_MODELS, "data", conftest.FAKE_MODELS["data"] + models)
    return fake_lemonade


def _loaded(monkeypatch, model: str) -> None:
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": model, "is_busy": False, "is_streaming": False, "device": "npu"}],
    )


def _cfg(model: str, engine: str) -> SweepConfig:
    return SweepConfig(
        model_name=model, model_class="dense", quantization="none", context_length=4096,
        compute_engine=engine, backend=BACKEND, os="windows", power_state="plugged",
        warmup_trials=1, measured_trials=2,
    )


def test_hybrid_run_is_valid_and_says_why_the_device_was_not_used(ryzenai_server, monkeypatch):
    _loaded(monkeypatch, HYBRID)
    client = LemonadeClient(ryzenai_server)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(HYBRID, "hybrid"))

    validate_result(result)
    assert result["validity"]["valid"] is True, result["validity"]
    assert "device not verifiable" in result["validity"]["notes"]
    assert result["config"]["compute_engine"] == "hybrid"


def test_cpu_model_claimed_as_npu_is_invalid(ryzenai_server, monkeypatch):
    _loaded(monkeypatch, CPU)
    client = LemonadeClient(ryzenai_server)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(CPU, "npu"))

    validate_result(result)
    assert result["validity"]["valid"] is False
    assert "runs this model on 'cpu'" in result["validity"]["notes"]


def test_sweep_auto_engine_follows_the_model(ryzenai_server, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(cli, ["profile", "add", "laptop", "--url", ryzenai_server]).exit_code == 0

    result = runner.invoke(cli, [
        "sweep", "--profile", "laptop", "--model", HYBRID, "--model", NPU, "--model", CPU,
        "--backend", BACKEND, "--warmup", "1", "--trials", "2", "--out", str(tmp_path / "results"),
    ])

    assert result.exit_code == 0, result.output
    assert "invalid combo" not in result.output
    engines = {
        json.loads(p.read_text())["model"]["name"]: json.loads(p.read_text())["config"]["compute_engine"]
        for p in (tmp_path / "results" / "laptop").glob("*.json")
    }
    assert engines == {HYBRID: "hybrid", NPU: "npu", CPU: "cpu"}
