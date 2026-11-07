import json

import jsonschema
import pytest
from click.testing import CliRunner

from lemonmatrix.cli import cli
from lemonmatrix.importers import LEMONADE_LAB, LEMONMETRICS, detect_source, import_result
from lemonmatrix.results_store import list_imported_results, list_results, save_imported_result
from lemonmatrix.validate import validate_imported_result


def _lab_file(results=None) -> dict:
    return {
        "timestamp": "2026-08-19T22:48:30+00:00",
        "system": {
            "cpu": "AMD Ryzen AI 9 HX 470",
            "memory": "64GB",
            "gpu": "AMD Radeon 890M",
            "npu": "AMD XDNA2 (ready)",
            "os": "Windows 11",
        },
        "results": results or {
            "llm": {
                "Llama-3.2-1B-Instruct": {
                    "cpu": {"value": 76.5, "unit": "tok/s", "ttft_ms": 146.7, "memory_gb": 9.0},
                    "vulkan": {"value": 85.7, "unit": "tok/s", "ttft_ms": 42.1, "memory_gb": 8.8},
                    "npu": {"value": 41.1, "unit": "tok/s", "ttft_ms": 254.0, "memory_gb": 10.7},
                    "hybrid": {"value": 52.4, "unit": "tok/s", "ttft_ms": 183.7, "memory_gb": 10.0},
                }
            }
        },
        "settings": {"runs": 3, "warmup": 1, "timeout": 300},
        "models": {"duplicated": "and ignored"},
        "submitted_by": "someone",
    }


def _lemonmetrics_file(with_power: bool = True) -> dict:
    efficiency = {"available": False, "note": "no power source available"}
    if with_power:
        efficiency = {
            "available": True,
            "avg_watts": 25.4,
            "baseline_watts": 6.9,
            "by_scenario": [{
                "name": "chat-short", "recipe": "llamacpp", "backend": "cpu",
                "estimated_joules_per_token": 0.289, "estimated_joules_per_output_token": 0.679,
            }],
        }
    return {
        "schema_version": "1.0",
        "lemonmetrics_version": "0.1.0",
        "generated_at": "2026-08-10T05:35:00+00:00",
        "device": {
            "cpu": {"model": "Some CPU", "physical_cores": 6},
            "gpu": ["GPU A", "GPU B"],
            "npu": [],
            "hostname": "private-laptop.local",
            "os": "Darwin",
            "memory_gb": 32.0,
        },
        "lemonade": {
            "api_base": "http://localhost:13305/api/v1",
            "cli_version": "lemonade version 11.5.2",
            "command": ["lemonade", "bench", "--secret-path", "/home/me"],
            "bench_output": {"models": [{
                "model": "Qwen3-0.6B-GGUF",
                "results": [{
                    "recipe": "llamacpp", "backend": "cpu",
                    "scenarios": [
                        {"name": "chat-short", "tps": {"mean": 48.3}, "ttft_ms": {"mean": 103.6}, "memory_peak_gb": 1.7},
                        {"name": "code-debug", "tps": {"mean": 38.0}, "ttft_ms": {"mean": 120.0}, "memory_peak_gb": 1.7},
                    ],
                }],
            }]},
        },
        "power": {"available": with_power, "samples_raw": [{"ts": 1, "watts": 6.7}]},
        "efficiency": efficiency,
    }


def _import(data: dict, source=None) -> dict:
    raw = json.dumps(data).encode()
    doc = import_result(data, raw, source)
    validate_imported_result(doc)
    return doc


def test_detect_source_recognizes_both_formats():
    assert detect_source(_lab_file()) == LEMONADE_LAB
    assert detect_source(_lemonmetrics_file()) == LEMONMETRICS
    assert detect_source({"metrics": {}}) is None


def test_lemonade_lab_import_keeps_npu_and_hybrid_rows():
    doc = _import(_lab_file())

    engines = {row["backend"]: row["compute_engine"] for row in doc["rows"]}
    assert engines == {"cpu": "cpu", "vulkan": "gpu", "npu": "npu", "hybrid": "hybrid"}
    assert doc["environment"]["memory_gb"] == 64.0
    assert doc["settings"] == {"runs": 3, "warmup": 1}
    assert doc["source"]["submitted_by"] == "someone"
    assert doc["validity"]["validated"] is False
    assert "warnings" not in doc["validity"]


def test_lemonade_lab_import_warns_when_backends_give_the_same_number():
    doc = _import(_lab_file({"image_gen": {"SD-Turbo": {
        "cpu": {"value": 6.04, "unit": "img/min"},
        "vulkan": {"value": 6.05, "unit": "img/min"},
        "rocm": {"value": 6.07, "unit": "img/min"},
    }}}))

    assert len(doc["validity"]["warnings"]) == 1
    assert "backend switch may not have taken effect" in doc["validity"]["warnings"][0]


def test_lemonmetrics_import_keeps_scenarios_and_energy():
    doc = _import(_lemonmetrics_file())

    chat, debug = doc["rows"]
    assert chat["backend"] == "llamacpp-cpu"
    assert chat["scenario"] == "chat-short"
    assert chat["joules_per_token"] == 0.289
    assert "joules_per_token" not in debug
    assert doc["power"] == {"available": True, "avg_watts": 25.4, "baseline_watts": 6.9}
    assert doc["environment"]["lemonade_version"] == "11.5.2"
    assert doc["environment"]["gpu"] == "GPU A, GPU B"
    assert "npu" not in doc["environment"]


def test_lemonmetrics_import_without_power_has_no_energy():
    doc = _import(_lemonmetrics_file(with_power=False))

    assert doc["power"] == {"available": False}
    assert all("joules_per_token" not in row for row in doc["rows"])


def test_imports_never_copy_host_names_commands_urls_or_raw_samples():
    text = json.dumps(_import(_lemonmetrics_file()))

    for private in ("private-laptop", "localhost", "/home/me", "samples_raw"):
        assert private not in text


def test_import_rejects_unknown_formats_and_empty_files():
    with pytest.raises(ValueError):
        import_result({"foo": 1}, b"{}")
    with pytest.raises(ValueError):
        import_result(_lab_file({"llm": {}}), b"{}")


def test_import_id_is_stable_for_the_same_file():
    data = _lab_file()
    raw = json.dumps(data).encode()
    assert import_result(data, raw)["import_id"] == import_result(data, raw)["import_id"]


def test_schema_refuses_an_imported_result_marked_validated():
    doc = _import(_lab_file())
    doc["validity"]["validated"] = True
    with pytest.raises(jsonschema.ValidationError):
        validate_imported_result(doc)


def test_imported_results_stay_off_the_leaderboard(tmp_path):
    save_imported_result(tmp_path, _import(_lab_file()))

    assert list_results(tmp_path) == []
    assert len(list_imported_results(tmp_path)) == 1


def test_cli_import_writes_lists_and_validates(tmp_path):
    src = tmp_path / "lab.json"
    src.write_text(json.dumps(_lab_file()))
    out = tmp_path / "results"
    runner = CliRunner()

    imported = runner.invoke(cli, ["import", str(src), "--out", str(out)])
    listed = runner.invoke(cli, ["imported-results", "--results-dir", str(out)])
    validated = runner.invoke(cli, ["validate-submission", "--results-dir", str(out)])

    assert imported.exit_code == 0, imported.output
    assert "hybrid (hybrid): 52.40 tok/s" in listed.output
    assert validated.exit_code == 0, validated.output
    assert "1 passed" in validated.output


def test_cli_import_fails_on_an_unrecognized_file(tmp_path):
    src = tmp_path / "other.json"
    src.write_text("{}")

    result = CliRunner().invoke(cli, ["import", str(src), "--out", str(tmp_path / "r")])

    assert result.exit_code == 1
    assert "SKIP" in result.output
