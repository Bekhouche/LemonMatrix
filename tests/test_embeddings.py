import conftest
import pytest
from conftest import get_fake_server

from lemonmatrix.bench import EmbeddingsConfig, run_embeddings
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.results_store import list_embeddings_results, list_results, save_embeddings_result
from lemonmatrix.validate import validate_embeddings_result


def _cfg(**overrides) -> EmbeddingsConfig:
    defaults = dict(
        model_name=conftest.FAKE_EMBED_MODEL_ID,
        compute_engine="cpu",
        backend="llama.cpp-cpu",
        os="windows",
        power_state="plugged",
        input_texts=["hello world", "a second sentence", "a third one"],
        warmup_trials=1,
        measured_trials=2,
    )
    defaults.update(overrides)
    return EmbeddingsConfig(**defaults)


def test_run_embeddings_produces_schema_conformant_result(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_embeddings(client, environment, _cfg())

    validate_embeddings_result(result)  # raises on any schema violation
    assert result["run_type"] == "embeddings"
    assert result["model"]["name"] == conftest.FAKE_EMBED_MODEL_ID
    assert result["model"]["checkpoint"] == conftest.FAKE_EMBED_CHECKPOINT
    assert result["metrics"]["trial_count"] == 2
    assert result["metrics"]["batch_size"] == 3
    assert result["metrics"]["embedding_dim"] == conftest.FAKE_EMBEDDING_DIM
    assert result["metrics"]["latency_ms"] > 0
    assert result["metrics"]["embeddings_per_sec"] == pytest.approx(
        3 / (result["metrics"]["latency_ms"] / 1000)
    )
    assert result["metrics"]["input_chars_total"] == sum(len(t) for t in _cfg().input_texts)
    assert result["validity"]["model_reload_free"] is True
    assert result["validity"]["warmup_discarded"] is True
    # llamacpp/cpu is "installed" in the fake server's recipes tree.
    assert result["environment"]["backend_version"] == "b10375"
    assert len(trials) == 2
    assert all("latency_ms" in t for t in trials)


def test_run_embeddings_tells_lemonade_which_backend_to_load(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_embeddings(client, environment, _cfg(backend="llama.cpp-vulkan"))

    payload = get_fake_server(fake_lemonade).last_load_payload
    assert payload["llamacpp_backend"] == "vulkan"


def test_run_embeddings_marks_invalid_when_engine_backend_incompatible(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_embeddings(client, environment, _cfg(compute_engine="npu"))

    validate_embeddings_result(result)
    assert result["validity"]["valid"] is False
    assert "engine/backend mismatch" in result["validity"]["notes"]


def test_run_embeddings_detects_competing_model_via_health_polling(fake_lemonade, monkeypatch):
    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "someone-elses-model", "is_busy": True, "is_streaming": False}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_embeddings(client, environment, _cfg(exclusive_run=True))

    validate_embeddings_result(result)
    assert result["validity"]["exclusive_run"] is False
    assert result["validity"]["valid"] is False
    assert "competing workload detected during measurement: someone-elses-model" in result["validity"]["notes"]


def test_save_and_list_embeddings_results_is_isolated_from_list_results(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    result, _ = run_embeddings(client, environment, _cfg())

    save_embeddings_result(tmp_path, "demo-profile", result)

    embeddings_results = list_embeddings_results(tmp_path)
    assert len(embeddings_results) == 1
    assert embeddings_results[0]["run_id"] == result["run_id"]
    assert embeddings_results[0]["_profile"] == "demo-profile"

    # The main leaderboard's one-level-deep glob must never see this file --
    # it doesn't conform to result.schema.json.
    assert list_results(tmp_path) == []
