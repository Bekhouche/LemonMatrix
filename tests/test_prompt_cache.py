"""Trials repeat one prompt, so llama.cpp's prompt cache would turn every
trial after the first into a cache hit: TTFT and prefill would time the cache,
not the prompt. Model runs send cache_prompt: false, and a run whose measured
trials still report cached prompt tokens is invalid."""

import conftest
from conftest import get_fake_server

from lemonmatrix.bench import SweepConfig, _build_sweep_job_steps, _prompt_cache_verdict, run_sweep, run_sweep_via_job
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.validate import validate_result


def _cfg(**overrides) -> SweepConfig:
    defaults = dict(
        model_name="Llama-3.1-8B-Instruct-GGUF", model_class="dense", quantization="Q4_K_M",
        context_length=4096, compute_engine="igpu", backend="llama.cpp-vulkan", os="windows",
        power_state="plugged", warmup_trials=1, measured_trials=2,
    )
    defaults.update(overrides)
    return SweepConfig(**defaults)


def _capture_chat(monkeypatch) -> list[dict]:
    calls: list[dict] = []
    original = LemonadeClient.chat_completion

    def spy(self, model, messages, max_tokens, **kwargs):
        calls.append(kwargs)
        return original(self, model, messages, max_tokens, **kwargs)

    monkeypatch.setattr(LemonadeClient, "chat_completion", spy)
    return calls


def test_model_runs_ask_the_server_not_to_reuse_the_prompt_cache(fake_lemonade, monkeypatch):
    calls = _capture_chat(monkeypatch)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_sweep(client, environment, _cfg())

    assert calls and all(c.get("cache_prompt") is False for c in calls)


def test_router_runs_keep_the_default_request(fake_lemonade, monkeypatch):
    calls = _capture_chat(monkeypatch)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_sweep(client, environment, _cfg(
        run_type="router", model_name=conftest.FAKE_ROUTER_MODEL_ID, model_class="router",
        quantization="none", compute_engine="router", backend="collection.router",
    ))

    assert calls and all("cache_prompt" not in c for c in calls)


def test_job_engine_chat_steps_disable_the_prompt_cache():
    steps = _build_sweep_job_steps(_cfg(warmup_trials=2, measured_trials=3), {})
    chats = [s for s in steps if s["op"] == "chat"]
    assert len(chats) == 5
    assert all(s["params"]["cache_prompt"] is False for s in chats)


def test_prompt_cache_verdict():
    assert _prompt_cache_verdict([{"cache_tokens": 0}, {"cache_tokens": 0}]) == (True, None)
    ok, note = _prompt_cache_verdict([{"cache_tokens": 0}, {"cache_tokens": 23}])
    assert ok is False and "1 of 2 measured trials" in note and "23 prompt tokens" in note
    ok, note = _prompt_cache_verdict([{"cache_tokens": None}, {}])
    assert ok is True and "not reported" in note


def test_a_run_whose_trials_hit_the_cache_is_invalid(fake_lemonade, monkeypatch):
    monkeypatch.setitem(conftest.FAKE_STATS, "cache_tokens", 39)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_sweep(client, environment, _cfg())

    validate_result(result)
    assert result["validity"]["valid"] is False
    assert "prompt cache reused on 2 of 2" in result["validity"]["notes"]
    assert [t["cache_tokens"] for t in trials] == [39, 39]


def test_a_run_with_no_cache_hits_stays_valid(fake_lemonade, monkeypatch):
    monkeypatch.setitem(conftest.FAKE_STATS, "cache_tokens", 0)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg())

    assert result["validity"]["valid"] is True, result["validity"]
    assert "prompt cache" not in result["validity"].get("notes", "")


def test_job_engine_run_reads_cache_n_from_the_chat_timings(fake_lemonade, monkeypatch):
    server = get_fake_server(fake_lemonade)
    assert server is not None
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    original = client.create_job

    def with_cache_hits(steps, *args, **kwargs):
        job = original(steps, *args, **kwargs)
        for step_id, output in (server.fake_jobs[job["id"]].get("context") or {}).items():
            if step_id.startswith("trial_"):
                output.setdefault("timings", {})["cache_n"] = 12
        return job

    monkeypatch.setattr(client, "create_job", with_cache_hits)

    result, trials = run_sweep_via_job(client, environment, _cfg())

    assert [t["cache_tokens"] for t in trials] == [12, 12]
    assert result["validity"]["valid"] is False


def test_validity_records_prompt_cache_free_only_when_the_backend_reports_it(fake_lemonade, monkeypatch):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    unreported, _ = run_sweep(client, environment, _cfg())
    monkeypatch.setitem(conftest.FAKE_STATS, "cache_tokens", 0)
    clean, _ = run_sweep(client, environment, _cfg())
    monkeypatch.setitem(conftest.FAKE_STATS, "cache_tokens", 5)
    hit, _ = run_sweep(client, environment, _cfg())

    for result in (unreported, clean, hit):
        validate_result(result)
    assert "prompt_cache_free" not in unreported["validity"]
    assert clean["validity"]["prompt_cache_free"] is True
    assert hit["validity"]["prompt_cache_free"] is False
