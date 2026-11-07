import json

import pytest
from click.testing import CliRunner
from conftest import FAKE_ROUTER_MODEL_ID

import lemonmatrix.bench as bench_mod
from lemonmatrix.bench import SweepConfig, run_sweep
from lemonmatrix.cli import cli
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.prompt_sets import PROMPT_SETS, load_prompt_file
from lemonmatrix.routing_cost import parse_cost_override, resolve_price, summarize_routing
from lemonmatrix.validate import validate_result

LOCAL = "Qwen3-0.6B-GGUF"
CLOUD = "fireworks.kimi-k2p5"
RECIPES = {LOCAL: "llamacpp", CLOUD: "cloud"}
CLOUD_PRICE = {"cost_input_per_million": 0.6, "cost_output_per_million": 2.5}


def _trial(route_to, prompt_tokens=100, output_tokens=200, estimated_cost=None):
    trial = {"route_to": route_to, "prompt_tokens": prompt_tokens, "output_tokens": output_tokens}
    if estimated_cost is not None:
        trial["outputs"] = {"estimated_cost": estimated_cost}
    return trial


def test_parse_cost_override_reads_model_and_prices():
    assert parse_cost_override("fireworks.kimi-k2p5=0.6,2.5") == (CLOUD, {"input": 0.6, "output": 2.5})


def test_parse_cost_override_keeps_equals_signs_in_the_model_name():
    model, _ = parse_cost_override("odd=name=1,2")
    assert model == "odd=name"


@pytest.mark.parametrize("text", ["no-equals", "m=1", "m=1,2,3", "m=a,b", "m=-1,2", "=1,2"])
def test_parse_cost_override_rejects_malformed_input(text):
    with pytest.raises(ValueError):
        parse_cost_override(text)


def test_resolve_price_prefers_override_over_server_cost():
    price, source = resolve_price(CLOUD, {"estimated_cost": CLOUD_PRICE}, RECIPES, {CLOUD: {"input": 1, "output": 1}})
    assert (price, source) == ({"input": 1, "output": 1}, "override")


def test_resolve_price_uses_server_estimated_cost():
    price, source = resolve_price(CLOUD, {"estimated_cost": CLOUD_PRICE}, RECIPES, {})
    assert (price, source) == ({"input": 0.6, "output": 2.5}, "server")


def test_resolve_price_treats_free_tier_as_zero():
    price, source = resolve_price(CLOUD, {"estimated_cost": {"cost_tier": "free"}}, RECIPES, {})
    assert (price, source) == ({"input": 0.0, "output": 0.0}, "server")


def test_resolve_price_defaults_local_candidates_to_free():
    assert resolve_price(LOCAL, None, RECIPES, {}) == ({"input": 0.0, "output": 0.0}, "local_default")


def test_resolve_price_leaves_unpriced_cloud_candidates_unpriced():
    assert resolve_price(CLOUD, {"estimated_cost": {"cost_tier": "high"}}, RECIPES, {}) == (None, "unpriced")


def test_resolve_price_does_not_assume_unknown_recipes_are_local():
    assert resolve_price("mystery", None, RECIPES, {}) == (None, "unpriced")


def test_summarize_routing_prices_a_local_cloud_split():
    trials = [_trial(LOCAL)] * 3 + [_trial(CLOUD, estimated_cost=CLOUD_PRICE)]
    summary = summarize_routing(trials, RECIPES)

    cloud_cost = (100 * 0.6 + 200 * 2.5) / 1_000_000
    assert summary["requests"] == 4
    assert summary["local_share"] == 0.75
    assert summary["free_share"] == 0.75
    assert summary["routed_cost_usd"] == pytest.approx(cloud_cost)
    assert summary["baseline_model"] == CLOUD
    assert summary["baseline_cost_usd"] == pytest.approx(4 * cloud_cost)
    assert summary["savings_pct"] == pytest.approx(75.0)
    assert summary["candidates"][LOCAL] == {
        "requests": 3, "share": 0.75, "prompt_tokens": 300, "output_tokens": 600,
        "cost_usd": 0.0, "price_source": "local_default",
    }
    assert summary["candidates"][CLOUD]["price_source"] == "server"


def test_summarize_routing_prices_the_baseline_even_when_it_never_served_a_request():
    summary = summarize_routing([_trial(LOCAL)] * 2, RECIPES, overrides={CLOUD: {"input": 1.0, "output": 1.0}})

    assert summary["routed_cost_usd"] == 0
    assert summary["baseline_model"] == CLOUD
    assert summary["baseline_cost_usd"] == pytest.approx(2 * 300 / 1_000_000)
    assert summary["savings_pct"] == pytest.approx(100.0)


def test_summarize_routing_skips_savings_when_a_request_is_unpriced():
    trials = [_trial(LOCAL), _trial(CLOUD)]
    summary = summarize_routing(trials, RECIPES, overrides={"other.cloud": {"input": 1, "output": 1}})

    assert summary["unpriced_requests"] == 1
    assert summary["candidates"][CLOUD]["cost_usd"] is None
    assert "baseline_cost_usd" not in summary
    assert "savings_pct" not in summary


def test_summarize_routing_keeps_an_explicit_baseline_without_a_price():
    summary = summarize_routing([_trial(LOCAL)], RECIPES, baseline_model=CLOUD)

    assert summary["baseline_model"] == CLOUD
    assert "baseline_cost_usd" not in summary


def test_summarize_routing_returns_none_without_route_decisions():
    assert summarize_routing([{"prompt_tokens": 1, "output_tokens": 1}], RECIPES) is None


def test_load_prompt_file_reads_lines_and_skips_blanks(tmp_path):
    path = tmp_path / "prompts.txt"
    path.write_text("first\n\n  second  \n")
    assert load_prompt_file(path) == ["first", "second"]


def test_load_prompt_file_reads_jsonl_strings_and_objects(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(json.dumps("plain") + "\n" + json.dumps({"prompt": "multi\nline"}) + "\n")
    assert load_prompt_file(path) == ["plain", "multi\nline"]


def test_load_prompt_file_rejects_an_empty_file(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("\n\n")
    with pytest.raises(ValueError):
        load_prompt_file(path)


def test_router_mix_has_both_short_and_long_prompts():
    lengths = sorted(len(p) for p in PROMPT_SETS["router-mix"])
    assert lengths[0] < 60 < lengths[-1]


def _router_cfg(**overrides) -> SweepConfig:
    values = dict(
        model_name=FAKE_ROUTER_MODEL_ID, model_class="router", quantization="none", context_length=4096,
        compute_engine="router", backend="collection.router", os="linux", power_state="plugged",
        warmup_trials=1, measured_trials=4, run_type="router",
    )
    values.update(overrides)
    return SweepConfig(**values)


def _route_by_prompt(monkeypatch):
    original = bench_mod._run_one
    seen_prompts = []

    def fake_run_one(client, cfg):
        trial = original(client, cfg)
        seen_prompts.append(cfg.prompt)
        if "code" in cfg.prompt:
            trial["route_to"] = CLOUD
            trial["outputs"] = {"estimated_cost": CLOUD_PRICE}
        else:
            trial["outputs"] = {"estimated_cost": {"cost_tier": "free"}}
        return trial

    monkeypatch.setattr(bench_mod, "_run_one", fake_run_one)
    return seen_prompts


def test_router_run_cycles_through_the_prompt_set(fake_lemonade, monkeypatch):
    seen_prompts = _route_by_prompt(monkeypatch)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    _, trials = run_sweep(client, environment, _router_cfg(prompts=["chat", "code"]))

    assert seen_prompts == ["chat", "chat", "code", "chat", "code"]
    assert [t["prompt_index"] for t in trials] == [0, 1, 0, 1]


def test_router_run_reports_a_schema_valid_routing_block(fake_lemonade, monkeypatch):
    _route_by_prompt(monkeypatch)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _router_cfg(prompts=["chat", "code"]))

    validate_result(result)
    routing = result["metrics"]["routing"]
    assert routing["requests"] == 4
    assert routing["prompt_set_size"] == 2
    assert routing["candidates"][CLOUD]["requests"] == 2
    assert routing["baseline_model"] == CLOUD
    assert routing["savings_pct"] == pytest.approx(50.0)


def test_router_run_hashes_the_whole_prompt_set(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    a, _ = run_sweep(client, environment, _router_cfg(prompts=["one", "two"]))
    b, _ = run_sweep(client, environment, _router_cfg(prompts=["one", "three"]))

    assert a["metrics"]["prompt_sha256"] != b["metrics"]["prompt_sha256"]


def test_model_run_has_no_routing_block(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    cfg = SweepConfig(
        model_name="Llama-3.1-8B-Instruct-GGUF", model_class="dense", quantization="Q4_K_M",
        context_length=4096, compute_engine="cpu", backend="llamacpp-cpu", os="linux",
        power_state="plugged", warmup_trials=0, measured_trials=1,
    )

    result, _ = run_sweep(client, environment, cfg)

    assert "routing" not in result["metrics"]


@pytest.mark.parametrize("flag", [["--prompt-set", "router-mix"], ["--cost", "m=1,2"], ["--cost-baseline", "m"]])
def test_cli_rejects_router_cost_flags_on_model_runs(flag):
    args = ["run", "--profile", "nope", "--model", "m", "--context-length", "4096", "--power-state", "plugged"]
    result = CliRunner().invoke(cli, args + flag)

    assert result.exit_code != 0
    assert "need --run-type router" in result.output


def test_cli_rejects_a_malformed_cost_override():
    args = ["run", "--profile", "nope", "--run-type", "router", "--model", "r", "--context-length", "4096",
            "--power-state", "plugged", "--cost", "broken"]
    result = CliRunner().invoke(cli, args)

    assert result.exit_code != 0
    assert "--cost:" in result.output
