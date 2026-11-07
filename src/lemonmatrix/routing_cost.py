"""What a router's choices cost, compared with sending everything to one model.

A router run sends a mixed prompt set through a collection.router and records
which candidate served each request. This module prices every request and sums
it up: how many requests each candidate took, how many stayed local, what the
run cost, and what it would have cost if every request had gone to a baseline
model (by default the most expensive candidate seen).

Prices are USD per million tokens, resolved per request in this order:

1. a caller override (``--cost MODEL=IN,OUT``);
2. the ``estimated_cost`` block Lemonade attaches to the route decision when the
   candidate carries cost metadata (cloud models get it from provider
   discovery; local models can register it);
3. free, when the candidate's recipe is not ``cloud``: a local model has no
   per-token bill.

A cloud candidate with no price is counted as unpriced rather than guessed.
The baseline is priced on the same token counts the routed run produced, so it
is an estimate: a different model would not write the same number of tokens.
"""

from __future__ import annotations

CLOUD_RECIPE = "cloud"


def parse_cost_override(text: str) -> tuple[str, dict]:
    """Parse ``MODEL=IN,OUT`` (USD per million input/output tokens)."""
    model, sep, prices = text.rpartition("=")
    if not sep or not model:
        raise ValueError(f"expected MODEL=IN,OUT, got {text!r}")
    parts = prices.split(",")
    if len(parts) != 2:
        raise ValueError(f"expected two prices IN,OUT after '=', got {prices!r}")
    try:
        input_price, output_price = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"prices must be numbers, got {prices!r}") from exc
    if input_price < 0 or output_price < 0:
        raise ValueError(f"prices must not be negative, got {prices!r}")
    return model, {"input": input_price, "output": output_price}


def _server_price(outputs: dict | None) -> dict | None:
    estimated = (outputs or {}).get("estimated_cost") or {}
    input_price = estimated.get("cost_input_per_million")
    output_price = estimated.get("cost_output_per_million")
    if input_price is not None or output_price is not None:
        return {"input": float(input_price or 0.0), "output": float(output_price or 0.0)}
    if estimated.get("cost_tier") == "free":
        return {"input": 0.0, "output": 0.0}
    return None


def resolve_price(
    model: str, outputs: dict | None, recipes: dict[str, str], overrides: dict[str, dict]
) -> tuple[dict | None, str]:
    """Return (price, source) for one routed request; price is None if unknown."""
    if model in overrides:
        return overrides[model], "override"
    server = _server_price(outputs)
    if server is not None:
        return server, "server"
    recipe = recipes.get(model)
    if recipe is not None and recipe != CLOUD_RECIPE:
        return {"input": 0.0, "output": 0.0}, "local_default"
    return None, "unpriced"


def _request_cost(price: dict, prompt_tokens: int, output_tokens: int) -> float:
    return (prompt_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def summarize_routing(
    trials: list[dict],
    recipes: dict[str, str],
    overrides: dict[str, dict] | None = None,
    baseline_model: str | None = None,
    prompt_set_size: int | None = None,
) -> dict | None:
    """Summarize routed trials into the result's ``metrics.routing`` block.

    Returns None when no trial carries a routing decision.
    """
    overrides = overrides or {}
    routed = [t for t in trials if t.get("route_to")]
    if not routed:
        return None

    candidates: dict[str, dict] = {}
    known_prices: dict[str, dict] = dict(overrides)
    total_cost = 0.0
    unpriced = 0
    local = 0
    free = 0

    for trial in routed:
        model = trial["route_to"]
        prompt_tokens = int(trial.get("prompt_tokens") or 0)
        output_tokens = int(trial.get("output_tokens") or 0)
        price, source = resolve_price(model, trial.get("outputs"), recipes, overrides)

        entry = candidates.setdefault(
            model,
            {"requests": 0, "prompt_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "price_source": source},
        )
        entry["requests"] += 1
        entry["prompt_tokens"] += prompt_tokens
        entry["output_tokens"] += output_tokens

        if recipes.get(model) != CLOUD_RECIPE:
            local += 1
        if price is None:
            unpriced += 1
            entry["cost_usd"] = None
            entry["price_source"] = "unpriced"
            continue
        known_prices.setdefault(model, price)
        if price["input"] == 0 and price["output"] == 0:
            free += 1
        cost = _request_cost(price, prompt_tokens, output_tokens)
        total_cost += cost
        if entry["cost_usd"] is not None:
            entry["cost_usd"] += cost

    count = len(routed)
    for entry in candidates.values():
        entry["share"] = entry["requests"] / count

    summary: dict = {
        "requests": count,
        "candidates": candidates,
        "local_share": local / count,
        "free_share": free / count,
        "routed_cost_usd": total_cost,
        "unpriced_requests": unpriced,
    }
    if prompt_set_size:
        summary["prompt_set_size"] = prompt_set_size

    if baseline_model is None and known_prices:
        baseline_model = max(known_prices, key=lambda m: (known_prices[m]["output"], known_prices[m]["input"]))
    baseline_price = known_prices.get(baseline_model) if baseline_model else None
    if baseline_model:
        summary["baseline_model"] = baseline_model
    if baseline_price is not None and unpriced == 0:
        baseline_cost = sum(
            _request_cost(baseline_price, int(t.get("prompt_tokens") or 0), int(t.get("output_tokens") or 0))
            for t in routed
        )
        summary["baseline_cost_usd"] = baseline_cost
        summary["savings_usd"] = baseline_cost - total_cost
        if baseline_cost > 0:
            summary["savings_pct"] = 100 * (baseline_cost - total_cost) / baseline_cost
    return summary
