"""Runs the actual sweep: warmup, measured trials, aggregation, validity gates.

Prefill throughput isn't directly exposed by Lemonade's /v1/stats -- it's
approximated as prompt_tokens / time_to_first_token, the same convention used
by most llama.cpp-style benchmark harnesses (TTFT is dominated by prompt
processing for any prompt long enough to matter). Decode throughput and TTFT
come straight from /v1/stats after each completion.

/v1/stats' time_to_first_token is assumed to be in seconds (undocumented in
the public API reference at the time this was written -- if a live instance
reports it in milliseconds instead, fix STATS_TTFT_IS_SECONDS below).

run_sweep returns (result_dict, raw_trials_list).  Callers that want raw
per-trial measurements (for sidecar files, reproducibility archives, etc.)
unpack the tuple; callers that only need the schema-conformant result can
ignore the second element.
"""

from __future__ import annotations

import hashlib
import statistics
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .backend_version import lookup_backend_version, resolve_backend
from .capabilities import engine_backend_compatible, parse_quantization
from .client import LemonadeClient

STATS_TTFT_IS_SECONDS = True

DEFAULT_PROMPT = (
    "Summarize the plot of a short story about a lighthouse keeper who "
    "discovers a message in a bottle, in about three sentences."
)


@dataclass
class SweepConfig:
    model_name: str
    model_class: str
    quantization: str
    context_length: int
    compute_engine: str
    backend: str
    os: str
    power_state: str
    power_cap_w: float | None = None
    warmup_trials: int = 2
    measured_trials: int = 5
    max_tokens: int = 256
    prompt: str = DEFAULT_PROMPT
    exclusive_run: bool = True
    parameters_b: float | None = None
    active_parameters_b: float | None = None
    energy_price_usd_per_kwh: float | None = None
    # Amortized hardware cost: the schema's cost_per_1k_tokens_usd has always
    # described "amortized hardware plus energy," but only the energy half
    # was ever implemented -- and since Lemonade reports no power draw on any
    # real instance checked, that half is always None in practice, so the
    # cost field has never actually populated on a real run. Hardware
    # amortization needs no power data at all, so these two inputs make the
    # field usable for the first time.
    hardware_cost_usd: float | None = None
    hardware_lifetime_hours: float | None = None
    # Router runs: set run_type="router" to benchmark a collection.router model.
    # Lemonade handles routing transparently -- the same chat/completions endpoint
    # is used; route_trace=true asks for x_lemonade_route in the response body.
    run_type: str = "model"  # "model" | "router"


def _ttft_ms(stats: dict) -> float:
    raw = stats.get("time_to_first_token", 0.0) or 0.0
    return raw * 1000 if STATS_TTFT_IS_SECONDS else raw


def _competing_model_names(health: dict, own_model_names: set[str]) -> list[str]:
    """Names of any OTHER loaded model that is currently busy or streaming,
    per Lemonade's own /api/v1/health.

    Confirmed live: each entry in all_models_loaded[] carries is_busy/
    is_streaming booleans, and is_busy genuinely flips true for the duration
    of an in-flight request and false once it completes (verified by firing
    a real chat completion and polling health mid-request). This only
    catches a model Lemonade itself is tracking (llm/embedding/tts/image/
    transcription/classification/reranking slots) -- a process entirely
    outside Lemonade competing for the same GPU would not show up here.

    own_model_names is a set, not a single name, because a router run's
    "own model" is never the router's own collection name -- confirmed live
    that a router is virtual and never itself appears in all_models_loaded[];
    only whichever candidate it actually dispatched to does. Without the
    full candidate set here, every router run would misidentify its own
    selected candidate as an unrelated competing workload.
    """
    return sorted(
        {
            m["model_name"]
            for m in health.get("all_models_loaded") or []
            if m.get("model_name") not in own_model_names and (m.get("is_busy") or m.get("is_streaming"))
        }
    )


def _own_model_entry(health: dict, own_model_names: set[str]) -> dict | None:
    """The all_models_loaded[] entry for the model this run itself loaded, if
    Lemonade's health snapshot includes it this poll."""
    return next(
        (m for m in health.get("all_models_loaded") or [] if m.get("model_name") in own_model_names),
        None,
    )


def _device_matches_engine(device_str: str | None, compute_engine: str) -> bool | None:
    """True/False if Lemonade's own /api/v1/health-reported `device` for the
    loaded model (confirmed live: a "|"-joined bitmask string like "cpu",
    "gpu", or "cpu|gpu" -- Lemonade's DeviceType is a bitmask, not an enum)
    confirms or contradicts the caller's compute_engine claim. None if there
    is nothing to check against (device never observed this run).

    The device field distinguishes cpu/gpu/npu only -- it cannot tell an
    integrated GPU apart from a discrete one, so igpu and dgpu both map to
    the same expected token here; a genuine igpu-vs-dgpu mislabel is not
    something this check can catch.
    """
    if not device_str:
        return None
    expected = {"cpu": {"cpu"}, "igpu": {"gpu"}, "dgpu": {"gpu"}, "npu": {"npu"}}.get(compute_engine)
    if expected is None:
        return None
    return set(device_str.split("|")) == expected


class _ExclusivityMonitor:
    """Background-polls the profiled instance's own /api/v1/health while a
    run's trials execute, watching for any OTHER loaded model reporting
    is_busy/is_streaming -- a genuinely concurrent competing workload, not
    just a before/after snapshot. Also tracks the RUN'S OWN model's reported
    `device` (to cross-check the caller's compute_engine claim) and
    `watchdog_reset` (true once Lemonade's own backend watchdog force-
    restarts the model's subprocess -- a mid-run crash/recovery, not steady-
    state inference). Polls through the same client as everything else, so
    it always targets the profiled instance (never local hardware) and is
    Lemonade's own API, not a bypass.

    This is a spot check at whatever cadence `interval_s` allows, not
    continuous monitoring -- a competing request that starts and finishes
    entirely between two polls would not be caught. health_unreachable is
    set both when every poll failed AND when the run finished before the
    first poll had a chance to fire at all (e.g. an unrealistically fast
    run, or in tests against an instant-responding fake server) -- either
    way, verification did not actually happen, so callers must not treat
    an empty competing_models set as a confirmed-clean result.
    """

    def __init__(self, client: LemonadeClient, own_model_name: str | set[str], interval_s: float = 0.3):
        self.client = client
        # A router run passes the full candidate set (see run_sweep) since
        # the router's own collection name never appears in health -- every
        # other call site still passes a single model name, normalized here
        # to a one-element set so _competing_model_names/_own_model_entry
        # have one consistent shape to check membership against.
        self.own_model_names = {own_model_name} if isinstance(own_model_name, str) else set(own_model_name)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.competing_models: set[str] = set()
        self.poll_count = 0
        self.poll_failures = 0
        self.own_device_seen: str | None = None
        self.watchdog_reset_seen = False

    def _run(self) -> None:
        while not self._stop.is_set():
            self.poll_count += 1
            try:
                health = self.client.health()
                self.competing_models.update(_competing_model_names(health, self.own_model_names))
                own = _own_model_entry(health, self.own_model_names)
                if own:
                    if own.get("device"):
                        self.own_device_seen = own["device"]
                    if own.get("watchdog_reset"):
                        self.watchdog_reset_seen = True
            except Exception:
                self.poll_failures += 1
            self._stop.wait(self.interval_s)

    @property
    def health_unreachable(self) -> bool:
        # Covers both "every poll failed" and "zero polls happened" (0 == 0
        # is True) -- a run that finishes before the first poll fires must
        # not be mistaken for a clean, verified result.
        return self.poll_count == self.poll_failures

    def __enter__(self) -> "_ExclusivityMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_s * 2)


def _resource_samples(client: LemonadeClient) -> dict:
    """Best-effort vram_gb/host_memory_gb/watts from the *profiled instance's
    own* /api/v1/system-stats -- never local hardware tools.

    A profile can point at any machine (IDEA.md: local, LAN, or cloud), not
    necessarily the one this process runs on, so the only legitimate source
    for facts about it is Lemonade's own API against that profile's client.
    Confirmed live that vram_gb can be null even mid-inference (seen with
    llamacpp's Vulkan backend on an NVIDIA host) -- host_memory_gb is the
    fallback used for peak_memory_gb when that happens, since the schema
    requires a number. Lemonade does not currently report power draw through
    this or any other endpoint we've found; the candidate keys below are
    forward-compatible guesses in case a future version adds one -- if none
    match, watts stays None and energy metrics are correctly omitted rather
    than estimated some other way.
    """
    try:
        stats = client.system_stats()
    except Exception:
        return {"vram_gb": None, "host_memory_gb": None, "watts": None}
    watts = stats.get("power_w") or stats.get("power_watts") or stats.get("watts")
    return {"vram_gb": stats.get("vram_gb"), "host_memory_gb": stats.get("memory_gb"), "watts": watts}


def _run_one(client: LemonadeClient, cfg: SweepConfig) -> dict:
    """Run one trial.

    For router runs we pass route_trace=True so Lemonade adds x_lemonade_route
    to the response body.  That object carries { route_to, matched_rule,
    default_used, outputs, trace[] } and is captured per-trial in the sidecar
    without touching the schema-conformant result.
    """
    kwargs: dict = {}
    if cfg.run_type == "router":
        kwargs["route_trace"] = True

    response = client.chat_completion(
        model=cfg.model_name,
        messages=[{"role": "user", "content": cfg.prompt}],
        max_tokens=cfg.max_tokens,
        **kwargs,
    )
    stats = client.stats()
    resources = _resource_samples(client)

    trial: dict = {
        "ttft_ms": _ttft_ms(stats),
        "decode_tokens_per_sec": stats.get("tokens_per_second"),
        "prompt_tokens": stats.get("prompt_tokens") or stats.get("input_tokens") or 0,
        "output_tokens": stats.get("output_tokens", 0),
        "watts": resources["watts"],
        "vram_gb": resources["vram_gb"],
        "host_memory_gb": resources["host_memory_gb"],
    }

    # Capture routing decision when route_trace was requested.
    route_info = response.get("x_lemonade_route") if isinstance(response, dict) else None
    if route_info:
        trial["route_to"] = route_info.get("route_to")
        trial["matched_rule"] = route_info.get("matched_rule")
        trial["default_used"] = route_info.get("default_used")

    return trial


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _stddev(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.stdev(vals) if len(vals) >= 2 else None


def _p95(values: list) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    idx = (len(vals) - 1) * 0.95
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


# Two-tailed 95% critical values from the Student's t-distribution, keyed by
# degrees of freedom (trial_count - 1). A handful of measured trials is a
# small sample -- using the normal distribution's fixed 1.96 there
# understates the true interval width. Falls back to 1.96 (the
# infinite-df limit) past this table's range, where the two are close enough
# that it's not worth extending this by hand.
_T95_TABLE: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074,
    23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _ci95_half_width(values: list) -> float | None:
    """Half-width of a 95% confidence interval on the mean, using the
    Student's t critical value for the actual trial count rather than a
    fixed 1.96 -- see _T95_TABLE. None below 2 trials (no stddev to build
    an interval from)."""
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    sd = statistics.stdev(vals)
    df = len(vals) - 1
    t = _T95_TABLE.get(df, 1.96)
    return t * sd / (len(vals) ** 0.5)


def run_sweep(
    client: LemonadeClient, environment: dict, cfg: SweepConfig
) -> tuple[dict, list[dict]]:
    """Load the model, warm up, run measured trials, and return
    (schema-conformant result dict, list of raw per-trial measurements).

    The raw trials list is a sidecar that callers can optionally persist;
    it is NOT embedded in the result dict so that it never touches schema
    validation.
    """
    # Router runs: the backend is a routing policy, not a recipe-based engine
    # selector.  Lemonade's routing engine handles backend selection internally,
    # so none of the model-run validity checks apply here.
    is_router = cfg.run_type == "router"

    engine_ok = True
    engine_reason = ""
    if not is_router:
        engine_ok, engine_reason = engine_backend_compatible(cfg.compute_engine, cfg.backend)

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    # Without this, /api/v1/load has no backend selector at all and Lemonade
    # silently falls back to the recipe's configured default_backend (e.g.
    # llamacpp defaults to cuda) -- which may not be the backend the user
    # picked, or even installed, regardless of what cfg.backend says.
    # For router runs, /api/v1/load is still called (Lemonade loads the routing
    # policy), but there is no recipe-based backend selector to inject -- the
    # policy's routing rules determine which downstream model and backend are
    # used, not a load-time flag from us.
    load_kwargs = {"ctx_size": cfg.context_length}
    backend_unresolved_note = None
    if not is_router:
        resolved = resolve_backend(system_info, cfg.backend)
        if resolved:
            recipe_key, backend_key = resolved
            load_kwargs[f"{recipe_key}_backend"] = backend_key
        else:
            backend_unresolved_note = (
                f"could not resolve backend '{cfg.backend}' to a recipe -- Lemonade used its "
                "recipe's default backend for this model instead"
            )

    # A router is virtual and never itself appears in /api/v1/health --
    # confirmed live that only whichever candidate it actually dispatched to
    # shows up there. Without this, the exclusivity monitor would misidentify
    # a router run's own selected candidate as an unrelated competing
    # workload on every single trial. `models` carries the router's own
    # declared candidate list under "components" (confirmed live) -- this
    # MUST be resolved before client.load() below: confirmed live that
    # POST /api/v1/load on a router's own name makes Lemonade drop it from
    # subsequent GET /api/v1/models responses (the router itself has nothing
    # to "load", per its own virtual-collection design, but issuing the call
    # still has this listing side effect) -- looking it up afterward would
    # silently find nothing and fall back to the buggy pre-fix behavior.
    own_model_names: str | set[str] = cfg.model_name
    if is_router:
        try:
            router_entry = next((m for m in client.models() if m.get("id") == cfg.model_name), None)
            components = (router_entry or {}).get("components") or []
            own_model_names = {cfg.model_name, *components}
        except Exception:
            own_model_names = cfg.model_name

    client.load(cfg.model_name, **load_kwargs)

    exclusivity = _ExclusivityMonitor(client, own_model_names)
    with exclusivity:
        for _ in range(cfg.warmup_trials):
            _run_one(client, cfg)

        # The model stays resident for every measured trial, so model_reload_free
        # holds for the whole run -- there is no unload here by design.
        raw_trials = [_run_one(client, cfg) for _ in range(cfg.measured_trials)]

    return _aggregate_sweep_result(
        client, environment, cfg, raw_trials, exclusivity, system_info,
        engine_ok, engine_reason, backend_unresolved_note, is_router,
    )


def _aggregate_sweep_result(
    client: LemonadeClient,
    environment: dict,
    cfg: SweepConfig,
    raw_trials: list[dict],
    exclusivity: "_ExclusivityMonitor",
    system_info: dict,
    engine_ok: bool,
    engine_reason: str,
    backend_unresolved_note: str | None,
    is_router: bool,
) -> tuple[dict, list[dict]]:
    """Turns raw_trials (in _run_one()'s dict shape: ttft_ms,
    decode_tokens_per_sec, prompt_tokens, output_tokens, watts, vram_gb,
    host_memory_gb, plus route_to/matched_rule/default_used for router runs)
    into a schema-conformant result. Shared by run_sweep() (collects
    raw_trials via N direct per-trial HTTP calls from this process) and
    run_sweep_via_job() (collects the identical shape from a single Lemonade
    job's per-step context) -- the statistics/validity/cost-model logic that
    turns raw measurements into a result is identical either way, only how
    the measurements were taken differs.
    """
    ttft_vals = [t["ttft_ms"] for t in raw_trials]
    decode_vals = [t["decode_tokens_per_sec"] for t in raw_trials if t["decode_tokens_per_sec"] is not None]
    prompt_vals = [t["prompt_tokens"] for t in raw_trials]

    ttft_ms = _mean(ttft_vals)
    decode_tps = _mean(decode_vals)
    mean_watts = _mean([t["watts"] for t in raw_trials])

    prefill_tps = None
    prompt_tokens = _mean(prompt_vals)
    if prompt_tokens and ttft_ms:
        prefill_tps = prompt_tokens / (ttft_ms / 1000)

    decode: dict = {"tokens_per_sec": decode_tps}
    if decode_tps:
        decode["inter_token_latency_ms"] = 1000 / decode_tps
    stddev_d = _stddev(decode_vals)
    p95_d = _p95(decode_vals)
    ci95_d = _ci95_half_width(decode_vals)
    if stddev_d is not None:
        decode["stddev"] = stddev_d
    if p95_d is not None:
        decode["p95"] = p95_d
    if ci95_d is not None:
        decode["ci95_half_width"] = ci95_d

    prefill: dict = {"tokens_per_sec": prefill_tps}
    # Prefill stats derived from per-trial prompt_tokens / ttft_ms pairs
    prefill_per_trial = [
        (t["prompt_tokens"] / (t["ttft_ms"] / 1000))
        for t in raw_trials
        if t["prompt_tokens"] and t["ttft_ms"]
    ]
    stddev_p = _stddev(prefill_per_trial)
    p95_p = _p95(prefill_per_trial)
    ci95_p = _ci95_half_width(prefill_per_trial)
    if stddev_p is not None:
        prefill["stddev"] = stddev_p
    if p95_p is not None:
        prefill["p95"] = p95_p
    if ci95_p is not None:
        prefill["ci95_half_width"] = ci95_p

    if mean_watts is not None:
        if prefill_tps:
            prefill["joules_per_token"] = mean_watts / prefill_tps
        if decode_tps:
            decode["joules_per_token"] = mean_watts / decode_tps

    vram_samples = [t["vram_gb"] for t in raw_trials if t["vram_gb"] is not None]
    host_memory_samples = [t["host_memory_gb"] for t in raw_trials if t["host_memory_gb"] is not None]
    peak_memory_gb = None
    memory_is_host_fallback = False
    if vram_samples:
        peak_memory_gb = max(vram_samples)
    elif host_memory_samples:
        # vram_gb came back null for every trial (seen live: llamacpp's
        # Vulkan backend doesn't report it on this host) -- host RAM is not
        # the same thing as VRAM, but the schema requires a number here, and
        # reporting nothing would be worse than reporting the wrong memory
        # pool with a clear note.
        peak_memory_gb = max(host_memory_samples)
        memory_is_host_fallback = True

    metrics: dict = {
        "prefill": prefill,
        "decode": decode,
        "ttft_ms": ttft_ms,
        "trial_count": cfg.measured_trials,
        "peak_memory_gb": peak_memory_gb,
        # Reproducibility: lets two results be confirmed to have run the
        # identical prompt without embedding the (potentially long) prompt
        # text itself in every result file.
        "prompt_sha256": hashlib.sha256(cfg.prompt.encode()).hexdigest(),
    }
    ttft_sd = _stddev(ttft_vals)
    ttft_ci95 = _ci95_half_width(ttft_vals)
    if ttft_sd is not None:
        metrics["ttft_ms_stddev"] = ttft_sd
    if ttft_ci95 is not None:
        metrics["ttft_ms_ci95_half_width"] = ttft_ci95

    # Energy cost: needs mean_watts, which is None on every real instance
    # confirmed so far (Lemonade reports no power draw at all -- see
    # _resource_samples). Hardware amortization needs no power data, so it
    # can populate cost_per_1k_tokens_usd even when the energy half can't.
    energy_cost_per_1k = None
    if cfg.energy_price_usd_per_kwh and mean_watts and decode_tps:
        kwh_per_token = (mean_watts / decode_tps) / 3600 / 1000
        energy_cost_per_1k = kwh_per_token * cfg.energy_price_usd_per_kwh * 1000

    hardware_cost_per_1k = None
    if cfg.hardware_cost_usd and cfg.hardware_lifetime_hours and decode_tps:
        hardware_cost_per_hour = cfg.hardware_cost_usd / cfg.hardware_lifetime_hours
        hardware_cost_per_1k = (hardware_cost_per_hour / 3600 / decode_tps) * 1000

    if energy_cost_per_1k is not None or hardware_cost_per_1k is not None:
        metrics["cost_per_1k_tokens_usd"] = (energy_cost_per_1k or 0) + (hardware_cost_per_1k or 0)
        if energy_cost_per_1k is not None:
            metrics["energy_cost_per_1k_tokens_usd"] = energy_cost_per_1k
            metrics["energy_price_usd_per_kwh"] = cfg.energy_price_usd_per_kwh
        if hardware_cost_per_1k is not None:
            metrics["hardware_cost_per_1k_tokens_usd"] = hardware_cost_per_1k
            metrics["hardware_cost_usd"] = cfg.hardware_cost_usd
            metrics["hardware_lifetime_hours"] = cfg.hardware_lifetime_hours

    # Server-verified exclusivity, device, and reload-freedom: trust
    # Lemonade's own /api/v1/health data over the caller's assertions where
    # we could actually poll it -- see _exclusivity_verdict/_device_verdict/
    # _watchdog_verdict's docstrings for the full rationale on each.
    exclusive_run, exclusivity_note = _exclusivity_verdict(exclusivity, cfg.exclusive_run)
    device_ok, device_note = _device_verdict(exclusivity, cfg.compute_engine, is_router)
    model_reload_free, reload_note = _watchdog_verdict(exclusivity)
    # Quantization has no meaningful claim for router runs (schema forces it
    # to "none"), so skip the check the same way engine/backend and device
    # checks already skip router runs.
    quantization_ok, quantization_note = (
        (True, None) if is_router else _quantization_verdict(client, cfg.model_name, cfg.quantization)
    )

    thermal_ok = True
    notes = []
    if not engine_ok:
        notes.append(f"engine/backend mismatch: {engine_reason}")
    if mean_watts is None:
        notes.append("Lemonade doesn't report power draw for this instance; energy metrics omitted")
    if backend_unresolved_note:
        notes.append(backend_unresolved_note)
    if memory_is_host_fallback:
        notes.append("peak_memory_gb is host RAM, not VRAM -- vram_gb was unavailable from this instance")
    if device_note:
        notes.append(device_note)
    if reload_note:
        notes.append(reload_note)
    if quantization_note:
        notes.append(quantization_note)
    if exclusivity_note:
        notes.append(exclusivity_note)

    validity = {
        # An unresolvable backend or an engine/backend mismatch both invalidate
        # the run: the former means Lemonade may have loaded a different backend
        # than config.backend claims; the latter means the declared engine does
        # not physically match the backend's execution path.
        "valid": thermal_ok and exclusive_run and not backend_unresolved_note and engine_ok and device_ok and model_reload_free and quantization_ok,
        "warmup_discarded": cfg.warmup_trials > 0,
        "thermal_ok": thermal_ok,
        "exclusive_run": exclusive_run,
        "model_reload_free": model_reload_free,
    }
    if notes:
        validity["notes"] = "; ".join(notes)

    # Best-effort: only set when the engine build behind cfg.backend is
    # actually installed on this instance, not merely installable.  Routers
    # are routing policies, not recipe-based engines, so there is no backend
    # build version to record for them.
    backend_version = None if is_router else lookup_backend_version(system_info, cfg.backend)
    run_environment = {**environment, "backend_version": backend_version} if backend_version else environment

    # For router runs, derive a router_default_model from the first trial that
    # reported default_used=True -- that is the policy fallback candidate.
    router_default_model = None
    if cfg.run_type == "router":
        for t in raw_trials:
            if t.get("default_used") and t.get("route_to"):
                router_default_model = t["route_to"]
                break

    config_block: dict = {
        "compute_engine": cfg.compute_engine,
        "backend": cfg.backend,
        "os": cfg.os,
        "power_state": cfg.power_state,
        **({"power_cap_w": cfg.power_cap_w} if cfg.power_cap_w else {}),
        **({"router_default_model": router_default_model} if router_default_model else {}),
    }

    result: dict = {
        "schema_version": "0.1.0",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": cfg.model_name,
            "class": cfg.model_class,
            "quantization": cfg.quantization,
            "context_length": cfg.context_length,
            **({"parameters_b": cfg.parameters_b} if cfg.parameters_b else {}),
            **({"active_parameters_b": cfg.active_parameters_b} if cfg.active_parameters_b else {}),
        },
        "config": config_block,
        "environment": run_environment,
        "metrics": metrics,
        "validity": validity,
    }
    if cfg.run_type != "model":
        result["run_type"] = cfg.run_type

    return result, raw_trials


class JobFailedError(RuntimeError):
    """Raised when a Lemonade job (POST /v1/jobs) finishes with status
    "failed" or "interrupted" instead of "completed", or never reaches a
    terminal status before the poll timeout."""


def _build_sweep_job_steps(cfg: SweepConfig, load_kwargs: dict) -> list[dict]:
    """The step sequence for one run_sweep_via_job() run: unload, load, N
    warmup chats (discarded), M measured chats each immediately followed by
    a system_stats snapshot (so raw_trials gets the same per-trial
    vram_gb/host_memory_gb/watts fields the direct-HTTP path gets from
    _resource_samples()), unload. Lemonade's job engine has no loop
    primitive (confirmed against its own docs/source) -- trials must be
    explicit steps, same as this project's own sweep-generation code already
    does for the direct-HTTP path.
    """
    steps = [
        {"id": "unload_before", "op": "unload"},
        {"id": "load", "op": "load", "params": {"model": cfg.model_name, "ctx_size": cfg.context_length, **load_kwargs}},
    ]
    for i in range(cfg.warmup_trials):
        steps.append({
            "id": f"warmup_{i}", "op": "chat",
            "params": {"model": cfg.model_name, "messages": [{"role": "user", "content": cfg.prompt}], "max_tokens": cfg.max_tokens},
        })
    for i in range(cfg.measured_trials):
        steps.append({
            "id": f"trial_{i}", "op": "chat",
            "params": {"model": cfg.model_name, "messages": [{"role": "user", "content": cfg.prompt}], "max_tokens": cfg.max_tokens},
        })
        steps.append({"id": f"stats_{i}", "op": "system_stats"})
    steps.append({"id": "unload_after", "op": "unload"})
    return steps


def _poll_job_until_done(client: LemonadeClient, job_id: str, poll_interval_s: float = 1.0, timeout_s: float = 1800) -> dict:
    """Blocks until the job reaches a terminal status, returning the final
    GET /v1/jobs/{id} record. "queued"/"running"/"paused" mean still in
    flight; "completed"/"failed"/"interrupted" are terminal (confirmed
    against Lemonade's own JobStatus enum and live against a real instance).
    """
    deadline = time.monotonic() + timeout_s
    while True:
        job = client.get_job(job_id)
        status = job.get("status")
        if status in ("completed", "failed", "interrupted"):
            return job
        if time.monotonic() > deadline:
            raise JobFailedError(f"job {job_id} did not finish within {timeout_s}s (last status: {status})")
        time.sleep(poll_interval_s)


def run_sweep_via_job(client: LemonadeClient, environment: dict, cfg: SweepConfig) -> tuple[dict, list[dict]]:
    """Same benchmark as run_sweep(), but delegates execution and durability
    to a single Lemonade job (POST /v1/jobs) instead of this process making N
    sequential direct HTTP calls -- the job survives this process
    disconnecting or being killed mid-run, since Lemonade itself owns and
    persists it (confirmed against its own docs/source: jobs.json, atomic
    writes). Router runs are not supported here: a router has no fixed
    backend/ctx_size to put in a job's "load" step the way a model run does.

    Live-verified end to end against a real instance: a job's chat-step
    output embeds `timings`/`usage` directly (no separate /v1/stats call
    needed, unlike the direct-HTTP path), and `context[step_id]` gives clean
    access to each step's output after completion. This is the reason
    run_sweep() and this function share _aggregate_sweep_result() rather
    than duplicating the statistics logic: once raw_trials is in the same
    shape, everything downstream is identical.
    """
    if cfg.run_type == "router":
        raise ValueError(
            "run_sweep_via_job does not support router runs -- a router has no fixed "
            "backend/ctx_size to put in a job's load step. Use run_sweep() for router runs."
        )
    is_router = False

    engine_ok, engine_reason = engine_backend_compatible(cfg.compute_engine, cfg.backend)

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    load_kwargs = {}
    backend_unresolved_note = None
    resolved = resolve_backend(system_info, cfg.backend)
    if resolved:
        recipe_key, backend_key = resolved
        load_kwargs[f"{recipe_key}_backend"] = backend_key
    else:
        backend_unresolved_note = (
            f"could not resolve backend '{cfg.backend}' to a recipe -- Lemonade used its "
            "recipe's default backend for this model instead"
        )

    steps = _build_sweep_job_steps(cfg, load_kwargs)

    exclusivity = _ExclusivityMonitor(client, cfg.model_name)
    with exclusivity:
        job = client.create_job(f"lemonmatrix-sweep-{cfg.model_name}", steps)
        job_id = job["id"]
        try:
            done = _poll_job_until_done(client, job_id)
        finally:
            try:
                client.delete_job(job_id)
            except Exception:
                pass  # best-effort cleanup -- a leftover job record isn't worth failing the run over

    if done.get("status") != "completed":
        raise JobFailedError(
            f"job {job_id} finished with status '{done.get('status')}': {done.get('error') or 'no error message reported'}"
        )

    context = done.get("context") or {}
    raw_trials = []
    for i in range(cfg.measured_trials):
        chat_out = context.get(f"trial_{i}") or {}
        stats_out = context.get(f"stats_{i}") or {}
        timings = chat_out.get("timings") or {}
        usage = chat_out.get("usage") or {}
        raw_trials.append({
            "ttft_ms": timings.get("prompt_ms") or 0.0,
            "decode_tokens_per_sec": timings.get("predicted_per_second"),
            "prompt_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens", 0),
            "watts": stats_out.get("power_w") or stats_out.get("power_watts") or stats_out.get("watts"),
            "vram_gb": stats_out.get("vram_gb"),
            "host_memory_gb": stats_out.get("memory_gb"),
        })

    return _aggregate_sweep_result(
        client, environment, cfg, raw_trials, exclusivity, system_info,
        engine_ok, engine_reason, backend_unresolved_note, is_router,
    )


@dataclass
class ClassifyConfig:
    """Benchmarks an ONNX text-classifier (Lemonade's onnxruntime recipe,
    POST /v1/classify) -- see classify_result.schema.json for why this is a
    deliberately separate pipeline from SweepConfig/run_sweep rather than a
    bolt-on run_type: classification latency/throughput isn't comparable to
    LLM token throughput.
    """

    model_name: str
    compute_engine: str
    backend: str
    os: str
    power_state: str
    input_text: str
    power_cap_w: float | None = None
    warmup_trials: int = 2
    measured_trials: int = 5
    top_k: int | None = None
    exclusive_run: bool = True


def _run_one_classify(client: LemonadeClient, cfg: ClassifyConfig) -> dict:
    """Time one classify request wall-clock, client-side.

    Confirmed against Lemonade's own server source (handle_classify in
    src/cpp/server/server.cpp): the /v1/classify response envelope carries
    no timing field at all, so there is nothing to read off the response --
    the round trip has to be timed here instead. This times the HTTP call to
    whatever machine the profile's client actually points at, not local
    hardware, so it stays within the "only Lemonade's own API" rule.
    """
    start = time.perf_counter()
    client.classify(cfg.input_text, model=cfg.model_name, top_k=cfg.top_k)
    latency_ms = (time.perf_counter() - start) * 1000
    resources = _resource_samples(client)
    return {
        "latency_ms": latency_ms,
        "vram_gb": resources["vram_gb"],
        "host_memory_gb": resources["host_memory_gb"],
    }


def run_classify(
    client: LemonadeClient, environment: dict, cfg: ClassifyConfig
) -> tuple[dict, list[dict]]:
    """Load the classifier, warm up, run measured trials, and return
    (schema-conformant classify result dict, list of raw per-trial measurements).
    """
    engine_ok, engine_reason = engine_backend_compatible(cfg.compute_engine, cfg.backend)

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    # Unlike run_sweep, no backend selector is passed to /api/v1/load: every
    # onnxruntime recipe seen so far reports "selectable_backend": false --
    # there is exactly one backend (cpu), so there is nothing to select.
    client.load(cfg.model_name)

    exclusivity = _ExclusivityMonitor(client, cfg.model_name)
    with exclusivity:
        for _ in range(cfg.warmup_trials):
            _run_one_classify(client, cfg)
        raw_trials = [_run_one_classify(client, cfg) for _ in range(cfg.measured_trials)]

    latency_vals = [t["latency_ms"] for t in raw_trials]
    latency_ms = _mean(latency_vals)
    latency_sd = _stddev(latency_vals)
    latency_p95 = _p95(latency_vals)
    classifications_per_sec = 1000 / latency_ms if latency_ms else None

    vram_samples = [t["vram_gb"] for t in raw_trials if t["vram_gb"] is not None]
    host_memory_samples = [t["host_memory_gb"] for t in raw_trials if t["host_memory_gb"] is not None]
    peak_memory_gb = None
    if vram_samples:
        peak_memory_gb = max(vram_samples)
    elif host_memory_samples:
        peak_memory_gb = max(host_memory_samples)

    metrics: dict = {
        "latency_ms": latency_ms,
        "classifications_per_sec": classifications_per_sec,
        "trial_count": cfg.measured_trials,
        "input_chars": len(cfg.input_text),
    }
    if latency_sd is not None:
        metrics["latency_ms_stddev"] = latency_sd
    if latency_p95 is not None:
        metrics["latency_ms_p95"] = latency_p95
    if cfg.top_k is not None:
        metrics["top_k"] = cfg.top_k
    if peak_memory_gb is not None:
        metrics["peak_memory_gb"] = peak_memory_gb

    validity = _build_simple_validity(
        exclusivity, engine_ok, engine_reason, cfg.compute_engine, cfg.exclusive_run, cfg.warmup_trials
    )

    run_environment = _make_run_environment(system_info, environment, cfg.backend)
    checkpoint = _lookup_checkpoint(client, cfg.model_name)

    result: dict = {
        "schema_version": "0.1.0",
        "run_type": "classify",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": cfg.model_name,
            **({"checkpoint": checkpoint} if checkpoint else {}),
        },
        "config": {
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "os": cfg.os,
            "power_state": cfg.power_state,
            **({"power_cap_w": cfg.power_cap_w} if cfg.power_cap_w else {}),
        },
        "environment": run_environment,
        "metrics": metrics,
        "validity": validity,
    }
    return result, raw_trials


def _make_run_environment(system_info: dict, environment: dict, backend: str) -> dict:
    backend_version = lookup_backend_version(system_info, backend)
    return {**environment, "backend_version": backend_version} if backend_version else environment


def _lookup_checkpoint(client: LemonadeClient, model_name: str) -> str | None:
    try:
        return next((m.get("checkpoint") for m in client.models() if m.get("id") == model_name), None)
    except Exception:
        return None


def _exclusivity_verdict(exclusivity: "_ExclusivityMonitor", asserted: bool) -> tuple[bool, str | None]:
    """Shared exclusivity resolution used by run_sweep/run_classify/run_tts/
    run_stt/run_imagegen -- see run_sweep's original comment for the full
    rationale. Returns (exclusive_run, note-or-None)."""
    competing_models = sorted(exclusivity.competing_models)
    if exclusivity.health_unreachable:
        return asserted, "exclusive_run could not be verified (no successful health poll during the run); using the caller's assertion"
    if competing_models:
        return False, f"competing workload detected during measurement: {', '.join(competing_models)}"
    return asserted, (
        "exclusive_run verified via /api/v1/health polling (no other loaded model was busy)" if asserted else None
    )


def _device_verdict(
    exclusivity: "_ExclusivityMonitor", compute_engine: str, is_router: bool = False
) -> tuple[bool, str | None]:
    """Cross-checks the caller's compute_engine claim against Lemonade's own
    /api/v1/health-reported device for the model this run actually loaded.
    Router runs have no single physical device to verify (IDEA.md: the
    downstream model's engine varies per routed request and is captured in
    the trial sidecar instead), so they always pass. A device never observed
    this run (own_device_seen is None) is treated as unverified, not a
    mismatch -- absence of evidence is not evidence of a mismatch.
    """
    if is_router:
        return True, None
    if _device_matches_engine(exclusivity.own_device_seen, compute_engine) is False:
        return False, (
            f"compute_engine claimed '{compute_engine}' but Lemonade's own /api/v1/health reported "
            f"device '{exclusivity.own_device_seen}' for this model"
        )
    return True, None


def _quantization_verdict(
    client: LemonadeClient, model_name: str, claimed_quantization: str
) -> tuple[bool, str | None]:
    """Cross-checks the caller's claimed model.quantization against
    Lemonade's own checkpoint metadata for the model actually loaded, using
    the same best-effort parse_quantization() heuristic the dashboard's run
    form uses to pre-fill this field (there is no dedicated quantization
    field in Lemonade's own /api/v1/models response -- confirmed against its
    model_info_to_json source -- only a checkpoint string that sometimes
    encodes it). A model whose checkpoint doesn't parse to anything, or that
    Lemonade doesn't report back at all, is treated as unverified, not a
    mismatch.
    """
    try:
        models = client.models()
    except Exception:
        return True, None
    entry = next((m for m in models if m.get("id") == model_name), None)
    if not entry:
        return True, None
    parsed = parse_quantization(model_name, entry.get("checkpoint") or "")
    if not parsed:
        return True, None
    if parsed.lower() != (claimed_quantization or "").lower():
        return False, (
            f"quantization claimed '{claimed_quantization}' but Lemonade's own checkpoint metadata "
            f"implies '{parsed}'"
        )
    return True, None


def _watchdog_verdict(exclusivity: "_ExclusivityMonitor") -> tuple[bool, str | None]:
    """True (model_reload_free) unless Lemonade's own backend watchdog
    force-restarted the model's subprocess at some point during measurement
    -- previously this was unconditionally hardcoded True ("there is no
    unload here by design"), which is true of LemonMatrix's own calls but
    doesn't rule out Lemonade restarting the backend on its own after a
    crash or hang."""
    if exclusivity.watchdog_reset_seen:
        return False, (
            "Lemonade's backend watchdog force-restarted the model process during measurement -- "
            "results may reflect a restart, not steady-state inference"
        )
    return True, None


def _resolve_backend_load_kwargs(system_info: dict, backend: str) -> tuple[dict, str | None]:
    """Backend selector kwargs for /api/v1/load, resolved from cfg.backend.

    Without this, /api/v1/load has no backend selector at all and Lemonade
    silently falls back to the recipe's configured default_backend --
    confirmed live that whispercpp and sd-cpp both default to "cpu" despite
    offering cuda/rocm/vulkan backends, so skipping this for any recipe with
    "selectable_backend": true would silently benchmark the wrong backend.
    Only onnxruntime and kokoro are confirmed "selectable_backend": false
    (exactly one backend each) -- run_classify/run_tts skip this call
    entirely rather than call it and discard an always-empty result.
    """
    resolved = resolve_backend(system_info, backend)
    if resolved:
        recipe_key, backend_key = resolved
        return {f"{recipe_key}_backend": backend_key}, None
    return {}, (
        f"could not resolve backend '{backend}' to a recipe -- Lemonade used its "
        "recipe's default backend for this model instead"
    )


def _build_simple_validity(
    exclusivity: "_ExclusivityMonitor",
    engine_ok: bool,
    engine_reason: str,
    compute_engine: str,
    exclusive_run_asserted: bool,
    warmup_trials: int,
    backend_unresolved_note: str | None = None,
) -> dict:
    """Shared validity block for run_classify/run_tts/run_stt/run_imagegen --
    identical shape across all four (unlike run_sweep, which has extra
    thermal/power checks of its own). backend_unresolved_note is only ever
    set by run_stt/run_imagegen (see _resolve_backend_load_kwargs) --
    run_classify/run_tts never pass one, since onnxruntime/kokoro have
    nothing to resolve.
    """
    exclusive_run, exclusivity_note = _exclusivity_verdict(exclusivity, exclusive_run_asserted)
    device_ok, device_note = _device_verdict(exclusivity, compute_engine)
    model_reload_free, reload_note = _watchdog_verdict(exclusivity)

    notes = []
    if not engine_ok:
        notes.append(f"engine/backend mismatch: {engine_reason}")
    if backend_unresolved_note:
        notes.append(backend_unresolved_note)
    if device_note:
        notes.append(device_note)
    if reload_note:
        notes.append(reload_note)
    if exclusivity_note:
        notes.append(exclusivity_note)

    validity = {
        "valid": exclusive_run and engine_ok and device_ok and model_reload_free and not backend_unresolved_note,
        "warmup_discarded": warmup_trials > 0,
        "exclusive_run": exclusive_run,
        "model_reload_free": model_reload_free,
    }
    if notes:
        validity["notes"] = "; ".join(notes)
    return validity


def _wav_duration_seconds(wav_bytes: bytes) -> float:
    """Exact clip duration read directly from the RIFF/WAVE header via
    struct, not the stdlib `wave` module.

    Confirmed live that this can't just use `wave.open()`: kokoro's real
    /v1/audio/speech response is IEEE-float PCM (fmt tag 3, 32-bit float
    samples), and the stdlib `wave` module unconditionally rejects any
    format tag other than 1 (integer PCM) with "unknown format" -- even
    though duration only ever depends on sample_rate/channels/bits_per_sample
    /data size, none of which differ for float vs. integer PCM. Parsing the
    header directly avoids that stdlib limitation for any PCM variant.

    Also confirmed live: kokoro streams its response and never comes back to
    patch the header once the real length is known, so both the RIFF size
    and the "data" chunk size are written as the WAV "unknown length"
    sentinel, 0xFFFFFFFF -- trusting that value literally as a byte count
    (as a spec-following reader normally could) computed a ~44739s "clip"
    out of a one-sentence recording. When the declared data size is that
    sentinel (or otherwise overruns the actual buffer), the real length is
    just whatever bytes are actually left in the buffer.
    """
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")

    pos = 12
    sample_rate = num_channels = bits_per_sample = data_size = None
    while pos + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[pos : pos + 4]
        chunk_size = struct.unpack_from("<I", wav_bytes, pos + 4)[0]
        body = pos + 8
        if chunk_id == b"fmt ":
            _, num_channels, sample_rate, _, _, bits_per_sample = struct.unpack_from("<HHIIHH", wav_bytes, body)
        elif chunk_id == b"data":
            if chunk_size == 0xFFFFFFFF or body + chunk_size > len(wav_bytes):
                data_size = len(wav_bytes) - body
            else:
                data_size = chunk_size
        pos = body + chunk_size + (chunk_size % 2)  # chunks are word-aligned
        if sample_rate is not None and data_size is not None:
            break

    if sample_rate is None or data_size is None:
        raise ValueError("WAV file missing 'fmt ' or 'data' chunk")
    return data_size / (sample_rate * num_channels * (bits_per_sample / 8))


@dataclass
class TTSConfig:
    """Benchmarks a text-to-speech model (Lemonade's kokoro/openmoss recipes,
    POST /v1/audio/speech) -- see tts_result.schema.json for why this is a
    deliberately separate pipeline rather than a bolt-on run_type:
    real-time-factor isn't comparable to LLM token throughput.
    """

    model_name: str
    compute_engine: str
    backend: str
    os: str
    power_state: str
    input_text: str
    power_cap_w: float | None = None
    warmup_trials: int = 2
    measured_trials: int = 5
    voice: str | None = None
    speed: float | None = None
    exclusive_run: bool = True


def _run_one_tts(client: LemonadeClient, cfg: TTSConfig) -> dict:
    """Time one speech-generation request wall-clock, client-side, and read
    the generated clip's exact duration off its own WAV header.

    Confirmed against Lemonade's own server source (handle_audio_speech in
    src/cpp/server/server.cpp): the response is the raw audio body with no
    timing field anywhere, so -- same as classify -- there is nothing to
    read off the response except by timing the call ourselves. This times
    the HTTP call to whatever machine the profile's client actually points
    at, not local hardware.
    """
    start = time.perf_counter()
    audio_bytes = client.text_to_speech(
        cfg.input_text, model=cfg.model_name, voice=cfg.voice, speed=cfg.speed, response_format="wav"
    )
    generation_time_ms = (time.perf_counter() - start) * 1000
    resources = _resource_samples(client)
    return {
        "generation_time_ms": generation_time_ms,
        "audio_duration_s": _wav_duration_seconds(audio_bytes),
        "vram_gb": resources["vram_gb"],
        "host_memory_gb": resources["host_memory_gb"],
    }


def run_tts(client: LemonadeClient, environment: dict, cfg: TTSConfig) -> tuple[dict, list[dict]]:
    """Load the TTS model, warm up, run measured trials, and return
    (schema-conformant TTS result dict, list of raw per-trial measurements).
    """
    engine_ok, engine_reason = engine_backend_compatible(cfg.compute_engine, cfg.backend)

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    # Same reasoning as run_classify: kokoro's cpu backend is the only
    # backend Lemonade currently reports for that recipe ("selectable_backend":
    # false), so there is nothing to select at load time.
    client.load(cfg.model_name)

    exclusivity = _ExclusivityMonitor(client, cfg.model_name)
    with exclusivity:
        for _ in range(cfg.warmup_trials):
            _run_one_tts(client, cfg)
        raw_trials = [_run_one_tts(client, cfg) for _ in range(cfg.measured_trials)]

    gen_time_vals = [t["generation_time_ms"] for t in raw_trials]
    duration_vals = [t["audio_duration_s"] for t in raw_trials]
    generation_time_ms = _mean(gen_time_vals)
    audio_duration_s = _mean(duration_vals)
    gen_time_sd = _stddev(gen_time_vals)
    gen_time_p95 = _p95(gen_time_vals)
    real_time_factor = (
        audio_duration_s / (generation_time_ms / 1000)
        if audio_duration_s and generation_time_ms
        else None
    )

    vram_samples = [t["vram_gb"] for t in raw_trials if t["vram_gb"] is not None]
    host_memory_samples = [t["host_memory_gb"] for t in raw_trials if t["host_memory_gb"] is not None]
    peak_memory_gb = None
    if vram_samples:
        peak_memory_gb = max(vram_samples)
    elif host_memory_samples:
        peak_memory_gb = max(host_memory_samples)

    metrics: dict = {
        "generation_time_ms": generation_time_ms,
        "audio_duration_s": audio_duration_s,
        "real_time_factor": real_time_factor,
        "trial_count": cfg.measured_trials,
        "input_chars": len(cfg.input_text),
    }
    if gen_time_sd is not None:
        metrics["generation_time_ms_stddev"] = gen_time_sd
    if gen_time_p95 is not None:
        metrics["generation_time_ms_p95"] = gen_time_p95
    if cfg.voice is not None:
        metrics["voice"] = cfg.voice
    if cfg.speed is not None:
        metrics["speed"] = cfg.speed
    if peak_memory_gb is not None:
        metrics["peak_memory_gb"] = peak_memory_gb

    validity = _build_simple_validity(
        exclusivity, engine_ok, engine_reason, cfg.compute_engine, cfg.exclusive_run, cfg.warmup_trials
    )

    run_environment = _make_run_environment(system_info, environment, cfg.backend)
    checkpoint = _lookup_checkpoint(client, cfg.model_name)

    result: dict = {
        "schema_version": "0.1.0",
        "run_type": "tts",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": cfg.model_name,
            **({"checkpoint": checkpoint} if checkpoint else {}),
        },
        "config": {
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "os": cfg.os,
            "power_state": cfg.power_state,
            **({"power_cap_w": cfg.power_cap_w} if cfg.power_cap_w else {}),
        },
        "environment": run_environment,
        "metrics": metrics,
        "validity": validity,
    }
    return result, raw_trials


@dataclass
class STTConfig:
    """Benchmarks a speech-to-text model (Lemonade's whispercpp/moonshine
    recipes, POST /v1/audio/transcriptions) -- see stt_result.schema.json for
    why this is a deliberately separate pipeline: real-time-factor isn't
    comparable to LLM token throughput.

    audio_bytes must be a WAV file -- its exact duration is read from its
    own header (same approach as tts_result's output clip), and that duration
    is what real_time_factor is computed against.
    """

    model_name: str
    compute_engine: str
    backend: str
    os: str
    power_state: str
    audio_bytes: bytes
    audio_filename: str = "input.wav"
    language: str | None = None
    power_cap_w: float | None = None
    warmup_trials: int = 2
    measured_trials: int = 5
    exclusive_run: bool = True


def _run_one_stt(client: LemonadeClient, cfg: STTConfig) -> dict:
    """Time one transcription request wall-clock, client-side.

    Confirmed against Lemonade's own upstream test suite
    (test/server_whisper.py): the response ({"text": ...} for
    response_format="json") carries no timing field, so -- same as
    classify/tts -- there is nothing to read off the response except by
    timing the call ourselves.
    """
    start = time.perf_counter()
    client.speech_to_text(cfg.audio_bytes, cfg.audio_filename, model=cfg.model_name, language=cfg.language)
    transcription_time_ms = (time.perf_counter() - start) * 1000
    resources = _resource_samples(client)
    return {
        "transcription_time_ms": transcription_time_ms,
        "vram_gb": resources["vram_gb"],
        "host_memory_gb": resources["host_memory_gb"],
    }


def run_stt(client: LemonadeClient, environment: dict, cfg: STTConfig) -> tuple[dict, list[dict]]:
    """Load the STT model, warm up, run measured trials, and return
    (schema-conformant STT result dict, list of raw per-trial measurements).
    """
    engine_ok, engine_reason = engine_backend_compatible(cfg.compute_engine, cfg.backend)

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    audio_duration_s = _wav_duration_seconds(cfg.audio_bytes)

    load_kwargs, backend_unresolved_note = _resolve_backend_load_kwargs(system_info, cfg.backend)
    client.load(cfg.model_name, **load_kwargs)

    exclusivity = _ExclusivityMonitor(client, cfg.model_name)
    with exclusivity:
        for _ in range(cfg.warmup_trials):
            _run_one_stt(client, cfg)
        raw_trials = [_run_one_stt(client, cfg) for _ in range(cfg.measured_trials)]

    time_vals = [t["transcription_time_ms"] for t in raw_trials]
    transcription_time_ms = _mean(time_vals)
    time_sd = _stddev(time_vals)
    time_p95 = _p95(time_vals)
    real_time_factor = (
        audio_duration_s / (transcription_time_ms / 1000)
        if audio_duration_s and transcription_time_ms
        else None
    )

    vram_samples = [t["vram_gb"] for t in raw_trials if t["vram_gb"] is not None]
    host_memory_samples = [t["host_memory_gb"] for t in raw_trials if t["host_memory_gb"] is not None]
    peak_memory_gb = None
    if vram_samples:
        peak_memory_gb = max(vram_samples)
    elif host_memory_samples:
        peak_memory_gb = max(host_memory_samples)

    metrics: dict = {
        "transcription_time_ms": transcription_time_ms,
        "audio_duration_s": audio_duration_s,
        "real_time_factor": real_time_factor,
        "trial_count": cfg.measured_trials,
    }
    if time_sd is not None:
        metrics["transcription_time_ms_stddev"] = time_sd
    if time_p95 is not None:
        metrics["transcription_time_ms_p95"] = time_p95
    if cfg.language is not None:
        metrics["language"] = cfg.language
    if peak_memory_gb is not None:
        metrics["peak_memory_gb"] = peak_memory_gb

    validity = _build_simple_validity(
        exclusivity, engine_ok, engine_reason, cfg.compute_engine, cfg.exclusive_run, cfg.warmup_trials,
        backend_unresolved_note=backend_unresolved_note,
    )

    run_environment = _make_run_environment(system_info, environment, cfg.backend)
    checkpoint = _lookup_checkpoint(client, cfg.model_name)

    result: dict = {
        "schema_version": "0.1.0",
        "run_type": "stt",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": cfg.model_name,
            **({"checkpoint": checkpoint} if checkpoint else {}),
        },
        "config": {
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "os": cfg.os,
            "power_state": cfg.power_state,
            **({"power_cap_w": cfg.power_cap_w} if cfg.power_cap_w else {}),
        },
        "environment": run_environment,
        "metrics": metrics,
        "validity": validity,
    }
    return result, raw_trials


@dataclass
class ImageGenConfig:
    """Benchmarks an image-generation model (Lemonade's sd-cpp/thenoise
    recipes) across three operations sharing one pipeline since they run
    through the same model and produce the same metric shape -- see
    imagegen_result.schema.json's docstring for the full rationale (and for
    why /v1/images/upscale is deliberately NOT covered by any pipeline):

    - "generate" (default): POST /v1/images/generations, text-to-image.
      Requires `prompt`.
    - "edit": POST /v1/images/edits, prompt-guided edit of an input image.
      Requires `prompt` and `input_image_bytes`; `mask_bytes` optional.
    - "variation": POST /v1/images/variations, unguided variation of an
      input image. Requires `input_image_bytes` only -- Lemonade's own
      endpoint doesn't accept a prompt, mask, steps, cfg_scale, or seed at
      all for this operation (confirmed against its server source).

    steps defaults to 4 (SD-Turbo's own default, confirmed against
    Lemonade's upstream test suite) rather than None, since the schema
    requires a concrete step count for reproducibility on "generate"/"edit"
    -- generation time depends heavily on it. Not used for "variation".
    """

    model_name: str
    compute_engine: str
    backend: str
    os: str
    power_state: str
    operation: str = "generate"  # "generate" | "edit" | "variation"
    prompt: str = ""
    input_image_bytes: bytes | None = None
    input_image_filename: str = "input.png"
    mask_bytes: bytes | None = None
    mask_filename: str = "mask.png"
    image_size: str = "512x512"
    steps: int = 4
    cfg_scale: float | None = None
    seed: int | None = None
    power_cap_w: float | None = None
    warmup_trials: int = 2
    measured_trials: int = 5
    exclusive_run: bool = True


def _run_one_imagegen(client: LemonadeClient, cfg: ImageGenConfig) -> dict:
    """Time one image request wall-clock, client-side, dispatching to
    whichever of Lemonade's three image endpoints cfg.operation names.

    Confirmed against Lemonade's own upstream test suite (test/server_sd.py)
    and server source: every one of these responses ({"data":
    [{"b64_json": ...}], "created": <unix ts>}) carries no timing field --
    "created" is a timestamp, not a duration -- so, same as every other
    benchmarked modality, there is nothing to read off the response except
    by timing the call ourselves.
    """
    start = time.perf_counter()
    if cfg.operation == "edit":
        client.edit_image(
            cfg.input_image_bytes, cfg.input_image_filename, cfg.prompt, model=cfg.model_name,
            mask_bytes=cfg.mask_bytes, mask_filename=cfg.mask_filename, size=cfg.image_size,
            steps=cfg.steps, cfg_scale=cfg.cfg_scale, seed=cfg.seed,
        )
    elif cfg.operation == "variation":
        client.create_image_variation(
            cfg.input_image_bytes, cfg.input_image_filename, model=cfg.model_name, size=cfg.image_size,
        )
    else:
        client.generate_image(
            cfg.prompt, model=cfg.model_name, size=cfg.image_size, steps=cfg.steps,
            cfg_scale=cfg.cfg_scale, seed=cfg.seed,
        )
    generation_time_ms = (time.perf_counter() - start) * 1000
    resources = _resource_samples(client)
    return {
        "generation_time_ms": generation_time_ms,
        "vram_gb": resources["vram_gb"],
        "host_memory_gb": resources["host_memory_gb"],
    }


def run_imagegen(client: LemonadeClient, environment: dict, cfg: ImageGenConfig) -> tuple[dict, list[dict]]:
    """Load the image-generation model, warm up, run measured trials, and
    return (schema-conformant image-generation result dict, list of raw
    per-trial measurements).
    """
    engine_ok, engine_reason = engine_backend_compatible(cfg.compute_engine, cfg.backend)

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    load_kwargs, backend_unresolved_note = _resolve_backend_load_kwargs(system_info, cfg.backend)
    client.load(cfg.model_name, **load_kwargs)

    exclusivity = _ExclusivityMonitor(client, cfg.model_name)
    with exclusivity:
        for _ in range(cfg.warmup_trials):
            _run_one_imagegen(client, cfg)
        raw_trials = [_run_one_imagegen(client, cfg) for _ in range(cfg.measured_trials)]

    gen_time_vals = [t["generation_time_ms"] for t in raw_trials]
    generation_time_ms = _mean(gen_time_vals)
    gen_time_sd = _stddev(gen_time_vals)
    gen_time_p95 = _p95(gen_time_vals)
    images_per_sec = 1000 / generation_time_ms if generation_time_ms else None

    vram_samples = [t["vram_gb"] for t in raw_trials if t["vram_gb"] is not None]
    host_memory_samples = [t["host_memory_gb"] for t in raw_trials if t["host_memory_gb"] is not None]
    peak_memory_gb = None
    if vram_samples:
        peak_memory_gb = max(vram_samples)
    elif host_memory_samples:
        peak_memory_gb = max(host_memory_samples)

    metrics: dict = {
        "generation_time_ms": generation_time_ms,
        "images_per_sec": images_per_sec,
        "trial_count": cfg.measured_trials,
        "operation": cfg.operation,
        "image_size": cfg.image_size,
    }
    if gen_time_sd is not None:
        metrics["generation_time_ms_stddev"] = gen_time_sd
    if gen_time_p95 is not None:
        metrics["generation_time_ms_p95"] = gen_time_p95
    # steps/cfg_scale/seed/prompt are not accepted at all by Lemonade's own
    # /v1/images/variations (confirmed against its server source) -- reporting
    # them for that operation would misrepresent what was actually requested.
    if cfg.operation != "variation":
        metrics["steps"] = cfg.steps
        metrics["prompt_chars"] = len(cfg.prompt)
        if cfg.cfg_scale is not None:
            metrics["cfg_scale"] = cfg.cfg_scale
        if cfg.seed is not None:
            metrics["seed"] = cfg.seed
    if cfg.operation == "edit":
        metrics["has_mask"] = cfg.mask_bytes is not None
    if peak_memory_gb is not None:
        metrics["peak_memory_gb"] = peak_memory_gb

    validity = _build_simple_validity(
        exclusivity, engine_ok, engine_reason, cfg.compute_engine, cfg.exclusive_run, cfg.warmup_trials,
        backend_unresolved_note=backend_unresolved_note,
    )

    run_environment = _make_run_environment(system_info, environment, cfg.backend)
    checkpoint = _lookup_checkpoint(client, cfg.model_name)

    result: dict = {
        "schema_version": "0.1.0",
        "run_type": "imagegen",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": cfg.model_name,
            **({"checkpoint": checkpoint} if checkpoint else {}),
        },
        "config": {
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "os": cfg.os,
            "power_state": cfg.power_state,
            **({"power_cap_w": cfg.power_cap_w} if cfg.power_cap_w else {}),
        },
        "environment": run_environment,
        "metrics": metrics,
        "validity": validity,
    }
    return result, raw_trials


@dataclass
class AudioGenConfig:
    """Benchmarks an audio-generation model (Lemonade's acestep/thinksound
    recipes, POST /v1/audio/generations) -- text/music/sound-effect
    generation, NOT speech (see TTSConfig/run_tts for that). See
    audiogen_result.schema.json for why this is a deliberately separate
    pipeline from both result.schema.json (real-time-factor isn't LLM token
    throughput) and tts_result.schema.json (a different task, different
    compute profile).
    """

    model_name: str
    compute_engine: str
    backend: str
    os: str
    power_state: str
    prompt: str
    lyrics: str | None = None
    vocal_language: str | None = None
    power_cap_w: float | None = None
    warmup_trials: int = 2
    measured_trials: int = 5
    exclusive_run: bool = True


def _run_one_audiogen(client: LemonadeClient, cfg: AudioGenConfig) -> dict:
    """Time one audio-generation request wall-clock, client-side, and read
    the generated clip's exact duration off its own WAV header -- same
    approach as _run_one_tts, since /v1/audio/generations has the identical
    no-timing-field situation confirmed against Lemonade's own server
    source (handle_audio_generations in src/cpp/server/server.cpp).
    """
    start = time.perf_counter()
    audio_bytes = client.generate_audio(
        cfg.prompt, model=cfg.model_name, lyrics=cfg.lyrics, vocal_language=cfg.vocal_language,
        response_format="wav",
    )
    generation_time_ms = (time.perf_counter() - start) * 1000
    resources = _resource_samples(client)
    return {
        "generation_time_ms": generation_time_ms,
        "audio_duration_s": _wav_duration_seconds(audio_bytes),
        "vram_gb": resources["vram_gb"],
        "host_memory_gb": resources["host_memory_gb"],
    }


def run_audiogen(client: LemonadeClient, environment: dict, cfg: AudioGenConfig) -> tuple[dict, list[dict]]:
    """Load the audio-generation model, warm up, run measured trials, and
    return (schema-conformant audio-generation result dict, list of raw
    per-trial measurements).
    """
    engine_ok, engine_reason = engine_backend_compatible(cfg.compute_engine, cfg.backend)

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    # Unlike onnxruntime/kokoro, acestep and thinksound both report
    # "selectable_backend": true with several real backends (cuda/rocm/
    # vulkan) -- confirmed live -- so the backend selector must actually be
    # resolved and passed, same as run_sweep/run_stt/run_imagegen.
    load_kwargs, backend_unresolved_note = _resolve_backend_load_kwargs(system_info, cfg.backend)
    client.load(cfg.model_name, **load_kwargs)

    exclusivity = _ExclusivityMonitor(client, cfg.model_name)
    with exclusivity:
        for _ in range(cfg.warmup_trials):
            _run_one_audiogen(client, cfg)
        raw_trials = [_run_one_audiogen(client, cfg) for _ in range(cfg.measured_trials)]

    gen_time_vals = [t["generation_time_ms"] for t in raw_trials]
    duration_vals = [t["audio_duration_s"] for t in raw_trials]
    generation_time_ms = _mean(gen_time_vals)
    audio_duration_s = _mean(duration_vals)
    gen_time_sd = _stddev(gen_time_vals)
    gen_time_p95 = _p95(gen_time_vals)
    real_time_factor = (
        audio_duration_s / (generation_time_ms / 1000)
        if audio_duration_s and generation_time_ms
        else None
    )

    vram_samples = [t["vram_gb"] for t in raw_trials if t["vram_gb"] is not None]
    host_memory_samples = [t["host_memory_gb"] for t in raw_trials if t["host_memory_gb"] is not None]
    peak_memory_gb = None
    if vram_samples:
        peak_memory_gb = max(vram_samples)
    elif host_memory_samples:
        peak_memory_gb = max(host_memory_samples)

    metrics: dict = {
        "generation_time_ms": generation_time_ms,
        "audio_duration_s": audio_duration_s,
        "real_time_factor": real_time_factor,
        "trial_count": cfg.measured_trials,
        "prompt_chars": len(cfg.prompt),
    }
    if gen_time_sd is not None:
        metrics["generation_time_ms_stddev"] = gen_time_sd
    if gen_time_p95 is not None:
        metrics["generation_time_ms_p95"] = gen_time_p95
    if cfg.lyrics is not None:
        metrics["lyrics_chars"] = len(cfg.lyrics)
    if cfg.vocal_language is not None:
        metrics["vocal_language"] = cfg.vocal_language
    if peak_memory_gb is not None:
        metrics["peak_memory_gb"] = peak_memory_gb

    validity = _build_simple_validity(
        exclusivity, engine_ok, engine_reason, cfg.compute_engine, cfg.exclusive_run, cfg.warmup_trials,
        backend_unresolved_note=backend_unresolved_note,
    )

    run_environment = _make_run_environment(system_info, environment, cfg.backend)
    checkpoint = _lookup_checkpoint(client, cfg.model_name)

    result: dict = {
        "schema_version": "0.1.0",
        "run_type": "audiogen",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": cfg.model_name,
            **({"checkpoint": checkpoint} if checkpoint else {}),
        },
        "config": {
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "os": cfg.os,
            "power_state": cfg.power_state,
            **({"power_cap_w": cfg.power_cap_w} if cfg.power_cap_w else {}),
        },
        "environment": run_environment,
        "metrics": metrics,
        "validity": validity,
    }
    return result, raw_trials


@dataclass
class EmbeddingsConfig:
    """Benchmarks a batched-embedding request (llamacpp GGUF or FastFlowLM
    NPU, POST /v1/embeddings) -- see embeddings_result.schema.json for why
    this is a deliberately separate pipeline: embeddings throughput isn't
    comparable to LLM token throughput.
    """

    model_name: str
    compute_engine: str
    backend: str
    os: str
    power_state: str
    input_texts: list[str]
    power_cap_w: float | None = None
    warmup_trials: int = 2
    measured_trials: int = 5
    exclusive_run: bool = True


def _run_one_embeddings(client: LemonadeClient, cfg: EmbeddingsConfig) -> dict:
    """Time one embeddings request wall-clock, client-side, and read the
    returned embedding dimensionality off the response.

    Confirmed against Lemonade's own upstream test suite (test/server_llm.py):
    the response is a pure passthrough of llama.cpp's own /v1/embeddings --
    {"data": [{"embedding": [...]}], "usage": {...}} -- with no timing field,
    so, same as the other benchmarked modalities, there is nothing to read
    off the response except by timing the call ourselves.
    """
    start = time.perf_counter()
    response = client.get_embeddings(cfg.input_texts, model=cfg.model_name)
    latency_ms = (time.perf_counter() - start) * 1000
    resources = _resource_samples(client)
    data = response.get("data") if isinstance(response, dict) else None
    first = data[0] if data else {}
    embedding_dim = len(first.get("embedding") or []) if isinstance(first, dict) else 0
    return {
        "latency_ms": latency_ms,
        "embedding_dim": embedding_dim,
        "vram_gb": resources["vram_gb"],
        "host_memory_gb": resources["host_memory_gb"],
    }


def run_embeddings(
    client: LemonadeClient, environment: dict, cfg: EmbeddingsConfig
) -> tuple[dict, list[dict]]:
    """Load the model, warm up, run measured trials, and return
    (schema-conformant embeddings result dict, list of raw per-trial measurements).
    """
    engine_ok, engine_reason = engine_backend_compatible(cfg.compute_engine, cfg.backend)

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    load_kwargs, backend_unresolved_note = _resolve_backend_load_kwargs(system_info, cfg.backend)
    client.load(cfg.model_name, **load_kwargs)

    exclusivity = _ExclusivityMonitor(client, cfg.model_name)
    with exclusivity:
        for _ in range(cfg.warmup_trials):
            _run_one_embeddings(client, cfg)
        raw_trials = [_run_one_embeddings(client, cfg) for _ in range(cfg.measured_trials)]

    latency_vals = [t["latency_ms"] for t in raw_trials]
    latency_ms = _mean(latency_vals)
    latency_sd = _stddev(latency_vals)
    latency_p95 = _p95(latency_vals)
    batch_size = len(cfg.input_texts)
    embeddings_per_sec = batch_size / (latency_ms / 1000) if latency_ms else None

    embedding_dims = {t["embedding_dim"] for t in raw_trials if t["embedding_dim"]}
    embedding_dim = next(iter(embedding_dims)) if len(embedding_dims) == 1 else None

    vram_samples = [t["vram_gb"] for t in raw_trials if t["vram_gb"] is not None]
    host_memory_samples = [t["host_memory_gb"] for t in raw_trials if t["host_memory_gb"] is not None]
    peak_memory_gb = None
    if vram_samples:
        peak_memory_gb = max(vram_samples)
    elif host_memory_samples:
        peak_memory_gb = max(host_memory_samples)

    metrics: dict = {
        "latency_ms": latency_ms,
        "embeddings_per_sec": embeddings_per_sec,
        "trial_count": cfg.measured_trials,
        "batch_size": batch_size,
        "input_chars_total": sum(len(t) for t in cfg.input_texts),
    }
    if latency_sd is not None:
        metrics["latency_ms_stddev"] = latency_sd
    if latency_p95 is not None:
        metrics["latency_ms_p95"] = latency_p95
    if embedding_dim is not None:
        metrics["embedding_dim"] = embedding_dim
    if peak_memory_gb is not None:
        metrics["peak_memory_gb"] = peak_memory_gb

    validity = _build_simple_validity(
        exclusivity, engine_ok, engine_reason, cfg.compute_engine, cfg.exclusive_run, cfg.warmup_trials,
        backend_unresolved_note=backend_unresolved_note,
    )

    run_environment = _make_run_environment(system_info, environment, cfg.backend)
    checkpoint = _lookup_checkpoint(client, cfg.model_name)

    result: dict = {
        "schema_version": "0.1.0",
        "run_type": "embeddings",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": cfg.model_name,
            **({"checkpoint": checkpoint} if checkpoint else {}),
        },
        "config": {
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "os": cfg.os,
            "power_state": cfg.power_state,
            **({"power_cap_w": cfg.power_cap_w} if cfg.power_cap_w else {}),
        },
        "environment": run_environment,
        "metrics": metrics,
        "validity": validity,
    }
    return result, raw_trials


@dataclass
class RerankConfig:
    """Benchmarks a reranking request (llamacpp GGUF only, POST /v1/rerank)
    -- see rerank_result.schema.json for why this is a deliberately separate
    pipeline: reranking throughput isn't comparable to LLM token throughput.
    """

    model_name: str
    compute_engine: str
    backend: str
    os: str
    power_state: str
    query: str
    documents: list[str]
    top_n: int | None = None
    power_cap_w: float | None = None
    warmup_trials: int = 2
    measured_trials: int = 5
    exclusive_run: bool = True


def _run_one_rerank(client: LemonadeClient, cfg: RerankConfig) -> dict:
    """Time one reranking request wall-clock, client-side.

    Confirmed against Lemonade's own upstream test suite (test/server_llm.py):
    the response is a pure passthrough of llama.cpp's own /v1/rerank --
    {"results": [{"index": ..., "relevance_score": ...}]} -- with no timing
    field, so, same as the other benchmarked modalities, there is nothing to
    read off the response except by timing the call ourselves.
    """
    start = time.perf_counter()
    client.rerank(cfg.query, cfg.documents, model=cfg.model_name, top_n=cfg.top_n)
    latency_ms = (time.perf_counter() - start) * 1000
    resources = _resource_samples(client)
    return {
        "latency_ms": latency_ms,
        "vram_gb": resources["vram_gb"],
        "host_memory_gb": resources["host_memory_gb"],
    }


def run_rerank(client: LemonadeClient, environment: dict, cfg: RerankConfig) -> tuple[dict, list[dict]]:
    """Load the model, warm up, run measured trials, and return
    (schema-conformant rerank result dict, list of raw per-trial measurements).
    """
    engine_ok, engine_reason = engine_backend_compatible(cfg.compute_engine, cfg.backend)

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    load_kwargs, backend_unresolved_note = _resolve_backend_load_kwargs(system_info, cfg.backend)
    client.load(cfg.model_name, **load_kwargs)

    exclusivity = _ExclusivityMonitor(client, cfg.model_name)
    with exclusivity:
        for _ in range(cfg.warmup_trials):
            _run_one_rerank(client, cfg)
        raw_trials = [_run_one_rerank(client, cfg) for _ in range(cfg.measured_trials)]

    latency_vals = [t["latency_ms"] for t in raw_trials]
    latency_ms = _mean(latency_vals)
    latency_sd = _stddev(latency_vals)
    latency_p95 = _p95(latency_vals)
    document_count = len(cfg.documents)
    documents_per_sec = document_count / (latency_ms / 1000) if latency_ms else None

    vram_samples = [t["vram_gb"] for t in raw_trials if t["vram_gb"] is not None]
    host_memory_samples = [t["host_memory_gb"] for t in raw_trials if t["host_memory_gb"] is not None]
    peak_memory_gb = None
    if vram_samples:
        peak_memory_gb = max(vram_samples)
    elif host_memory_samples:
        peak_memory_gb = max(host_memory_samples)

    metrics: dict = {
        "latency_ms": latency_ms,
        "documents_per_sec": documents_per_sec,
        "trial_count": cfg.measured_trials,
        "document_count": document_count,
    }
    if latency_sd is not None:
        metrics["latency_ms_stddev"] = latency_sd
    if latency_p95 is not None:
        metrics["latency_ms_p95"] = latency_p95
    if cfg.top_n is not None:
        metrics["top_n"] = cfg.top_n
    if peak_memory_gb is not None:
        metrics["peak_memory_gb"] = peak_memory_gb

    validity = _build_simple_validity(
        exclusivity, engine_ok, engine_reason, cfg.compute_engine, cfg.exclusive_run, cfg.warmup_trials,
        backend_unresolved_note=backend_unresolved_note,
    )

    run_environment = _make_run_environment(system_info, environment, cfg.backend)
    checkpoint = _lookup_checkpoint(client, cfg.model_name)

    result: dict = {
        "schema_version": "0.1.0",
        "run_type": "rerank",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": cfg.model_name,
            **({"checkpoint": checkpoint} if checkpoint else {}),
        },
        "config": {
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "os": cfg.os,
            "power_state": cfg.power_state,
            **({"power_cap_w": cfg.power_cap_w} if cfg.power_cap_w else {}),
        },
        "environment": run_environment,
        "metrics": metrics,
        "validity": validity,
    }
    return result, raw_trials


@dataclass
class MeshGenConfig:
    """Benchmarks an image-to-3D-mesh model (Lemonade's trellis recipe,
    POST /v1/3d/generations) -- see meshgen_result.schema.json for why this
    is a deliberately separate pipeline: mesh-generation throughput isn't
    comparable to LLM token throughput.
    """

    model_name: str
    compute_engine: str
    backend: str
    os: str
    power_state: str
    input_image_bytes: bytes
    resolution: str | None = None
    bg_removal: str | None = None
    seed: int | None = None
    uv: str | None = None
    power_cap_w: float | None = None
    warmup_trials: int = 2
    measured_trials: int = 5
    exclusive_run: bool = True


def _run_one_meshgen(client: LemonadeClient, cfg: MeshGenConfig) -> dict:
    """Time one mesh-generation request wall-clock, client-side.

    Confirmed against Lemonade's own server source (handle_3d_generations in
    src/cpp/server/server.cpp): the response is the raw mesh binary itself,
    not a JSON envelope, and carries no timing field anywhere -- same as
    every other benchmarked modality, there is nothing to read off the
    response except by timing the call ourselves.
    """
    start = time.perf_counter()
    client.generate_3d(
        cfg.input_image_bytes, model=cfg.model_name, resolution=cfg.resolution,
        bg_removal=cfg.bg_removal, seed=cfg.seed, uv=cfg.uv,
    )
    generation_time_ms = (time.perf_counter() - start) * 1000
    resources = _resource_samples(client)
    return {
        "generation_time_ms": generation_time_ms,
        "vram_gb": resources["vram_gb"],
        "host_memory_gb": resources["host_memory_gb"],
    }


def run_meshgen(client: LemonadeClient, environment: dict, cfg: MeshGenConfig) -> tuple[dict, list[dict]]:
    """Load the mesh-generation model, warm up, run measured trials, and
    return (schema-conformant meshgen result dict, list of raw per-trial measurements).
    """
    engine_ok, engine_reason = engine_backend_compatible(cfg.compute_engine, cfg.backend)

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    load_kwargs, backend_unresolved_note = _resolve_backend_load_kwargs(system_info, cfg.backend)
    client.load(cfg.model_name, **load_kwargs)

    exclusivity = _ExclusivityMonitor(client, cfg.model_name)
    with exclusivity:
        for _ in range(cfg.warmup_trials):
            _run_one_meshgen(client, cfg)
        raw_trials = [_run_one_meshgen(client, cfg) for _ in range(cfg.measured_trials)]

    gen_time_vals = [t["generation_time_ms"] for t in raw_trials]
    generation_time_ms = _mean(gen_time_vals)
    gen_time_sd = _stddev(gen_time_vals)
    gen_time_p95 = _p95(gen_time_vals)
    meshes_per_sec = 1000 / generation_time_ms if generation_time_ms else None

    vram_samples = [t["vram_gb"] for t in raw_trials if t["vram_gb"] is not None]
    host_memory_samples = [t["host_memory_gb"] for t in raw_trials if t["host_memory_gb"] is not None]
    peak_memory_gb = None
    if vram_samples:
        peak_memory_gb = max(vram_samples)
    elif host_memory_samples:
        peak_memory_gb = max(host_memory_samples)

    metrics: dict = {
        "generation_time_ms": generation_time_ms,
        "meshes_per_sec": meshes_per_sec,
        "trial_count": cfg.measured_trials,
    }
    if gen_time_sd is not None:
        metrics["generation_time_ms_stddev"] = gen_time_sd
    if gen_time_p95 is not None:
        metrics["generation_time_ms_p95"] = gen_time_p95
    if cfg.resolution is not None:
        metrics["resolution"] = cfg.resolution
    if cfg.bg_removal is not None:
        metrics["bg_removal"] = cfg.bg_removal
    if cfg.uv is not None:
        metrics["uv"] = cfg.uv
    if cfg.seed is not None:
        metrics["seed"] = cfg.seed
    if peak_memory_gb is not None:
        metrics["peak_memory_gb"] = peak_memory_gb

    validity = _build_simple_validity(
        exclusivity, engine_ok, engine_reason, cfg.compute_engine, cfg.exclusive_run, cfg.warmup_trials,
        backend_unresolved_note=backend_unresolved_note,
    )

    run_environment = _make_run_environment(system_info, environment, cfg.backend)
    checkpoint = _lookup_checkpoint(client, cfg.model_name)

    result: dict = {
        "schema_version": "0.1.0",
        "run_type": "meshgen",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "name": cfg.model_name,
            **({"checkpoint": checkpoint} if checkpoint else {}),
        },
        "config": {
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "os": cfg.os,
            "power_state": cfg.power_state,
            **({"power_cap_w": cfg.power_cap_w} if cfg.power_cap_w else {}),
        },
        "environment": run_environment,
        "metrics": metrics,
        "validity": validity,
    }
    return result, raw_trials
