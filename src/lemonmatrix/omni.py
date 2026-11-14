"""Omni workflow benchmark: a Lemonade ``collection.omni`` model end to end.

An omni model (e.g. LMX-Omni-5.5B-Lite: an LLM planner plus image, speech
and transcription models) is addressed like any chat model. Lemonade runs the
tool loop server-side and embeds the generated media in the assistant message
as data URIs (docs/dev/lemonade-omni.md, "Server-Side Orchestration"). The
response carries no per-tool timing, so this measures what a user sees: the
end-to-end time of each task, and whether the task produced the media it asked
for (an image, a clip, or both).

Kept apart from result.schema.json: a task's latency covers several models and
tool calls, so it is not comparable to LLM token throughput.
"""

from __future__ import annotations

import re
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from .bench import (
    _build_simple_validity,
    _ExclusivityMonitor,
    _make_run_environment,
    _mean,
    _p95,
    _resource_samples,
    _stddev,
)
from .client import LemonadeAPIError, LemonadeClient

OMNI_RECIPE = "collection.omni"

# Each task says which media the answer must contain. "text" tasks need no
# tool at all; they show the planner's own overhead.
OMNI_TASK_SETS: dict[str, list[dict]] = {
    "omni-basic": [
        {"category": "text", "prompt": "What is the capital of Canada?", "expects": {}},
        {"category": "text", "prompt": "Give me three name ideas for a lemonade stand.", "expects": {}},
        {"category": "image", "prompt": "Draw a red apple on a wooden table.", "expects": {"image": 1}},
        {"category": "image", "prompt": "Create an image of a lighthouse at sunset.", "expects": {"image": 1}},
        {"category": "speech", "prompt": "Say 'good morning, Lemonade' out loud.", "expects": {"audio": 1}},
        {"category": "speech", "prompt": "Read this aloud: the quick brown fox jumps over the lazy dog.",
         "expects": {"audio": 1}},
        {"category": "image+speech",
         "prompt": "Write a two-line poem about lemons, read it aloud, and draw a lemon tree.",
         "expects": {"image": 1, "audio": 1}},
        {"category": "image+speech", "prompt": "Draw a cat wearing sunglasses, then describe it out loud.",
         "expects": {"image": 1, "audio": 1}},
    ],
}

_IMAGE = re.compile(r"data:image/[a-z0-9.+-]+;base64,")
_AUDIO = re.compile(r"data:audio/[a-z0-9.+-]+;base64,")
_DATA_URI = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+")

# Above this share of failed requests a run is not ranked (same rule as model runs).
MAX_FAILED_SHARE = 0.05


@dataclass
class OmniConfig:
    model_name: str
    os: str
    power_state: str
    task_set: str = "omni-basic"
    warmup_passes: int = 1
    measured_passes: int = 3
    max_tokens: int = 512
    power_cap_w: float | None = None
    exclusive_run: bool = True
    unload_others: bool = False


def media_counts(content: str) -> dict[str, int]:
    """Images and audio clips embedded in an omni answer."""
    return {"image": len(_IMAGE.findall(content)), "audio": len(_AUDIO.findall(content))}


def task_succeeded(expects: dict[str, int], counts: dict[str, int], content: str) -> bool:
    """Every expected medium is there; a text task needs a non-empty answer."""
    if not expects:
        return bool(_DATA_URI.sub("", content).strip())
    return all(counts.get(kind, 0) >= n for kind, n in expects.items())


def _run_task(client: LemonadeClient, cfg: OmniConfig, task: dict) -> dict:
    start = time.perf_counter()
    response = client.chat_completion(
        model=cfg.model_name, messages=[{"role": "user", "content": task["prompt"]}], max_tokens=cfg.max_tokens,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    content = ((response.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    counts = media_counts(content)
    resources = _resource_samples(client)
    return {
        "category": task["category"],
        "latency_ms": latency_ms,
        "images": counts["image"],
        "audio": counts["audio"],
        "success": task_succeeded(task["expects"], counts, content),
        "vram_gb": resources["vram_gb"],
        "host_memory_gb": resources["host_memory_gb"],
    }


def _latency_block(values: list[float]) -> dict:
    block = {"latency_ms": _mean(values), "latency_ms_p50": statistics.median(values)}
    p95, sd = _p95(values), _stddev(values)
    if p95 is not None:
        block["latency_ms_p95"] = p95
    if sd is not None:
        block["latency_ms_stddev"] = sd
    return block


def run_omni(client: LemonadeClient, environment: dict, cfg: OmniConfig) -> tuple[dict, list[dict]]:
    """Run every task of the set warmup_passes times (discarded), then
    measured_passes times, and return (schema-conformant result, raw trials)."""
    tasks = OMNI_TASK_SETS[cfg.task_set]
    models = client.models()
    entry = next((m for m in models if m.get("id") == cfg.model_name), None)
    if entry is None or entry.get("recipe") != OMNI_RECIPE:
        raise ValueError(f"{cfg.model_name} is not a {OMNI_RECIPE} model on this instance")
    components = list(entry.get("components") or [])

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}

    if cfg.unload_others:
        client.unload()

    exclusivity = _ExclusivityMonitor(client, {cfg.model_name, *components})
    raw_trials: list[dict] = []
    failed: list[dict] = []
    with exclusivity:
        for _ in range(cfg.warmup_passes):
            for task in tasks:
                try:
                    _run_task(client, cfg, task)
                except requests.RequestException:
                    pass  # a warmup only has to load the components
        for p in range(cfg.measured_passes):
            for i, task in enumerate(tasks):
                try:
                    trial = _run_task(client, cfg, task)
                except requests.RequestException as exc:
                    # Lemonade's own message; other errors by type only (their text carries the address).
                    error = str(exc) if isinstance(exc, LemonadeAPIError) else type(exc).__name__
                    failed.append({"task": i, "category": task["category"], "error": error[:200]})
                    continue
                trial["task_index"] = i
                trial["pass"] = p
                raw_trials.append(trial)
    if not raw_trials:
        raise RuntimeError(f"every omni request failed (first: {failed[0]['error'] if failed else 'none run'})")

    attempted = len(raw_trials) + len(failed)
    latencies = [t["latency_ms"] for t in raw_trials]
    by_category: dict[str, dict] = {}
    for category in dict.fromkeys(t["category"] for t in tasks):
        rows = [t for t in raw_trials if t["category"] == category]
        n_failed = sum(1 for f in failed if f["category"] == category)
        if not rows:
            continue
        by_category[category] = {
            "requests": len(rows) + n_failed,
            "success_rate": sum(t["success"] for t in rows) / (len(rows) + n_failed),
            **_latency_block([t["latency_ms"] for t in rows]),
            "images": sum(t["images"] for t in rows),
            "audio": sum(t["audio"] for t in rows),
        }

    mean_latency = _mean(latencies)
    metrics: dict = {
        "task_set": cfg.task_set,
        "task_count": len(tasks),
        "trial_count": len(raw_trials),
        "success_rate": sum(t["success"] for t in raw_trials) / attempted,
        "tasks_per_minute": 60000 / mean_latency if mean_latency else 0.0,
        **_latency_block(latencies),
        "by_category": by_category,
    }
    vram = [t["vram_gb"] for t in raw_trials if t["vram_gb"] is not None]
    host = [t["host_memory_gb"] for t in raw_trials if t["host_memory_gb"] is not None]
    if vram or host:
        metrics["peak_memory_gb"] = max(vram or host)

    # No single engine or backend: each component runs on its own. The device
    # check has nothing to compare against, so "omni" passes it as unknown.
    validity = _build_simple_validity(exclusivity, True, "", "omni", cfg.exclusive_run, cfg.warmup_passes)
    if failed:
        failures_ok = len(failed) / attempted <= MAX_FAILED_SHARE
        note = (f"{len(failed)} of {attempted} omni requests failed and count as unsuccessful"
                f"{'' if failures_ok else ' -- not ranked'} (first: {failed[0]['error']})")
        validity["notes"] = f"{validity['notes']}; {note}" if validity.get("notes") else note
        validity["valid"] = validity["valid"] and failures_ok

    result = {
        "schema_version": "0.1.0",
        "run_type": "omni",
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {"name": cfg.model_name, "components": components},
        "config": {
            "os": cfg.os,
            "power_state": cfg.power_state,
            "max_tokens": cfg.max_tokens,
            **({"power_cap_w": cfg.power_cap_w} if cfg.power_cap_w else {}),
        },
        "environment": _make_run_environment(system_info, environment, "collection.omni"),
        "metrics": metrics,
        "validity": validity,
    }
    return result, raw_trials
