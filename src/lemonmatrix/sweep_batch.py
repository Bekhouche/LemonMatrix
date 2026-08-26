"""Runs a queue of benchmark sweeps in the background.

Lemonade holds only one model in its LLM slot at a time (confirmed live:
`max_models: {"llm": 1, ...}`), so "run all these backends" can't be
parallelized against a single profile -- it has to be a sequential queue.
This makes IDEA.md's sweep model literal: one profile, many result rows,
run one at a time, each row independent (IDEA.md: an invalid run is shown
but never ranked, not one that aborts the batch).
"""

from __future__ import annotations

import itertools
import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .bench import SweepConfig, run_sweep, run_sweep_via_job
from .client import BENCH_TIMEOUT, LemonadeClient
from .results_store import save_failure, save_trials
from .validate import validate_result

MAX_SWEEP_COMBINATIONS = 200


def expand_combinations(
    quant_variants: list[dict], engines: list[str], backends: list[str], power_states: list[str]
) -> list[dict]:
    """Cartesian product of the sweepable axes.

    A pulled model IS a fixed quantization, so each quant variant carries
    its own model_name/context_length rather than those being independent
    axes -- only engine/backend/power_state actually multiply out.
    """
    combos = []
    for variant, engine, backend, power_state in itertools.product(quant_variants, engines, backends, power_states):
        combos.append(
            {
                "model_name": variant["id"],
                "quantization": variant["quantization"],
                "context_length": variant["context_length"],
                "compute_engine": engine,
                "backend": backend,
                "power_state": power_state,
            }
        )
    return combos


class SweepBatch:
    """One queued set of runs against a single profile.

    Mutated in place by the background thread; a polling dashboard page
    reads `items` for live progress. Plain-dict items and simple attribute
    writes are enough here -- CPython's GIL makes these individually atomic,
    and the only reader (a GET handler) only ever needs the latest snapshot,
    not a consistent view across multiple items.
    """

    def __init__(self, profile_name: str, combos: list[dict]):
        self.id = uuid.uuid4().hex[:12]
        self.profile_name = profile_name
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = "running"  # running | done | interrupted
        self.items = [{**combo, "status": "pending", "run_id": None, "error": None} for combo in combos]

    @property
    def completed_count(self) -> int:
        return sum(1 for i in self.items if i["status"] in ("completed", "failed"))

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def failed_count(self) -> int:
        return sum(1 for i in self.items if i["status"] == "failed")


def _run_batch(
    batch: SweepBatch,
    base_url: str,
    api_key: str | None,
    environment: dict,
    results_dir: Path,
    model_class: str,
    os_name: str,
    power_cap_w: float | None,
    warmup_trials: int,
    measured_trials: int,
    max_tokens: int,
    exclusive_run: bool,
    energy_price_usd_per_kwh: float | None,
    hardware_cost_usd: float | None = None,
    hardware_lifetime_hours: float | None = None,
    via_job_engine: bool = False,
    run_type: str = "model",
    store=None,  # optional SweepStore; avoids circular import with type annotation
) -> None:
    client = LemonadeClient(base_url, api_key=api_key, timeout=BENCH_TIMEOUT)
    out_dir = Path(results_dir) / batch.profile_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if store is not None:
        store.save_batch(batch)

    for idx, item in enumerate(batch.items):
        item["status"] = "running"
        if store is not None:
            store.update_item(batch.id, idx, item)
        try:
            # A hand-built queue (webapp's /queue route) can mix model
            # classes/run types across items -- e.g. a dense chat model next
            # to a router -- so each item's own "model_class"/"run_type", if
            # present, wins over the batch-level default a Cartesian sweep
            # applies uniformly. Absent keys (every combo expand_combinations()
            # produces) fall back to the batch-level value, unchanged.
            cfg = SweepConfig(
                model_name=item["model_name"],
                model_class=item.get("model_class", model_class),
                quantization=item["quantization"],
                context_length=item["context_length"],
                compute_engine=item["compute_engine"],
                backend=item["backend"],
                os=os_name,
                power_state=item["power_state"],
                power_cap_w=power_cap_w,
                warmup_trials=warmup_trials,
                measured_trials=measured_trials,
                max_tokens=max_tokens,
                exclusive_run=exclusive_run,
                energy_price_usd_per_kwh=energy_price_usd_per_kwh,
                hardware_cost_usd=hardware_cost_usd,
                hardware_lifetime_hours=hardware_lifetime_hours,
                run_type=item.get("run_type", run_type),
            )
            # A router item never runs via the job engine (the dashboard's
            # queue/sweep routes block that combination, same rule as the
            # CLI's --via-job-engine + --run-type router guard) --
            # run_sweep_via_job() only supports model runs anyway.
            if via_job_engine and cfg.run_type != "router":
                result, raw_trials = run_sweep_via_job(client, environment, cfg)
            else:
                result, raw_trials = run_sweep(client, environment, cfg)
            validate_result(result)
            (out_dir / f"{result['run_id']}.json").write_text(json.dumps(result, indent=2))
            save_trials(results_dir, batch.profile_name, result["run_id"], raw_trials)
            item["status"] = "completed"
            item["run_id"] = result["run_id"]
        except Exception as exc:
            # One combination failing (e.g. an unresolvable backend, a model
            # that won't load with this engine) must not abort the rest of
            # the queue -- each row is independent, same as a single run.
            item["status"] = "failed"
            item["error"] = str(exc)
            save_failure(
                results_dir,
                batch.profile_name,
                {
                    "model_name": item["model_name"],
                    "quantization": item["quantization"],
                    "compute_engine": item["compute_engine"],
                    "backend": item["backend"],
                    "power_state": item["power_state"],
                },
                "sweep_item",
                str(exc),
                batch_id=batch.id,
            )
        finally:
            if store is not None:
                store.update_item(batch.id, idx, item)

    batch.status = "done"
    if store is not None:
        store.finish_batch(batch.id)


def start_batch(batch: SweepBatch, **run_batch_kwargs) -> threading.Thread:
    """Starts `batch` running in a background thread and returns it. The
    caller is expected to have already stored `batch` somewhere pollable
    (e.g. app.config) before calling this, since the thread mutates it
    directly rather than returning anything."""
    thread = threading.Thread(target=_run_batch, args=(batch,), kwargs=run_batch_kwargs, daemon=True)
    thread.start()
    return thread
