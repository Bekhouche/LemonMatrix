"""Import results from other Lemonade benchmark tools.

Three formats are read:

- lemonade_lab (skywalk1411/lemonade_lab, published to amdaibenchmarks): one
  JSON per run with ``system``, ``results[workload][model][backend]`` and
  ``settings``. It is the only public source of Ryzen AI NPU and hybrid rows.
- LemonMetrics (lemonmetrics/lemonmetrics): a ``report.json`` that wraps
  ``lemonade bench --json`` output, with optional power samples and per-scenario
  energy estimates.
- Lemonade's own bench site (bench.lemonade-server.ai): a saved copy of its
  ``/api/benchmarks`` list, one row per model, backend and scenario. One file
  can hold several machines; each machine becomes its own document.

All are normalized into ``imported_result.schema.json``. Only measurements are
kept: host names, command lines, server URLs and raw power samples are dropped.
LemonMatrix's validity checks never ran on these numbers, so every document is
marked ``validated: false``; the one check that can run on the numbers alone
(identical results across backends) is reported as a warning.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone

LEMONADE_LAB = "lemonade_lab"
LEMONMETRICS = "lemonmetrics"
LEMONADE_BENCH_SITE = "lemonade_bench_site"
_BENCH_SITE_KEYS = {"benchmark_id", "model", "backend", "scenario", "tps_mean"}

_LAB_WORKLOADS = {"llm": "llm", "embedding": "embedding", "image_gen": "image_gen"}
_GPU_BACKENDS = {"vulkan", "rocm", "cuda", "metal", "opencl"}
# Results this close across different backends usually mean the backend
# switch never reached the runtime.
_SAME_VALUE_TOLERANCE = 0.02


def _engine_for(backend: str, recipe: str | None = None) -> str | None:
    if recipe == "flm" or backend == "npu":
        return "npu"
    if backend == "hybrid":
        return "hybrid"
    if backend == "cpu":
        return "cpu"
    if backend in _GPU_BACKENDS:
        return "gpu"
    return None


def _memory_gb(text) -> float | None:
    if isinstance(text, (int, float)):
        return float(text)
    match = re.match(r"\s*([\d.]+)\s*GB", str(text or ""), re.IGNORECASE)
    return float(match.group(1)) if match else None


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def detect_source(data) -> str | None:
    if isinstance(data, list):
        if data and all(isinstance(r, dict) and _BENCH_SITE_KEYS <= r.keys() for r in data):
            return LEMONADE_BENCH_SITE
        return None
    if isinstance(data.get("results"), dict) and isinstance(data.get("system"), dict):
        return LEMONADE_LAB
    if "lemonmetrics_version" in data and isinstance(data.get("lemonade"), dict):
        return LEMONMETRICS
    return None


def _same_value_warnings(rows: list[dict]) -> list[str]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["workload"], row["model"], row.get("scenario")), []).append(row)
    warnings = []
    for (workload, model, scenario), group in groups.items():
        backends = {r["backend"] for r in group}
        values = [r["value"] for r in group if r["value"] > 0]
        if len(backends) < 2 or len(values) < 2:
            continue
        if (max(values) - min(values)) / max(values) <= _SAME_VALUE_TOLERANCE:
            label = f"{model} ({scenario})" if scenario else model
            warnings.append(
                f"{workload} {label}: {', '.join(sorted(backends))} are within "
                f"{_SAME_VALUE_TOLERANCE:.0%} of each other; the backend switch may not have taken effect"
            )
    return warnings


def _lemonade_lab(data: dict) -> dict:
    system = data.get("system") or {}
    settings = data.get("settings") or {}
    rows = []
    for workload_key, models in (data.get("results") or {}).items():
        workload = _LAB_WORKLOADS.get(workload_key)
        if workload is None:
            continue
        for model, backends in (models or {}).items():
            for backend, m in (backends or {}).items():
                if not isinstance(m, dict) or m.get("value") is None:
                    continue
                rows.append(_drop_none({
                    "workload": workload,
                    "model": model,
                    "backend": backend,
                    "compute_engine": _engine_for(backend),
                    "value": float(m["value"]),
                    "unit": m.get("unit") or "",
                    "ttft_ms": m.get("ttft_ms"),
                    "memory_gb": m.get("memory_gb"),
                }))
    return {
        "measured_at": data.get("timestamp"),
        "submitted_by": data.get("submitted_by"),
        "tool_version": None,
        "environment": _drop_none({
            "cpu": system.get("cpu"),
            "gpu": system.get("gpu"),
            "npu": system.get("npu"),
            "os": system.get("os"),
            "memory_gb": _memory_gb(system.get("memory")),
        }),
        "settings": _drop_none({"runs": settings.get("runs"), "warmup": settings.get("warmup")}),
        "power": None,
        "rows": rows,
        "notes": (
            "Imported from lemonade_lab: one mean per backend from lemonade bench, no per-trial data. "
            "memory_gb is lemonade bench's peak memory figure. Device, exclusivity and quantization were "
            "not checked by LemonMatrix."
        ),
    }


def _lemonmetrics(data: dict) -> dict:
    device = data.get("device") or {}
    lemonade = data.get("lemonade") or {}
    bench = lemonade.get("bench_output") or {}
    efficiency = data.get("efficiency") or {}
    energy_by_scenario = {
        (s.get("name"), s.get("recipe"), s.get("backend")): s
        for s in (efficiency.get("by_scenario") or [])
        if efficiency.get("available")
    }

    rows = []
    for model_entry in bench.get("models") or []:
        model = model_entry.get("model")
        for result in model_entry.get("results") or []:
            recipe, backend_key = result.get("recipe"), result.get("backend")
            backend = f"{recipe}-{backend_key}" if recipe else backend_key
            for scenario in result.get("scenarios") or []:
                tps = (scenario.get("tps") or {}).get("mean")
                if tps is None or not model or not backend:
                    continue
                energy = energy_by_scenario.get((scenario.get("name"), recipe, backend_key)) or {}
                rows.append(_drop_none({
                    "workload": "llm",
                    "model": model,
                    "backend": backend,
                    "compute_engine": _engine_for(backend_key, recipe),
                    "scenario": scenario.get("name"),
                    "value": float(tps),
                    "unit": "tok/s",
                    "ttft_ms": (scenario.get("ttft_ms") or {}).get("mean"),
                    "memory_gb": scenario.get("memory_peak_gb"),
                    "joules_per_token": energy.get("estimated_joules_per_token"),
                    "joules_per_output_token": energy.get("estimated_joules_per_output_token"),
                }))

    cpu = device.get("cpu") or {}
    version = re.search(r"(\d+\.\d+(?:\.\d+)?)", str(lemonade.get("cli_version") or ""))
    power = data.get("power") or {}
    return {
        "measured_at": data.get("generated_at"),
        "submitted_by": None,
        "tool_version": data.get("lemonmetrics_version"),
        "environment": _drop_none({
            "cpu": cpu.get("model") if isinstance(cpu, dict) else cpu,
            "gpu": ", ".join(device.get("gpu") or []) or None,
            "npu": ", ".join(device.get("npu") or []) or None,
            "os": device.get("os"),
            "memory_gb": device.get("memory_gb"),
            "lemonade_version": version.group(1) if version else None,
        }),
        "settings": None,
        "power": _drop_none({
            "available": bool(power.get("available")),
            "avg_watts": efficiency.get("avg_watts") if efficiency.get("available") else None,
            "baseline_watts": efficiency.get("baseline_watts") if efficiency.get("available") else None,
        }),
        "rows": rows,
        "notes": (
            "Imported from LemonMetrics: lemonade bench scenarios, mean tok/s and TTFT. Energy figures are "
            "LemonMetrics' own estimates from whole-run power, which includes model load. Device, exclusivity "
            "and quantization were not checked by LemonMatrix."
        ),
    }


def _bench_site_machines(rows: list[dict]) -> list[dict]:
    """One parsed document per machine in a bench.lemonade-server.ai row list."""
    machines: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row.get("gpu_name") or "", row.get("cpu") or "", row.get("os") or "")
        machines.setdefault(key, []).append(row)
    parsed = []
    for (gpu, cpu, os_name), group in sorted(machines.items()):
        out_rows = []
        for r in group:
            if r.get("tps_mean") is None:
                continue
            backend = str(r["backend"])
            out_rows.append(_drop_none({
                "workload": "llm",
                "model": r["model"],
                "backend": f"{backend} ({r['fork']})" if r.get("fork") else backend,
                "compute_engine": _engine_for(backend),
                "scenario": r.get("scenario"),
                "value": float(r["tps_mean"]),
                "unit": "tok/s",
                "ttft_ms": r.get("ttft_ms_max"),
            }))
        parsed.append({
            "key": f"{gpu}|{cpu}|{os_name}",
            "measured_at": None,
            "submitted_by": None,
            "tool_version": None,
            "environment": _drop_none({"gpu": gpu or None, "cpu": cpu or None, "os": os_name or None}),
            "settings": None,
            "power": None,
            "rows": out_rows,
            "notes": (
                "Imported from bench.lemonade-server.ai (lemonade bench submissions): mean tok/s per scenario. "
                "ttft_ms is the site's maximum over trials, not a mean. The site records no load arguments, "
                "checkpoint or concurrent activity, so device, exclusivity and quantization were not checked "
                "by LemonMatrix."
            ),
        })
    return parsed


def import_documents(data, raw_bytes: bytes, source: str | None = None) -> list[dict]:
    """Normalize one source file into one or more imported_result documents.

    lemonade_lab and LemonMetrics files hold one machine and give one document;
    a bench.lemonade-server.ai row list gives one document per machine.
    """
    source = source or detect_source(data)
    if source == LEMONADE_BENCH_SITE:
        if not isinstance(data, list):
            raise ValueError("a bench.lemonade-server.ai file is a JSON list of benchmark rows")
        return [_build_doc(parsed, raw_bytes, source) for parsed in _bench_site_machines(data)]
    return [import_result(data, raw_bytes, source)]


def import_result(data: dict, raw_bytes: bytes, source: str | None = None) -> dict:
    """Normalize one single-machine source document into the imported_result schema shape."""
    source = source or detect_source(data)
    if source == LEMONADE_LAB:
        parsed = _lemonade_lab(data)
    elif source == LEMONMETRICS:
        parsed = _lemonmetrics(data)
    elif source == LEMONADE_BENCH_SITE:
        raise ValueError("bench.lemonade-server.ai files can hold several machines; use import_documents")
    else:
        raise ValueError(
            "not a lemonade_lab result, a LemonMetrics report.json or a bench.lemonade-server.ai row list"
        )
    return _build_doc(parsed, raw_bytes, source)


def _build_doc(parsed: dict, raw_bytes: bytes, source: str) -> dict:
    if not parsed["rows"]:
        raise ValueError(f"no measurements found in this {source} file")

    digest = hashlib.sha256(raw_bytes).hexdigest()
    import_id = digest[:16]
    if parsed.get("key"):
        # Several machines from one file: keep each id stable and distinct.
        import_id = hashlib.sha256(f"{digest}|{parsed['key']}".encode()).hexdigest()[:16]
    validity: dict = {"validated": False, "notes": parsed["notes"]}
    warnings = _same_value_warnings(parsed["rows"])
    if warnings:
        validity["warnings"] = warnings

    doc: dict = {
        "schema_version": "0.1.0",
        "run_type": "imported",
        "import_id": import_id,
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "measured_at": parsed["measured_at"],
        "source": _drop_none({
            "tool": source,
            "file_sha256": digest,
            "submitted_by": parsed["submitted_by"],
            "tool_version": parsed["tool_version"],
        }),
        "environment": parsed["environment"],
        "rows": parsed["rows"],
        "validity": validity,
    }
    if parsed["settings"]:
        doc["settings"] = parsed["settings"]
    if parsed["power"]:
        doc["power"] = parsed["power"]
    return doc
