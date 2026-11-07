"""Reads and writes the result/trial JSON files that `lemonmatrix run` produces."""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def list_results(results_dir: Path | str) -> list[dict]:
    """All results under results_dir/<profile>/<run_id>.json, newest first."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    results = []
    for path in results_dir.glob("*/*.json"):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_profile"] = path.parent.name
        data["_run_id"] = path.stem
        results.append(data)

    return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


def load_result(results_dir: Path | str, profile: str, run_id: str) -> dict | None:
    path = Path(results_dir) / profile / f"{run_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    data["_profile"] = profile
    data["_run_id"] = run_id
    return data


def save_failure(results_dir: Path | str, profile: str, attempted: dict, stage: str, error: str, batch_id: str | None = None) -> Path:
    """Persist a run that raised before producing a schema-conformant result
    (e.g. a model that crashes on load) as its own record.

    Kept under <profile>/failures/ rather than alongside result JSON files,
    both so it never has to satisfy result.schema.json (a crash has no
    metrics to report) and so list_results()'s one-level-deep glob
    (<profile>/<run_id>.json) never has to distinguish the two -- a file
    nested one level further down simply doesn't match that pattern.
    """
    out_dir = Path(results_dir) / profile / "failures"
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "id": uuid.uuid4().hex[:12],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "error": error,
        "attempted": attempted,
    }
    if batch_id:
        record["batch_id"] = batch_id
    path = out_dir / f"{record['id']}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def list_failures(results_dir: Path | str, profile: str | None = None) -> list[dict]:
    """All persisted failure records, newest first. Scoped to one profile if
    given, else across every profile's failures/ directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    pattern = f"{profile}/failures/*.json" if profile else "*/failures/*.json"
    failures = []
    for path in results_dir.glob(pattern):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_profile"] = path.parent.parent.name
        failures.append(data)

    return sorted(failures, key=lambda f: f.get("timestamp", ""), reverse=True)


def get_path(result: dict, dotted: str) -> Any:
    """Reads a dotted path (e.g. "metrics.decode.tokens_per_sec") out of a result dict."""
    value: Any = result
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


# ---------------------------------------------------------------------------
# Trial sidecars (raw per-trial measurements)
# ---------------------------------------------------------------------------

def save_trials(results_dir: Path | str, profile: str, run_id: str, trials: list[dict]) -> Path:
    """Persist raw per-trial measurements as <results_dir>/<profile>/trials/<run_id>.json.

    These files live one directory level deeper than result JSONs so that
    list_results()'s glob (`*/*.json`) never picks them up.
    """
    out_dir = Path(results_dir) / profile / "trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{run_id}.json"
    path.write_text(json.dumps({"run_id": run_id, "trials": trials}, indent=2))
    return path


def load_trials(results_dir: Path | str, profile: str, run_id: str) -> list[dict] | None:
    """Return raw trials for a run, or None if no sidecar was saved."""
    path = Path(results_dir) / profile / "trials" / f"{run_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return data.get("trials")


# ---------------------------------------------------------------------------
# Classification results (separate from model/router results -- see
# classify_result.schema.json's docstring for why they're never mixed into
# the same leaderboard).
# ---------------------------------------------------------------------------

def save_classify_result(results_dir: Path | str, profile: str, result: dict) -> Path:
    """Persist a classify run as <results_dir>/<profile>/classify/<run_id>.json.

    Kept in a subdirectory (like failures/ and trials/) so list_results()'s
    one-level-deep glob (<profile>/<run_id>.json) never picks it up -- it
    doesn't conform to result.schema.json and must never appear on the
    model/router leaderboard.
    """
    out_dir = Path(results_dir) / profile / "classify"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def list_classify_results(results_dir: Path | str, profile: str | None = None) -> list[dict]:
    """All persisted classify results, newest first. Scoped to one profile if
    given, else across every profile's classify/ directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    pattern = f"{profile}/classify/*.json" if profile else "*/classify/*.json"
    results = []
    for path in results_dir.glob(pattern):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_profile"] = path.parent.parent.name
        results.append(data)

    return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


# ---------------------------------------------------------------------------
# TTS results (separate from model/router results -- see
# tts_result.schema.json's docstring for why they're never mixed into the
# same leaderboard).
# ---------------------------------------------------------------------------

def save_tts_result(results_dir: Path | str, profile: str, result: dict) -> Path:
    """Persist a TTS run as <results_dir>/<profile>/tts/<run_id>.json.

    Kept in a subdirectory (like failures/, trials/, and classify/) so
    list_results()'s one-level-deep glob (<profile>/<run_id>.json) never
    picks it up -- it doesn't conform to result.schema.json and must never
    appear on the model/router leaderboard.
    """
    out_dir = Path(results_dir) / profile / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def list_tts_results(results_dir: Path | str, profile: str | None = None) -> list[dict]:
    """All persisted TTS results, newest first. Scoped to one profile if
    given, else across every profile's tts/ directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    pattern = f"{profile}/tts/*.json" if profile else "*/tts/*.json"
    results = []
    for path in results_dir.glob(pattern):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_profile"] = path.parent.parent.name
        results.append(data)

    return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


# ---------------------------------------------------------------------------
# STT results (separate from model/router results -- see
# stt_result.schema.json's docstring for why they're never mixed into the
# same leaderboard).
# ---------------------------------------------------------------------------

def save_stt_result(results_dir: Path | str, profile: str, result: dict) -> Path:
    """Persist an STT run as <results_dir>/<profile>/stt/<run_id>.json.

    Kept in a subdirectory (like failures/, trials/, classify/, and tts/) so
    list_results()'s one-level-deep glob (<profile>/<run_id>.json) never
    picks it up -- it doesn't conform to result.schema.json and must never
    appear on the model/router leaderboard.
    """
    out_dir = Path(results_dir) / profile / "stt"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def list_stt_results(results_dir: Path | str, profile: str | None = None) -> list[dict]:
    """All persisted STT results, newest first. Scoped to one profile if
    given, else across every profile's stt/ directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    pattern = f"{profile}/stt/*.json" if profile else "*/stt/*.json"
    results = []
    for path in results_dir.glob(pattern):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_profile"] = path.parent.parent.name
        results.append(data)

    return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


# ---------------------------------------------------------------------------
# Image-generation results (separate from model/router results -- see
# imagegen_result.schema.json's docstring for why they're never mixed into
# the same leaderboard).
# ---------------------------------------------------------------------------

def save_imagegen_result(results_dir: Path | str, profile: str, result: dict) -> Path:
    """Persist an image-generation run as
    <results_dir>/<profile>/imagegen/<run_id>.json.

    Kept in a subdirectory (like failures/, trials/, classify/, tts/, and
    stt/) so list_results()'s one-level-deep glob (<profile>/<run_id>.json)
    never picks it up -- it doesn't conform to result.schema.json and must
    never appear on the model/router leaderboard.
    """
    out_dir = Path(results_dir) / profile / "imagegen"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def list_imagegen_results(results_dir: Path | str, profile: str | None = None) -> list[dict]:
    """All persisted image-generation results, newest first. Scoped to one
    profile if given, else across every profile's imagegen/ directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    pattern = f"{profile}/imagegen/*.json" if profile else "*/imagegen/*.json"
    results = []
    for path in results_dir.glob(pattern):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_profile"] = path.parent.parent.name
        results.append(data)

    return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


# ---------------------------------------------------------------------------
# Audio-generation results (separate from model/router results -- see
# audiogen_result.schema.json's docstring for why they're never mixed into
# the same leaderboard, or even into tts_result's).
# ---------------------------------------------------------------------------

def save_audiogen_result(results_dir: Path | str, profile: str, result: dict) -> Path:
    """Persist an audio-generation run as
    <results_dir>/<profile>/audiogen/<run_id>.json.

    Kept in a subdirectory (like failures/, trials/, classify/, tts/, stt/,
    and imagegen/) so list_results()'s one-level-deep glob
    (<profile>/<run_id>.json) never picks it up -- it doesn't conform to
    result.schema.json and must never appear on the model/router leaderboard.
    """
    out_dir = Path(results_dir) / profile / "audiogen"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def list_audiogen_results(results_dir: Path | str, profile: str | None = None) -> list[dict]:
    """All persisted audio-generation results, newest first. Scoped to one
    profile if given, else across every profile's audiogen/ directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    pattern = f"{profile}/audiogen/*.json" if profile else "*/audiogen/*.json"
    results = []
    for path in results_dir.glob(pattern):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_profile"] = path.parent.parent.name
        results.append(data)

    return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


# ---------------------------------------------------------------------------
# Embeddings results (separate from model/router results -- see
# embeddings_result.schema.json's docstring for why they're never mixed
# into the same leaderboard).
# ---------------------------------------------------------------------------

def save_embeddings_result(results_dir: Path | str, profile: str, result: dict) -> Path:
    """Persist an embeddings run as <results_dir>/<profile>/embeddings/<run_id>.json.

    Kept in a subdirectory (like failures/, trials/, classify/, tts/, stt/,
    imagegen/, and audiogen/) so list_results()'s one-level-deep glob
    (<profile>/<run_id>.json) never picks it up -- it doesn't conform to
    result.schema.json and must never appear on the model/router leaderboard.
    """
    out_dir = Path(results_dir) / profile / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def list_embeddings_results(results_dir: Path | str, profile: str | None = None) -> list[dict]:
    """All persisted embeddings results, newest first. Scoped to one profile
    if given, else across every profile's embeddings/ directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    pattern = f"{profile}/embeddings/*.json" if profile else "*/embeddings/*.json"
    results = []
    for path in results_dir.glob(pattern):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_profile"] = path.parent.parent.name
        results.append(data)

    return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


# ---------------------------------------------------------------------------
# Reranking results (separate from model/router results -- see
# rerank_result.schema.json's docstring for why they're never mixed into
# the same leaderboard).
# ---------------------------------------------------------------------------

def save_rerank_result(results_dir: Path | str, profile: str, result: dict) -> Path:
    """Persist a rerank run as <results_dir>/<profile>/rerank/<run_id>.json.

    Kept in a subdirectory (like failures/, trials/, classify/, tts/, stt/,
    imagegen/, audiogen/, and embeddings/) so list_results()'s
    one-level-deep glob (<profile>/<run_id>.json) never picks it up -- it
    doesn't conform to result.schema.json and must never appear on the
    model/router leaderboard.
    """
    out_dir = Path(results_dir) / profile / "rerank"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def list_rerank_results(results_dir: Path | str, profile: str | None = None) -> list[dict]:
    """All persisted rerank results, newest first. Scoped to one profile if
    given, else across every profile's rerank/ directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    pattern = f"{profile}/rerank/*.json" if profile else "*/rerank/*.json"
    results = []
    for path in results_dir.glob(pattern):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_profile"] = path.parent.parent.name
        results.append(data)

    return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


# ---------------------------------------------------------------------------
# 3D mesh generation results (separate from model/router results -- see
# meshgen_result.schema.json's docstring for why they're never mixed into
# the same leaderboard).
# ---------------------------------------------------------------------------

def save_meshgen_result(results_dir: Path | str, profile: str, result: dict) -> Path:
    """Persist a meshgen run as <results_dir>/<profile>/meshgen/<run_id>.json.

    Kept in a subdirectory (like failures/, trials/, classify/, tts/, stt/,
    imagegen/, audiogen/, embeddings/, and rerank/) so list_results()'s
    one-level-deep glob (<profile>/<run_id>.json) never picks it up -- it
    doesn't conform to result.schema.json and must never appear on the
    model/router leaderboard.
    """
    out_dir = Path(results_dir) / profile / "meshgen"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{result['run_id']}.json"
    path.write_text(json.dumps(result, indent=2))
    return path


def list_meshgen_results(results_dir: Path | str, profile: str | None = None) -> list[dict]:
    """All persisted meshgen results, newest first. Scoped to one profile if
    given, else across every profile's meshgen/ directory."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return []

    pattern = f"{profile}/meshgen/*.json" if profile else "*/meshgen/*.json"
    results = []
    for path in results_dir.glob(pattern):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        data["_profile"] = path.parent.parent.name
        results.append(data)

    return sorted(results, key=lambda r: r.get("timestamp", ""), reverse=True)


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

_CSV_HEADERS = [
    "run_id", "timestamp", "run_type", "profile",
    "device_model", "gpu", "os_version",
    "model_name", "model_class", "quantization", "context_length",
    "compute_engine", "backend", "router_default_model", "power_state",
    "decode_tokens_per_sec", "decode_stddev", "decode_p95",
    "prefill_tokens_per_sec", "ttft_ms", "ttft_ms_stddev",
    "trial_count", "peak_memory_gb",
    "valid", "notes",
]


def results_to_csv(results: list[dict]) -> str:
    """Serialise a list of result dicts (as returned by list_results) to CSV."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_HEADERS, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        # gpu: the schema uses igpu/dgpu, never a bare "gpu" field.
        # Prefer dgpu (the performance device), fall back to igpu.
        gpu = get_path(r, "environment.dgpu") or get_path(r, "environment.igpu") or ""
        row = {
            "run_id": r.get("run_id", ""),
            "timestamp": r.get("timestamp", ""),
            "run_type": r.get("run_type", "model"),
            "profile": r.get("_profile", ""),
            "device_model": get_path(r, "environment.device_model"),
            "gpu": gpu,
            "os_version": get_path(r, "environment.os_version"),
            "model_name": get_path(r, "model.name"),
            "model_class": get_path(r, "model.class"),
            "quantization": get_path(r, "model.quantization"),
            "context_length": get_path(r, "model.context_length"),
            "compute_engine": get_path(r, "config.compute_engine"),
            "backend": get_path(r, "config.backend"),
            "router_default_model": get_path(r, "config.router_default_model"),
            "power_state": get_path(r, "config.power_state"),
            "decode_tokens_per_sec": get_path(r, "metrics.decode.tokens_per_sec"),
            "decode_stddev": get_path(r, "metrics.decode.stddev"),
            "decode_p95": get_path(r, "metrics.decode.p95"),
            "prefill_tokens_per_sec": get_path(r, "metrics.prefill.tokens_per_sec"),
            "ttft_ms": get_path(r, "metrics.ttft_ms"),
            "ttft_ms_stddev": get_path(r, "metrics.ttft_ms_stddev"),
            "trial_count": get_path(r, "metrics.trial_count"),
            "peak_memory_gb": get_path(r, "metrics.peak_memory_gb"),
            "valid": get_path(r, "validity.valid"),
            "notes": get_path(r, "validity.notes"),
        }
        writer.writerow(row)
    return buf.getvalue()
