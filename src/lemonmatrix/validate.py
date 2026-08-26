"""Validates a benchmark result against the canonical schema."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import jsonschema

_SCHEMA_CACHE: dict[str, dict] = {}


def _find_schema_path(filename: str) -> Path:
    # Repo checkout layout: src/lemonmatrix/validate.py -> ../../schema/<filename>
    repo_schema = Path(__file__).resolve().parents[2] / "schema" / filename
    if repo_schema.exists():
        return repo_schema
    # Installed-package layout: schema/ shipped alongside the package data.
    packaged = resources.files("lemonmatrix").joinpath(f"schema/{filename}")
    return Path(str(packaged))


def _load_schema(filename: str) -> dict:
    if filename not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[filename] = json.loads(_find_schema_path(filename).read_text())
    return _SCHEMA_CACHE[filename]


def load_schema() -> dict:
    return _load_schema("result.schema.json")


def load_classify_schema() -> dict:
    return _load_schema("classify_result.schema.json")


def load_tts_schema() -> dict:
    return _load_schema("tts_result.schema.json")


def load_stt_schema() -> dict:
    return _load_schema("stt_result.schema.json")


def load_imagegen_schema() -> dict:
    return _load_schema("imagegen_result.schema.json")


def load_audiogen_schema() -> dict:
    return _load_schema("audiogen_result.schema.json")


def load_embeddings_schema() -> dict:
    return _load_schema("embeddings_result.schema.json")


def load_rerank_schema() -> dict:
    return _load_schema("rerank_result.schema.json")


def load_meshgen_schema() -> dict:
    return _load_schema("meshgen_result.schema.json")


def validate_result(result: dict) -> None:
    """Raises jsonschema.ValidationError if `result` doesn't conform."""
    jsonschema.validate(instance=result, schema=load_schema())


def validate_classify_result(result: dict) -> None:
    """Raises jsonschema.ValidationError if `result` doesn't conform.

    Deliberately a separate schema from validate_result(): classification
    latency/throughput is not comparable to LLM token throughput, so classify
    runs are never mixed into the model/router leaderboard.
    """
    jsonschema.validate(instance=result, schema=load_classify_schema())


def validate_tts_result(result: dict) -> None:
    """Raises jsonschema.ValidationError if `result` doesn't conform.

    Deliberately a separate schema from validate_result(): real-time-factor
    is not comparable to LLM token throughput, so TTS runs are never mixed
    into the model/router leaderboard.
    """
    jsonschema.validate(instance=result, schema=load_tts_schema())


def validate_stt_result(result: dict) -> None:
    """Raises jsonschema.ValidationError if `result` doesn't conform.

    Deliberately a separate schema from validate_result(): real-time-factor
    is not comparable to LLM token throughput, so STT runs are never mixed
    into the model/router leaderboard.
    """
    jsonschema.validate(instance=result, schema=load_stt_schema())


def validate_imagegen_result(result: dict) -> None:
    """Raises jsonschema.ValidationError if `result` doesn't conform.

    Deliberately a separate schema from validate_result(): images/sec is not
    comparable to LLM token throughput, so image-generation runs are never
    mixed into the model/router leaderboard.
    """
    jsonschema.validate(instance=result, schema=load_imagegen_schema())


def validate_audiogen_result(result: dict) -> None:
    """Raises jsonschema.ValidationError if `result` doesn't conform.

    Deliberately a separate schema from validate_tts_result(): generating
    music/sound effects (acestep/thinksound) is a different task with a
    different compute profile than text-to-speech, so the two are never
    mixed on the same leaderboard.
    """
    jsonschema.validate(instance=result, schema=load_audiogen_schema())


def validate_embeddings_result(result: dict) -> None:
    """Raises jsonschema.ValidationError if `result` doesn't conform.

    Deliberately a separate schema from validate_result(): embeddings
    throughput is not comparable to LLM token throughput, so embeddings runs
    are never mixed into the model/router leaderboard.
    """
    jsonschema.validate(instance=result, schema=load_embeddings_schema())


def validate_rerank_result(result: dict) -> None:
    """Raises jsonschema.ValidationError if `result` doesn't conform.

    Deliberately a separate schema from validate_result(): reranking
    throughput is not comparable to LLM token throughput, so rerank runs are
    never mixed into the model/router leaderboard.
    """
    jsonschema.validate(instance=result, schema=load_rerank_schema())


def validate_meshgen_result(result: dict) -> None:
    """Raises jsonschema.ValidationError if `result` doesn't conform.

    Deliberately a separate schema from validate_result(): mesh-generation
    throughput is not comparable to LLM token throughput, so meshgen runs
    are never mixed into the model/router leaderboard.
    """
    jsonschema.validate(instance=result, schema=load_meshgen_schema())
