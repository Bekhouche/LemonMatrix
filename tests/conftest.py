"""A fake Lemonade server for tests, so we don't need real AMD hardware to
exercise discovery, benchmarking, and the CLI end to end."""

from __future__ import annotations

import io
import json
import re
import threading
import wave
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

# Shapes modeled on the documented examples at
# https://lemonade-server.ai/docs/api/lemonade/ -- see profile.py's docstring
# for why discovery treats these field names as best-effort, not guaranteed.
FAKE_SYSTEM_INFO = {
    "OS Version": "Windows 11 Pro 24H2",
    "OEM System": "HP Ryzen AI Max+ 395 (Strix Halo)",
    "Physical Memory": "128 GB",
    "cpu": {"name": "AMD Ryzen AI 9 HX 370", "cores": 12, "threads": 24, "available": True},
    "amd_gpu": [{"name": "AMD Radeon 8060S", "vram_gb": 0.5, "available": True, "family": "gfx1150"}],
    "amd_npu": {"name": "XDNA 2", "available": True, "family": "XDNA2"},
    # A discrete NVIDIA card alongside the Strix Halo APU -- acestep/thinksound
    # and trellis only ever declare a "cuda" backend (confirmed live: neither
    # recipe has a cpu fallback), so a "dgpu" engine must actually be present
    # on this fake profile for any test to exercise them through a combo that
    # validate_combo_against_profile() -- and the dashboard's own pre-flight
    # check -- would accept as real, rather than one that only "worked" by
    # accident because nothing checked hardware presence before running.
    "nvidia_gpu": [{"name": "NVIDIA RTX 4090", "vram_gb": 24, "available": True}],
    "recipes": {
        "llamacpp": {
            "display_name": "Llama.cpp GPU",
            "modality": "Text generation",
            "backends": {
                "vulkan": {"state": "installed", "version": "b10375"},
                "cpu": {"state": "installed", "version": "b10375"},
                "cuda": {"state": "installable", "version": "b10397"},
            },
        },
        "onnxruntime": {
            "display_name": "ONNX Runtime",
            "modality": "Text classification",
            "selectable_backend": False,
            "backends": {
                "cpu": {"state": "installed", "version": "0.3.7"},
            },
        },
        "kokoro": {
            "display_name": "Kokoro",
            "modality": "Text-to-speech",
            "selectable_backend": False,
            "backends": {
                "cpu": {"state": "installed", "version": "b17"},
            },
        },
        "whispercpp": {
            "display_name": "Whisper.cpp",
            "modality": "Speech-to-text",
            "selectable_backend": True,
            "backends": {
                "cpu": {"state": "installed", "version": "1.8.4"},
                "vulkan": {"state": "installed", "version": "1.8.4"},
            },
        },
        "sd-cpp": {
            "display_name": "StableDiffusion.cpp",
            "modality": "Image generation",
            "selectable_backend": True,
            "backends": {
                "cpu": {"state": "installed", "version": "0.2.0"},
                "cuda": {"state": "installed", "version": "0.2.0"},
            },
        },
        "acestep": {
            "display_name": "ACE-Step",
            "modality": "Audio generation",
            "selectable_backend": True,
            "backends": {
                "cuda": {"state": "installed", "version": "1.0.0"},
                "vulkan": {"state": "installed", "version": "1.0.0"},
            },
        },
        "trellis": {
            "display_name": "TRELLIS",
            "modality": "3D generation",
            "selectable_backend": True,
            "backends": {
                "cuda": {"state": "installed", "version": "2.0.0"},
            },
        },
    },
}

FAKE_HEALTH = {"status": "ok", "version": "8.1.0", "model_loaded": None, "all_models_loaded": []}

FAKE_STATS = {
    "time_to_first_token": 0.18,
    "tokens_per_second": 42.0,
    "input_tokens": 40,
    "output_tokens": 256,
    "prompt_tokens": 40,
}

# Router model registered in the fake server's models list.
FAKE_ROUTER_MODEL_ID = "my-collection-router"

# ONNX classifier model registered in the fake server's models list.
FAKE_CLASSIFY_MODEL_ID = "Phishing-Email-Detection-ONNX"
FAKE_CLASSIFY_CHECKPOINT = "cybersectony/phishing-email-detection-distilbert_v2.4.1"

# Response shape confirmed against Lemonade's own server source
# (handle_classify in src/cpp/server/server.cpp): "labels" is an object
# keyed by label name, not a list, and there is no timing field.
FAKE_CLASSIFY_LABELS = {"LABEL_0": 0.982, "LABEL_1": 0.011, "LABEL_2": 0.005}

# TTS model registered in the fake server's models list.
FAKE_TTS_MODEL_ID = "kokoro-v1"
FAKE_TTS_CHECKPOINT = "mikkoph/kokoro-onnx"
FAKE_TTS_WAV_SECONDS = 1.5


def _make_wav_bytes(duration_s: float, sample_rate: int = 24000) -> bytes:
    """A real, minimal, silent WAV clip -- confirmed live that Lemonade's
    kokoro backend emits a standard RIFF/WAVE container for
    response_format="wav", so the fake server should too."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * int(duration_s * sample_rate))
    return buf.getvalue()


FAKE_TTS_WAV_BYTES = _make_wav_bytes(FAKE_TTS_WAV_SECONDS)

# STT model registered in the fake server's models list.
FAKE_STT_MODEL_ID = "Whisper-Tiny"
FAKE_STT_CHECKPOINT = "ggml-org/whisper-tiny"
FAKE_STT_TRANSCRIPT = "Lemonade can speak"

# Image-generation model registered in the fake server's models list.
FAKE_IMAGEGEN_MODEL_ID = "SD-Turbo"
FAKE_IMAGEGEN_CHECKPOINT = "stabilityai/sd-turbo"
# A minimal valid base64 payload -- benchmarking never decodes it as a real
# image, just confirms the field is present, so content doesn't matter.
FAKE_IMAGEGEN_B64 = "aGVsbG8="

# Audio-generation model registered in the fake server's models list.
FAKE_AUDIOGEN_MODEL_ID = "ACE-Step-v1"
FAKE_AUDIOGEN_CHECKPOINT = "ACE-Step/ACE-Step-v1-3.5B"
FAKE_AUDIOGEN_WAV_SECONDS = 4.0
FAKE_AUDIOGEN_WAV_BYTES = _make_wav_bytes(FAKE_AUDIOGEN_WAV_SECONDS)

# Embeddings/reranking model registered in the fake server's models list.
# Both recipes are llamacpp -- confirmed live these are ordinary GGUF models,
# not a separate recipe, so they share llamacpp's existing backend entries.
FAKE_EMBED_MODEL_ID = "BGE-Small-EN-GGUF"
FAKE_EMBED_CHECKPOINT = "CompendiumLabs/bge-small-en-v1.5-gguf"
FAKE_EMBEDDING_DIM = 384
FAKE_RERANK_MODEL_ID = "BGE-Reranker-Base-GGUF"
FAKE_RERANK_CHECKPOINT = "gpustack/bge-reranker-base-GGUF"

# 3D mesh-generation model registered in the fake server's models list.
FAKE_MESHGEN_MODEL_ID = "TRELLIS-image-large"
FAKE_MESHGEN_CHECKPOINT = "microsoft/TRELLIS-image-large"
# A minimal fake glTF-binary payload -- benchmarking never parses it as a
# real mesh, just confirms the raw bytes come back, so content doesn't matter.
FAKE_MESHGEN_GLB_BYTES = b"glTF" + b"\x00" * 16

# Route trace returned when route_trace=True is sent in a chat/completions request.
FAKE_ROUTE_TRACE = {
    "route_to": "Llama-3.1-8B-Instruct-GGUF",
    "matched_rule": "rule-llm",
    "default_used": False,
    "outputs": [],
    "trace": [],
}

FAKE_SYSTEM_STATS = {"cpu_percent": 12.0, "memory_gb": 24.0, "gpu_percent": 80.0, "vram_gb": 9.5, "npu_percent": None}

FAKE_MODELS = {
    "object": "list",
    "data": [
        {"id": "Llama-3.1-8B-Instruct-GGUF", "object": "model"},
        {
            "id": "Qwen3.8-27B-GGUF-Q4_K_M",
            "object": "model",
            "checkpoint": "unsloth/Qwen3.8-27B-GGUF:Q4_K_M",
            "max_context_window": 131072,
            "recipe": "llamacpp",
            "size": 15.9,
        },
        {
            "id": FAKE_ROUTER_MODEL_ID,
            "object": "model",
            "recipe": "collection.router",
            "components": ["Llama-3.1-8B-Instruct-GGUF"],
        },
        {
            "id": FAKE_CLASSIFY_MODEL_ID,
            "object": "model",
            "checkpoint": FAKE_CLASSIFY_CHECKPOINT,
            "recipe": "onnxruntime",
        },
        {
            "id": FAKE_TTS_MODEL_ID,
            "object": "model",
            "checkpoint": FAKE_TTS_CHECKPOINT,
            "recipe": "kokoro",
        },
        {
            "id": FAKE_STT_MODEL_ID,
            "object": "model",
            "checkpoint": FAKE_STT_CHECKPOINT,
            "recipe": "whispercpp",
        },
        {
            "id": FAKE_IMAGEGEN_MODEL_ID,
            "object": "model",
            "checkpoint": FAKE_IMAGEGEN_CHECKPOINT,
            "recipe": "sd-cpp",
        },
        {
            "id": FAKE_AUDIOGEN_MODEL_ID,
            "object": "model",
            "checkpoint": FAKE_AUDIOGEN_CHECKPOINT,
            "recipe": "acestep",
        },
        {
            "id": FAKE_EMBED_MODEL_ID,
            "object": "model",
            "checkpoint": FAKE_EMBED_CHECKPOINT,
            "recipe": "llamacpp",
        },
        {
            "id": FAKE_RERANK_MODEL_ID,
            "object": "model",
            "checkpoint": FAKE_RERANK_CHECKPOINT,
            "recipe": "llamacpp",
        },
        {
            "id": FAKE_MESHGEN_MODEL_ID,
            "object": "model",
            "checkpoint": FAKE_MESHGEN_CHECKPOINT,
            "recipe": "trellis",
        },
    ],
}

# Shapes confirmed live against a real Lemonade 11.6.0 instance (see the
# GET /v1/registry/search and GET /v1/pull/variants calls in this session).
FAKE_SEARCH_RESULTS = {
    "source": "huggingface",
    "query": "llama-3.2-1b",
    "total": 1,
    "results": [
        {
            "repository_id": "bartowski/Llama-3.2-1B-Instruct-GGUF",
            "display_name": "Llama-3.2-1B-Instruct-GGUF",
            "source": "huggingface",
            "repository_type": "model",
            "description": "",
            "tags": ["gguf", "text-generation"],
            "task": "text-generation",
            "downloads": 147493,
            "likes": 173,
            "has_gguf": True,
        }
    ],
}

FAKE_PULL_VARIANTS = {
    "checkpoint": "bartowski/Llama-3.2-1B-Instruct-GGUF",
    "recipe": "llamacpp",
    "source": "huggingface",
    "repo_kind": "gguf",
    "suggested_name": "Llama-3.2-1B-Instruct-GGUF",
    "suggested_labels": ["chat"],
    "mmproj_files": [],
    "draft_files": [],
    "variants": [
        {"name": "Q4_K_M", "primary_file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf", "files": ["Llama-3.2-1B-Instruct-Q4_K_M.gguf"], "sharded": False, "size_bytes": 807694464},
        {"name": "Q8_0", "primary_file": "Llama-3.2-1B-Instruct-Q8_0.gguf", "files": ["Llama-3.2-1B-Instruct-Q8_0.gguf"], "sharded": False, "size_bytes": 1321083008},
    ],
}


def _parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], set[str]]:
    """Parses a multipart/form-data body into (text field values, names of
    file fields present) -- shared by every endpoint that (per Lemonade's
    own server source) accepts multipart requests: /audio/transcriptions,
    /images/edits, /images/variations."""
    msg = BytesParser().parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
    )
    fields: dict[str, str] = {}
    file_fields: set[str] = set()
    for part in msg.get_payload() or []:
        match = re.search(r'name="([^"]+)"', part.get("Content-Disposition", ""))
        if not match:
            continue
        field_name = match.group(1)
        if part.get("Content-Disposition", "").find("filename=") != -1:
            file_fields.add(field_name)
        else:
            fields[field_name] = (part.get_payload(decode=True) or b"").decode(errors="replace")
    return fields, file_fields


def _fake_execute_job_steps(server, steps: list[dict]) -> tuple[dict, bool, str]:
    """Synchronously "executes" a job's steps against the same fake data the
    direct-HTTP endpoints use, for POST /v1/jobs -- real Lemonade jobs run
    asynchronously, but the fake server has no need for that complexity, so
    it just computes the final context immediately. Shapes (chat's
    timings/usage, system_stats' fields) are confirmed live against a real
    instance's job engine -- see bench.py's run_sweep_via_job docstring.

    Returns (context, failed, error_message). A "load" step whose model is
    the same "trigger-load-failure" sentinel the direct /api/v1/load fake
    handler recognizes stops execution and reports failure, mirroring how a
    real job would fail partway through.
    """
    context: dict = {}
    for step in steps:
        op = step.get("op")
        params = step.get("params") or {}
        if op == "load":
            server.last_load_payload = params
            if params.get("model") == "trigger-load-failure":
                return context, True, f"load requires a 'model' string: {params.get('model')} not found"
            output = {"loaded": True, "model": params.get("model"), "ctx_size": params.get("ctx_size")}
        elif op == "unload":
            output = {}
        elif op == "chat":
            output = {
                "id": "chatcmpl-fake-job",
                "choices": [{"message": {"content": "A keeper finds a note..."}}],
                "timings": {
                    "prompt_ms": FAKE_STATS["time_to_first_token"] * 1000,
                    "predicted_per_second": FAKE_STATS["tokens_per_second"],
                },
                "usage": {
                    "prompt_tokens": FAKE_STATS["prompt_tokens"],
                    "completion_tokens": FAKE_STATS["output_tokens"],
                },
            }
        elif op == "system_stats":
            output = dict(FAKE_SYSTEM_STATS)
        else:
            output = {}
        context[step.get("id")] = output
    return context, False, ""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence test output
        pass

    def _json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _binary(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        routes = {
            "/live": {"status": "ok"},
            "/api/v1/health": FAKE_HEALTH,
            "/api/v1/system-info": FAKE_SYSTEM_INFO,
            "/api/v1/system-stats": FAKE_SYSTEM_STATS,
            "/v1/stats": FAKE_STATS,
            "/api/v1/models": FAKE_MODELS,
            "/v1/registry/search": FAKE_SEARCH_RESULTS,
        }
        job_match = re.match(r"^(?:/api)?/v1/jobs/([^/]+)$", path)
        if path == "/v1/pull/variants":
            checkpoint = query.get("checkpoint", [""])[0]
            if checkpoint == FAKE_PULL_VARIANTS["checkpoint"]:
                self._json(FAKE_PULL_VARIANTS)
            else:
                self._json({"error": "checkpoint not found"}, status=404)
        elif path == "/api/v1/downloads":
            self._json(self.server.jobs)
        elif job_match:
            job = self.server.fake_jobs.get(job_match.group(1))
            if job is None:
                self._json({"error": "unknown job"}, status=404)
            else:
                self._json(job)
        elif path in routes:
            self._json(routes[path])
        else:
            self._json({"error": "not found"}, status=404)

    def do_DELETE(self):
        job_match = re.match(r"^(?:/api)?/v1/jobs/([^/]+)$", self.path)
        if job_match:
            self.server.fake_jobs.pop(job_match.group(1), None)
            self._json({"status": "deleted"})
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.path in ("/api/v1/load", "/api/v1/unload"):
            if self.path == "/api/v1/load":
                payload = json.loads(body or b"{}")
                self.server.last_load_payload = payload
                if payload.get("model_name") == "trigger-load-failure":
                    # Confirmed live: a failing /api/v1/load returns this
                    # nested-object error shape, not the bare-string shape
                    # /api/v1/install and /api/v1/pull use.
                    self._json(
                        {
                            "error": {
                                "code": "model_load_error",
                                "message": "Failed to load model 'trigger-load-failure': llama-server failed to start",
                                "param": "model",
                                "requested_model": "trigger-load-failure",
                                "type": "model_load_error",
                            }
                        },
                        status=500,
                    )
                    return
            self._json({"status": "ok"})
        elif self.path == "/api/v1/chat/completions":
            payload = json.loads(body or b"{}")
            resp: dict = {"id": "cmpl-fake", "choices": [{"message": {"content": "A keeper finds a note..."}}]}
            # If the caller passed route_trace: true, append a fake x_lemonade_route block
            # so that bench.py's router path can capture per-trial routing decisions.
            if payload.get("route_trace"):
                resp["x_lemonade_route"] = FAKE_ROUTE_TRACE
            self._json(resp)
        elif self.path == "/api/v1/install":
            # Both the error shape and the success shape are confirmed live:
            # missing/invalid recipe+backend -> 400 {"error": ...}; a real
            # install of llamacpp:vulkan returned exactly this success shape.
            payload = json.loads(body or b"{}")
            recipe, backend = payload.get("recipe"), payload.get("backend")
            if not recipe or not backend:
                self._json({"error": "Both 'recipe' and 'backend' are required"}, status=400)
                return
            entry = (FAKE_SYSTEM_INFO.get("recipes", {}).get(recipe, {}).get("backends", {}).get(backend))
            if entry is None:
                self._json({"error": f"backend_versions.json is missing version for: {recipe}:{backend}"}, status=400)
                return
            self._json({"status": "success", "recipe": recipe, "backend": backend})
        elif self.path == "/api/v1/pull":
            payload = json.loads(body or b"{}")
            model_name = payload.get("model_name")
            checkpoint = payload.get("checkpoint")

            if payload.get("stream") and not payload.get("subscribe", True):
                # Server-owned background job mode: shape confirmed from
                # Lemonade's docs (this session's research, not exercised
                # live to avoid an unrequested download).
                job = {
                    "id": f"model:{model_name}",
                    "type": "model",
                    "model_name": model_name,
                    "status": "downloading",
                    "running": True,
                    "file": "",
                    "file_index": 0,
                    "total_files": 0,
                    "bytes_downloaded": 0,
                    "bytes_total": 0,
                    "total_download_size": 0,
                    "cumulative_bytes_downloaded": 0,
                    "percent": 0,
                    "complete": False,
                }
                self.server.jobs.append(job)
                self._json(job)
                return

            # Blocking mode (stream=false): success/error shapes confirmed
            # live in this session's docs research.
            if checkpoint == "does-not-exist/repo:Q4_K_M":
                self._json({"status": "error", "message": f"checkpoint not found: {checkpoint}"})
                return
            self._json({"status": "success", "message": f"Installed model: {model_name}"})
        elif self.path in ("/v1/classify", "/api/v1/classify"):
            payload = json.loads(body or b"{}")
            model = payload.get("model")
            text = payload.get("input") or payload.get("text")
            if not text:
                self._json(
                    {"error": {"message": "Missing 'input' (or 'text') string in classify request", "type": "invalid_request_error"}},
                    status=400,
                )
                return
            if not model:
                self._json(
                    {"error": {"message": "No 'model' specified and no single classification model is loaded (load one, or name it in the request)", "type": "invalid_request_error"}},
                    status=400,
                )
                return
            labels = dict(FAKE_CLASSIFY_LABELS)
            top_k = payload.get("top_k")
            if top_k:
                labels = dict(sorted(labels.items(), key=lambda kv: -kv[1])[:top_k])
            self._json({"object": "classification", "model": model, "labels": labels})
        elif self.path in ("/v1/audio/speech", "/api/v1/audio/speech"):
            payload = json.loads(body or b"{}")
            if not payload.get("model"):
                self._json({"error": {"message": "Missing 'model' field in request", "type": "invalid_request_error"}}, status=400)
                return
            if not payload.get("input"):
                self._json({"error": {"message": "Missing 'input' field in request", "type": "invalid_request_error"}}, status=400)
                return
            response_format = payload.get("response_format", "wav")
            if response_format != "wav":
                self._json(
                    {"error": {"message": f"response_format '{response_format}' is not supported by this model (supported: wav)", "type": "invalid_request_error"}},
                    status=400,
                )
                return
            self._binary(FAKE_TTS_WAV_BYTES, "audio/wav")
        elif self.path in ("/v1/audio/transcriptions", "/api/v1/audio/transcriptions"):
            # multipart/form-data, not JSON -- confirmed against Lemonade's
            # own upstream test suite (test/server_whisper.py).
            content_type = self.headers.get("Content-Type", "")
            fields, file_fields = _parse_multipart(content_type, body)
            if "file" not in file_fields:
                self._json({"error": {"message": "Missing 'file' in request", "type": "invalid_request_error"}}, status=400)
                return
            if not fields.get("model"):
                self._json({"error": {"message": "Missing 'model' in request", "type": "invalid_request_error"}}, status=400)
                return
            response_format = fields.get("response_format", "json")
            if response_format != "json":
                self._json(
                    {"error": {"message": f"response_format '{response_format}' not supported by the fake server", "type": "invalid_request_error"}},
                    status=400,
                )
                return
            self._json({"text": FAKE_STT_TRANSCRIPT})
        elif self.path in ("/v1/jobs", "/api/v1/jobs"):
            payload = json.loads(body or b"{}")
            steps = payload.get("steps") or (payload.get("definition") or {}).get("steps") or []
            job_id = f"fake-job-{len(self.server.fake_jobs) + 1}"
            context, failed, error_message = _fake_execute_job_steps(self.server, steps)
            self.server.fake_jobs[job_id] = {
                "id": job_id,
                "name": payload.get("name", ""),
                "status": "failed" if failed else "completed",
                "inputs": payload.get("inputs") or {},
                "context": context,
                "steps": [{**s, "status": "completed", "output": context.get(s.get("id"))} for s in steps],
                "cursor": "",
                "created_at": "",
                "finished_at": "",
                **({"error": error_message} if failed else {}),
            }
            self._json({"id": job_id}, status=202)
        elif self.path in ("/v1/embeddings", "/api/v1/embeddings"):
            payload = json.loads(body or b"{}")
            inputs = payload.get("input")
            inputs = inputs if isinstance(inputs, list) else ([inputs] if inputs else [])
            if not inputs:
                self._json({"error": {"message": "Missing 'input' field in request", "type": "invalid_request_error"}}, status=400)
                return
            self._json(
                {
                    "object": "list",
                    "model": payload.get("model"),
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [0.01 * i] * FAKE_EMBEDDING_DIM}
                        for i in range(len(inputs))
                    ],
                    "usage": {"prompt_tokens": sum(len(str(i)) for i in inputs), "total_tokens": sum(len(str(i)) for i in inputs)},
                }
            )
        elif self.path in ("/v1/rerank", "/api/v1/rerank", "/v1/reranking", "/api/v1/reranking", "/v1/reranker", "/api/v1/reranker"):
            payload = json.loads(body or b"{}")
            documents = payload.get("documents") or []
            if not documents:
                self._json({"error": {"message": "Missing 'documents' field in request", "type": "invalid_request_error"}}, status=400)
                return
            results = sorted(
                (
                    {"index": i, "relevance_score": round(1.0 - i * 0.05, 4)}
                    for i in range(len(documents))
                ),
                key=lambda r: -r["relevance_score"],
            )
            top_n = payload.get("top_n")
            if top_n:
                results = results[:top_n]
            self._json({"results": results})
        elif self.path in ("/v1/audio/generations", "/api/v1/audio/generations"):
            payload = json.loads(body or b"{}")
            if not payload.get("model"):
                self._json({"error": {"message": "Missing 'model' field in request", "type": "invalid_request_error"}}, status=400)
                return
            if not payload.get("prompt"):
                self._json({"error": {"message": "Missing 'prompt' field in request", "type": "invalid_request_error"}}, status=400)
                return
            self._binary(FAKE_AUDIOGEN_WAV_BYTES, "audio/wav")
        elif self.path in ("/v1/images/generations", "/api/v1/images/generations"):
            payload = json.loads(body or b"{}")
            if not payload.get("model"):
                self._json({"error": {"message": "Missing 'model' field in request", "type": "invalid_request_error"}}, status=400)
                return
            if not payload.get("prompt"):
                self._json({"error": {"message": "Missing 'prompt' field in request", "type": "invalid_request_error"}}, status=400)
                return
            self._json({"data": [{"b64_json": FAKE_IMAGEGEN_B64}], "created": 1234567890})
        elif self.path in ("/v1/3d/generations", "/api/v1/3d/generations"):
            payload = json.loads(body or b"{}")
            if not payload.get("model"):
                self._json({"error": {"message": "Missing or non-string 'model' field in request", "type": "invalid_request_error"}}, status=400)
                return
            if not payload.get("image"):
                self._json({"error": {"message": "Missing or non-string 'image' field in request (base64-encoded input image)", "type": "invalid_request_error"}}, status=400)
                return
            self._binary(FAKE_MESHGEN_GLB_BYTES, "model/gltf-binary")
        elif self.path in ("/v1/images/edits", "/api/v1/images/edits"):
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self._json({"error": {"message": "Request must be multipart/form-data", "type": "invalid_request_error"}}, status=400)
                return
            fields, file_fields = _parse_multipart(content_type, body)
            if "image" not in file_fields:
                self._json({"error": {"message": "Missing 'image' field in request", "type": "invalid_request_error"}}, status=400)
                return
            if not fields.get("prompt"):
                self._json({"error": {"message": "Missing 'prompt' field in request", "type": "invalid_request_error"}}, status=400)
                return
            self._json({"data": [{"b64_json": FAKE_IMAGEGEN_B64}], "created": 1234567890})
        elif self.path in ("/v1/images/variations", "/api/v1/images/variations"):
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                self._json({"error": {"message": "Request must be multipart/form-data", "type": "invalid_request_error"}}, status=400)
                return
            fields, file_fields = _parse_multipart(content_type, body)
            if "image" not in file_fields:
                self._json({"error": {"message": "Missing 'image' field in request", "type": "invalid_request_error"}}, status=400)
                return
            self._json({"data": [{"b64_json": FAKE_IMAGEGEN_B64}], "created": 1234567890})
        elif self.path == "/api/v1/downloads/control":
            payload = json.loads(body or b"{}")
            job_id, action = payload.get("id"), payload.get("action")
            job = next((j for j in self.server.jobs if j["id"] == job_id), None)
            if job is None:
                self._json({"status": "ok", "missing": True})
                return
            if action == "pause":
                job["status"] = "paused"
                job["running"] = False
            elif action == "cancel":
                job["status"] = "cancelled"
                job["running"] = False
            elif action == "remove":
                self.server.jobs.remove(job)
                self._json({"status": "ok"})
                return
            self._json(job)
        else:
            self._json({"error": "not found"}, status=404)


_servers_by_port: dict[int, HTTPServer] = {}


def get_fake_server(base_url: str) -> HTTPServer:
    """Look up the running fake server behind a `fake_lemonade` base_url, so
    tests can inspect state (e.g. last_load_payload) the fixture itself
    doesn't expose -- keeps the fixture's plain-string return unchanged for
    the many tests that don't need this."""
    port = int(base_url.rsplit(":", 1)[-1])
    return _servers_by_port[port]


@pytest.fixture
def fake_lemonade():
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    server.jobs = []  # server-owned download jobs, mutated by pull/downloads/control
    server.last_load_payload = None  # captured by /api/v1/load for assertions
    server.fake_jobs = {}  # POST /v1/jobs -- executed synchronously, keyed by job id
    _servers_by_port[server.server_port] = server
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield base_url
    finally:
        del _servers_by_port[server.server_port]
        server.shutdown()
        thread.join()
