"""Thin client for a Lemonade Server instance's REST API.

Endpoint reference: https://lemonade-server.ai/docs/api/lemonade/
"""

from __future__ import annotations

import base64

import requests

DEFAULT_TIMEOUT = 30

# For load()/chat_completion() during an actual benchmark run: slow inference
# is exactly what a benchmarking tool exists to measure, so a request that
# takes minutes must not itself be treated as a client-side failure. Confirmed
# live: a 27B model on CPU with a 262144-token context legitimately took
# longer than DEFAULT_TIMEOUT (30s) just to finish loading (KV-cache
# allocation for that context size is substantial) before any inference ran.
BENCH_TIMEOUT = 600


class LemonadeError(RuntimeError):
    """Raised when a Lemonade endpoint returns an error or unexpected shape."""


class LemonadeAPIError(requests.HTTPError):
    """A non-2xx response, with Lemonade's own error message surfaced where
    possible instead of just "500 Server Error: ... for url: ...".

    Confirmed live, Lemonade uses at least two error shapes: a bare string
    ({"error": "Both 'recipe' and 'backend' are required"}, seen from
    /api/v1/install and /api/v1/pull) and a nested object
    ({"error": {"message": "...", "code": ..., ...}}, seen from a failing
    /api/v1/load). Subclasses HTTPError so existing `except requests.HTTPError`
    and `pytest.raises(requests.HTTPError)` call sites keep working unchanged.
    """


def _raise_for_status_with_message(resp: requests.Response) -> None:
    if resp.ok:
        return
    message = f"{resp.status_code} {resp.reason} for {resp.request.method} {resp.url}"
    try:
        error = resp.json().get("error")
        if isinstance(error, dict):
            message = error.get("message", message)
        elif isinstance(error, str):
            message = error
    except ValueError:
        pass
    raise LemonadeAPIError(message, response=resp)


class LemonadeClient:
    """Talks to a single Lemonade instance's OpenAI-compatible + management API."""

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _get(self, path: str, timeout: float | None = None, **kwargs) -> dict:
        resp = self._session.get(f"{self.base_url}{path}", timeout=timeout or self.timeout, **kwargs)
        _raise_for_status_with_message(resp)
        return resp.json()

    def _post(self, path: str, json: dict, timeout: float | None = None, **kwargs) -> dict:
        resp = self._session.post(f"{self.base_url}{path}", json=json, timeout=timeout or self.timeout, **kwargs)
        _raise_for_status_with_message(resp)
        return resp.json()

    def live(self) -> bool:
        try:
            resp = self._session.get(f"{self.base_url}/live", timeout=self.timeout)
            return resp.ok
        except requests.RequestException:
            return False

    def health(self) -> dict:
        return self._get("/api/v1/health")

    def system_info(self) -> dict:
        return self._get("/api/v1/system-info")

    def system_stats(self) -> dict:
        return self._get("/api/v1/system-stats")

    def stats(self) -> dict:
        """Performance stats for the most recently completed inference request."""
        return self._get("/v1/stats")

    def models(self) -> list[dict]:
        data = self._get("/api/v1/models")
        return data.get("data", data if isinstance(data, list) else [])

    def load(self, model_name: str, **options) -> dict:
        payload = {"model_name": model_name, **options}
        return self._post("/api/v1/load", payload)

    def unload(self, model_name: str | None = None) -> dict:
        payload = {"model_name": model_name} if model_name else {}
        return self._post("/api/v1/unload", payload)

    def chat_completion(self, model: str, messages: list[dict], max_tokens: int, **kwargs) -> dict:
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False, **kwargs}
        return self._post("/api/v1/chat/completions", payload)

    def install_backend(self, recipe: str, backend: str, timeout: float | None = 1800) -> dict:
        """POST /v1/install: downloads that backend's engine binaries. Confirmed
        live end to end: a missing/invalid recipe or backend returns 400 with
        a {"error": ...} body (not a 404), and a real install of llamacpp:vulkan
        returned {"status": "success", "recipe": "llamacpp", "backend": "vulkan"}
        and flipped its system-info state from "installable" to "installed" --
        that specific build was small and finished in under 3 seconds, but
        others (e.g. CUDA) may be much larger.

        This blocks for the whole download, hence the 30-minute default
        timeout, well above the client's normal 30s default. stream=true's
        SSE progress events are not parsed by this client.
        """
        return self._post("/api/v1/install", {"recipe": recipe, "backend": backend, "stream": False}, timeout=timeout)

    def search_registry(self, query: str, source: str = "huggingface", limit: int = 10, fmt: str = "gguf") -> dict:
        """GET /v1/registry/search: candidate repos from HF/ModelScope metadata --
        read-only, no download. has_gguf is a hint, not proof; pull_variants
        does the real file-level check."""
        return self._get(
            "/v1/registry/search", params={"query": query, "source": source, "limit": limit, "format": fmt}
        )

    def pull_variants(self, checkpoint: str) -> dict:
        """GET /v1/pull/variants: quantization variants + sizes for an HF GGUF
        repo -- reads public HF metadata only, no download."""
        return self._get("/v1/pull/variants", params={"checkpoint": checkpoint})

    def pull_model(
        self,
        model_name: str,
        recipe: str | None = None,
        checkpoint: str | None = None,
        timeout: float | None = 1800,
        **kwargs,
    ) -> dict:
        """POST /v1/pull: downloads real model weights, potentially GBs.

        Confirmed shape from Lemonade's own docs (not exercised live, to avoid
        an unrequested multi-GB download):
        - Installing an already-registered model needs only `model_name`.
        - Registering a new one needs `model_name` (must be "user.Name"-
          namespaced), `recipe`, and `checkpoint` in "owner/repo:VARIANT" form
          (e.g. "unsloth/Phi-4-mini-instruct-GGUF:Q4_K_M") -- pull_variants()
          gives you the exact variant names to build that string from.
        Response: {"status": "success"|"error", "message": "..."}.
        """
        payload = {"model_name": model_name, "stream": False, **kwargs}
        if recipe:
            payload["recipe"] = recipe
        if checkpoint:
            payload["checkpoint"] = checkpoint
        return self._post("/api/v1/pull", payload, timeout=timeout)

    def start_model_download(self, model_name: str, recipe: str | None = None, checkpoint: str | None = None, **kwargs) -> dict:
        """POST /v1/pull with stream=true, subscribe=false: starts a
        server-owned background download job and returns its initial
        snapshot immediately -- the download continues on the Lemonade
        server even if this client (or the dashboard page) disconnects, so
        it survives a page refresh and several can run concurrently.

        Shape confirmed from Lemonade's docs (not exercised live, to avoid
        an unrequested download): returns a job dict with "id" (format
        "model:<model_name>"), "status" ("downloading" initially), "percent",
        "bytes_downloaded"/"bytes_total" for the current file, and
        "cumulative_bytes_downloaded"/"total_download_size" for the whole
        job. Poll list_downloads() for progress; control_download() to
        pause/cancel/remove.
        """
        payload = {"model_name": model_name, "stream": True, "subscribe": False, **kwargs}
        if recipe:
            payload["recipe"] = recipe
        if checkpoint:
            payload["checkpoint"] = checkpoint
        return self._post("/api/v1/pull", payload)

    def list_downloads(self) -> list[dict]:
        """GET /v1/downloads: every server-owned download job (any state),
        started via start_model_download. Confirmed live: returns [] when
        no jobs exist."""
        return self._get("/api/v1/downloads")

    def control_download(self, download_id: str, action: str) -> dict:
        """POST /v1/downloads/control: action is "pause", "cancel", or
        "remove". Returns the latest job snapshot (pause/cancel) or
        {"status": "ok"} (remove) per Lemonade's docs."""
        return self._post("/api/v1/downloads/control", {"id": download_id, "action": action})

    def classify(self, input_text: str, model: str | None = None, top_k: int | None = None) -> dict:
        """POST /v1/classify — run an ONNX encoder text-classifier (PII, prompt-safety,
        domain, etc.) and return per-label scores in [0, 1].

        The target model must use the onnxruntime recipe. Passing `model`
        auto-loads it if it isn't already (no separate load() call needed);
        omitting it only works when exactly one classification model is
        already loaded, else Lemonade returns a 400. `top_k` limits the
        response to the k highest-scoring labels.

        Response shape confirmed from Lemonade's own server source
        (src/cpp/server/server.cpp's handle_classify -- the envelope it
        builds around the onnxruntime backend's raw output is the public
        contract, not the backend's own response):
          {
            "object": "classification",
            "model": "<model_id>",
            "labels": {"<label>": <score 0..1>, ...}
          }
        Note "labels" is an object keyed by label name, not a list, and
        there is no timing field (no time_to_classify_ms) anywhere in the
        envelope.
        """
        payload: dict = {"input": input_text}
        if model is not None:
            payload["model"] = model
        if top_k is not None:
            payload["top_k"] = top_k
        return self._post("/v1/classify", payload)

    def text_to_speech(
        self,
        input_text: str,
        model: str | None = None,
        voice: str | None = None,
        speed: float | None = None,
        response_format: str = "wav",
        timeout: float | None = None,
    ) -> bytes:
        """POST /v1/audio/speech -- generate spoken audio and return the raw
        audio bytes (NOT JSON, unlike every other method on this client).

        Confirmed from Lemonade's own server source (handle_audio_speech in
        src/cpp/server/server.cpp): on success the body is the raw encoded
        clip with Content-Type set from a fixed format->MIME table (wav ->
        "audio/wav", confirmed to start with a standard RIFF/WAVE header --
        parseable by Python's stdlib `wave` module for exact duration). On
        error the body is JSON, `{"error": {"message": ..., "type": ...}}`,
        which _raise_for_status_with_message already handles generically.
        response_format defaults to "wav" here (not Lemonade's own default of
        "mp3") because WAV's header gives an exact, dependency-free duration
        for computing real-time-factor; mp3/opus would need a decoder.
        """
        payload: dict = {"input": input_text, "response_format": response_format}
        if model is not None:
            payload["model"] = model
        if voice is not None:
            payload["voice"] = voice
        if speed is not None:
            payload["speed"] = speed
        resp = self._session.post(
            f"{self.base_url}/v1/audio/speech", json=payload, timeout=timeout or self.timeout
        )
        _raise_for_status_with_message(resp)
        return resp.content

    def speech_to_text(
        self,
        audio_bytes: bytes,
        filename: str,
        model: str,
        language: str | None = None,
        response_format: str = "json",
        timeout: float | None = None,
    ) -> dict:
        """POST /v1/audio/transcriptions -- multipart/form-data, NOT JSON,
        unlike every other method on this client (confirmed from Lemonade's
        own upstream test suite, test/server_whisper.py: the file is sent as
        `files={"file": (...)}` with `model`/`language`/`response_format` as
        form fields, not a JSON body).

        response_format defaults to "json" here (not passed through
        untouched) because that is the one format guaranteed to return
        {"text": ...} as real JSON -- "verbose_json"'s "segments" field is
        optional per the same test suite ("FLM has none to report"), and
        "text"/"srt"/"vtt" return plain text, not JSON.
        """
        files = {"file": (filename, audio_bytes)}
        data: dict = {"model": model, "response_format": response_format}
        if language is not None:
            data["language"] = language
        resp = self._session.post(
            f"{self.base_url}/v1/audio/transcriptions",
            files=files,
            data=data,
            timeout=timeout or self.timeout,
        )
        _raise_for_status_with_message(resp)
        return resp.json()

    def generate_image(
        self,
        prompt: str,
        model: str,
        size: str = "512x512",
        steps: int | None = None,
        n: int = 1,
        cfg_scale: float | None = None,
        seed: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        """POST /v1/images/generations -- confirmed from Lemonade's own
        upstream test suite (test/server_sd.py): response is
        {"data": [{"b64_json": "<base64 image>"}], "created": <unix ts>},
        with no timing field anywhere, same as classify/tts/speech_to_text.
        """
        payload: dict = {"model": model, "prompt": prompt, "size": size, "n": n}
        if steps is not None:
            payload["steps"] = steps
        if cfg_scale is not None:
            payload["cfg_scale"] = cfg_scale
        if seed is not None:
            payload["seed"] = seed
        return self._post("/v1/images/generations", payload, timeout=timeout)

    def edit_image(
        self,
        image_bytes: bytes,
        filename: str,
        prompt: str,
        model: str,
        mask_bytes: bytes | None = None,
        mask_filename: str = "mask.png",
        size: str = "512x512",
        steps: int | None = None,
        cfg_scale: float | None = None,
        seed: int | None = None,
        n: int = 1,
        timeout: float | None = None,
    ) -> dict:
        """POST /v1/images/edits -- multipart/form-data, NOT JSON (confirmed
        from Lemonade's own server source, src/cpp/server/server.cpp:
        non-multipart requests are rejected 400). Same response envelope as
        generate_image: {"data": [{"b64_json": ...}]}.
        """
        files: dict = {"image": (filename, image_bytes)}
        if mask_bytes is not None:
            files["mask"] = (mask_filename, mask_bytes)
        data: dict = {"model": model, "prompt": prompt, "size": size, "n": str(n)}
        if steps is not None:
            data["steps"] = str(steps)
        if cfg_scale is not None:
            data["cfg_scale"] = str(cfg_scale)
        if seed is not None:
            data["seed"] = str(seed)
        resp = self._session.post(
            f"{self.base_url}/v1/images/edits", files=files, data=data, timeout=timeout or self.timeout
        )
        _raise_for_status_with_message(resp)
        return resp.json()

    def create_image_variation(
        self,
        image_bytes: bytes,
        filename: str,
        model: str,
        size: str = "512x512",
        n: int = 1,
        timeout: float | None = None,
    ) -> dict:
        """POST /v1/images/variations -- multipart/form-data, NOT JSON, same
        as edit_image. No prompt or mask (confirmed from Lemonade's own
        server source: this endpoint doesn't accept them at all -- an
        unguided variation of the input image). Same response envelope as
        generate_image: {"data": [{"b64_json": ...}]}.
        """
        files = {"image": (filename, image_bytes)}
        data = {"model": model, "size": size, "n": str(n)}
        resp = self._session.post(
            f"{self.base_url}/v1/images/variations", files=files, data=data, timeout=timeout or self.timeout
        )
        _raise_for_status_with_message(resp)
        return resp.json()

    def generate_audio(
        self,
        prompt: str,
        model: str,
        lyrics: str | None = None,
        vocal_language: str | None = None,
        response_format: str = "wav",
        timeout: float | None = None,
    ) -> bytes:
        """POST /v1/audio/generations -- text/music/sound-effect generation
        (acestep, thinksound), NOT speech. Confirmed from Lemonade's own
        server source (handle_audio_generations in src/cpp/server/server.cpp):
        returns raw audio bytes on success (same response_format->MIME table
        as text_to_speech), JSON error envelope on failure. response_format
        defaults to "wav" here (not Lemonade's own default) for the same
        reason as text_to_speech: an exact, dependency-free duration via the
        stdlib `wave` module.
        """
        payload: dict = {"model": model, "prompt": prompt, "response_format": response_format}
        if lyrics is not None:
            payload["lyrics"] = lyrics
        if vocal_language is not None:
            payload["vocal_language"] = vocal_language
        resp = self._session.post(
            f"{self.base_url}/v1/audio/generations", json=payload, timeout=timeout or self.timeout
        )
        _raise_for_status_with_message(resp)
        return resp.content

    def get_embeddings(
        self,
        input_texts: list[str],
        model: str,
        encoding_format: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        """POST /v1/embeddings -- confirmed from Lemonade's own upstream test
        suite (test/server_llm.py): a pure passthrough of llama.cpp's own
        OpenAI-shaped /v1/embeddings response, not something Lemonade builds
        itself. `input` accepts a list of strings (batched embedding in one
        request) as well as a single string; this client always sends a
        list. Response: {"data": [{"embedding": [float, ...], "index": ...}],
        "usage": {"prompt_tokens": ..., "total_tokens": ...}} -- token counts
        only, no timing field. Supported by llamacpp (GGUF) and FastFlowLM
        (NPU) models.
        """
        payload: dict = {"model": model, "input": input_texts}
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format
        return self._post("/v1/embeddings", payload, timeout=timeout)

    def rerank(
        self,
        query: str,
        documents: list[str],
        model: str,
        top_n: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        """POST /v1/rerank (aliases /reranking, /reranker) -- confirmed from
        Lemonade's own upstream test suite (test/server_llm.py): a pure
        passthrough of llama.cpp's own /v1/rerank response, {"results":
        [{"index": int, "relevance_score": float}, ...]} -- token counts
        only, no timing field. llamacpp (GGUF) only; FastFlowLM does not
        support reranking.
        """
        payload: dict = {"query": query, "documents": documents, "model": model}
        if top_n is not None:
            payload["top_n"] = top_n
        return self._post("/v1/rerank", payload, timeout=timeout)

    def generate_3d(
        self,
        image_bytes: bytes,
        model: str,
        resolution: str | None = None,
        bg_removal: str | None = None,
        seed: int | None = None,
        uv: str | None = None,
        timeout: float | None = None,
    ) -> bytes:
        """POST /v1/3d/generations -- image-to-mesh (Lemonade's trellis
        recipe). Confirmed from Lemonade's own server source
        (handle_3d_generations in src/cpp/server/server.cpp): JSON request
        with a base64-encoded input image, `response_format` fixed to "glb"
        (the only supported value -- not exposed as a parameter here since
        there's nothing else to choose), and the response body is the raw
        binary mesh itself (Content-Type "model/gltf-binary"), not a JSON
        envelope -- same pattern as text_to_speech/generate_audio. `resolution`
        must be "512", "1024", or "1536" if given; `bg_removal` must be
        "threshold" or "birefnet"; `uv` must be "box" or "xatlas" -- all
        confirmed against the same source's request validation.
        """
        payload: dict = {
            "model": model,
            "image": base64.b64encode(image_bytes).decode("ascii"),
        }
        if resolution is not None:
            payload["resolution"] = resolution
        if bg_removal is not None:
            payload["bg_removal"] = bg_removal
        if seed is not None:
            payload["seed"] = seed
        if uv is not None:
            payload["uv"] = uv
        resp = self._session.post(
            f"{self.base_url}/v1/3d/generations", json=payload, timeout=timeout or self.timeout
        )
        _raise_for_status_with_message(resp)
        return resp.content

    def create_job(self, name: str, steps: list[dict], inputs: dict | None = None, timeout: float | None = None) -> dict:
        """POST /v1/jobs -- submits a durable, server-side sequence of ops
        (load/chat/unload/system_stats/etc.) that Lemonade itself executes
        and persists (survives client disconnect and server restart).
        Confirmed live and against Lemonade's own server source
        (handle_jobs_create in src/cpp/server/server.cpp): returns 202
        {"id": "<job_id>"} immediately -- the job runs asynchronously, poll
        get_job() for progress/completion. Each step dict needs "id", "op",
        and "params" (op-specific; e.g. "load"/"chat" take {"model": ...,
        ...} -- note the key is "model", NOT "model_name" like the direct
        /api/v1/load endpoint, confirmed against the job engine's own
        provider code).
        """
        payload: dict = {"name": name, "steps": steps}
        if inputs:
            payload["inputs"] = inputs
        return self._post("/v1/jobs", payload, timeout=timeout)

    def get_job(self, job_id: str, timeout: float | None = None) -> dict:
        """GET /v1/jobs/{id} -- the full job record: {id, name, status,
        inputs, context, steps: [...], cursor, created_at, ...}. `status` is
        "queued"/"running"/"paused" while in flight, "completed"/"failed"/
        "interrupted" when done (confirmed against Lemonade's own JobStatus
        enum). `context` maps each step's id directly to that step's raw
        output (confirmed live and against Lemonade's own integration test,
        test/server_jobs.py) -- the more convenient way to read a
        completed step's data than digging through the `steps` array.
        """
        return self._get(f"/v1/jobs/{job_id}", timeout=timeout)

    def delete_job(self, job_id: str, timeout: float | None = None) -> dict:
        """DELETE /v1/jobs/{id} -- removes a finished job record. Returns
        {"status": "deleted"} on success, per Lemonade's own server source."""
        resp = self._session.delete(f"{self.base_url}/v1/jobs/{job_id}", timeout=timeout or self.timeout)
        _raise_for_status_with_message(resp)
        return resp.json()


def job_progress_percent(job: dict) -> float:
    """Best-effort overall completion percent for a /v1/downloads job.

    Prefers cumulative_bytes_downloaded/total_download_size (whole-job
    progress) over the job's own "percent" field, which per Lemonade's docs
    tracks only the *current file* within a multi-file download and would
    misleadingly reset partway through a job.
    """
    total = job.get("total_download_size") or 0
    if total > 0:
        return max(0.0, min(100.0, 100.0 * (job.get("cumulative_bytes_downloaded") or 0) / total))
    if job.get("bytes_total"):
        return max(0.0, min(100.0, 100.0 * (job.get("bytes_downloaded") or 0) / job["bytes_total"]))
    return float(job.get("percent") or 0)
