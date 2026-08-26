# TODO

Deferred work, tracked here instead of just in conversation so it survives
context resets. Roughly ordered by what unlocks the most for the AMD
Lemonade Developer Challenge submission.

## 1. Public leaderboard / hosting

LemonMatrix's leaderboard is currently local-only (`lemonmatrix dashboard`).
LemonMetrics (a direct competitor project in the same challenge) already has
a public leaderboard at lemonmetrics.github.io. That's a real gap for the
challenge's "community impact" criterion: a link judges can click beats a
tool they have to install and run themselves.

Plan:
- Static site generator that reads `results/**/*.json` and renders a static
  leaderboard page (reuse `results_store.py`'s CSV/row logic rather than the
  live Flask views). Publish to GitHub Pages via a GitHub Action on push to
  `main` -- no hosted backend, keeps "Lemonade is the only source of truth"
  intact since the data still comes from local runs, not a server LemonMatrix
  itself operates.
- Community submission flow: accept results via PR (drop a result JSON under
  `results/<profile>/`), with a required CI check running `validate.py`'s
  schema validation before merge. This is the actual "open leaderboard people
  contribute to" story, not just a nicer local UI.

## 2. Demo video / recorded walkthrough

AMD's challenge page encourages a recorded demo or public walkthrough.
Suggested 3-5 minute structure, in order of what's actually distinctive:
1. `lemonmatrix profile detect` auto-discovering hardware live -- zero manual
   entry.
2. A sweep across two backends/power-states producing comparable rows.
3. A run that fails a validity gate, shown in the leaderboard but excluded
   from ranking -- sells the confounder-control pitch visually.
4. A router run with the route trace (`lemonmatrix trials`) -- nobody else in
   the Discord project roundup demos Lemonade's `collection.router` feature.

Screen recording + voiceover is enough; this doesn't need production value.

## 3. Bigger Lemonade-feature exposures needing a design decision first

- **`classify` run type -- done.** Live-verified `/v1/classify`'s real
  response envelope against Lemonade's own server source
  (`src/cpp/server/server.cpp`'s `handle_classify`, in the `lemonade-sdk/lemonade`
  GitHub repo): `{"object": "classification", "model": ..., "labels": {label:
  score, ...}}` -- `labels` is an object keyed by label name, not a list, and
  there is no timing field, so latency has to be measured client-side as
  wall-clock time around the request (still just timing a call to the
  profile's own client, not local hardware). Fixed the stale docstring in
  `client.py` that had guessed a list-of-objects shape plus a
  `time_to_classify_ms` field that doesn't exist. Shipped as a fully separate
  pipeline (`schema/classify_result.schema.json`, `validate_classify_result()`,
  `ClassifyConfig`/`run_classify()` in `bench.py`,
  `save_classify_result()`/`list_classify_results()` in `results_store.py`,
  `lemonmatrix classify`/`classify-results` CLI commands) rather than a
  bolt-on `run_type` on the existing schema -- classification latency isn't
  comparable to LLM token throughput, so results never appear on the
  model/router leaderboard. CLI-first for now, same as router runs were
  initially; no dashboard page yet (see below).
- **Non-text modality benchmarking -- TTS done, others remain.**
  `capabilities.available_backends()` still explicitly filters to
  `modality="Text generation"` today (`src/lemonmatrix/capabilities.py`) even
  though Lemonade also serves audio, image, and speech recipes. Confirmed
  live against a real 11.6.0 instance's `/api/v1/system-info` recipes tree:
  `acestep` (Audio generation), `kokoro` (Text-to-speech), `moonshine`/
  `whispercpp` (Speech-to-text), `sd-cpp`/`thenoise` (Image generation),
  `thinksound` (Audio generation), `trellis` (3D generation) all report their
  own `modality` string and `"uses_ctx_size": false`.

  **TTS is shipped**: live-verified `POST /v1/audio/speech`'s real contract
  against Lemonade's own server source (`handle_audio_speech` in
  `server.cpp`) and its upstream Python test suite (`test/server_tts.py`) --
  response is raw audio bytes (not JSON), Content-Type from a fixed
  format->MIME table (`wav` -> `audio/wav`, a real RIFF/WAVE container), no
  timing field. Metric: real-time-factor (`audio_duration_s` read exactly off
  the WAV header via the stdlib `wave` module, divided by client-measured
  wall-clock generation time). Shipped as its own pipeline
  (`schema/tts_result.schema.json`, `validate_tts_result()`,
  `TTSConfig`/`run_tts()`/`_wav_duration_seconds()` in `bench.py`,
  `save_tts_result()`/`list_tts_results()`, `lemonmatrix tts`/`tts-results`
  CLI commands), same pattern as `classify`. Could not execute an actual
  live round-trip on this sandbox -- kokoro's `koko` binary needs GLIBC_2.38,
  which this host doesn't have (same root cause hit with onnxruntime's
  `ort-server`) -- so the implementation rests on the source/test-suite
  ground truth rather than a live response, same evidentiary bar used
  successfully for `classify`.

  **STT and image-gen are shipped too.** Both hit the exact same GLIBC_2.38/
  GLIBCXX_3.4.32 wall as onnxruntime/kokoro when actually run on this sandbox
  (confirmed via `ldd` against the real `whisper-server`/`sd-cli` binaries
  after installing the backends live) -- llamacpp is apparently the only
  recipe built against an older glibc baseline. Both rest on source/upstream-
  test-suite ground truth (`test/server_whisper.py`, `test/server_sd.py` in
  `lemonade-sdk/lemonade`) rather than a live response, same bar used for
  `classify`/`tts`.
  - **STT** (`lemonmatrix stt`, whispercpp/moonshine via multipart
    `POST /v1/audio/transcriptions` -- confirmed NOT JSON, unlike every other
    endpoint on this client). Metric: real-time-factor, using the *input*
    clip's exact duration (read from its own WAV header, since the caller
    supplies it) divided by client-measured transcription wall-clock time.
  - **Image generation** (`lemonmatrix imagegen`, sd-cpp/thenoise via
    `POST /v1/images/generations`). Metric: images/sec at a fixed
    size+step-count (both required in the schema -- they materially change
    generation cost, so they're recorded like `context_length` is for LLM
    runs, not left as free-form metadata).

  **Audio generation (acestep/thinksound) -- done.** `POST /v1/audio/generations`
  (text/music/sound-effect generation, NOT speech) is structurally identical
  to `/v1/audio/speech`'s no-timing-field/WAV-duration situation, so
  `run_audiogen()` reuses `_wav_duration_seconds()` and the same
  real-time-factor metric as TTS -- but writes to its own schema/results
  tree (`audiogen_result.schema.json`, `results/<profile>/audiogen/`)
  rather than merging with TTS results, since generating music is a
  different task with a different compute profile than speech.

  **Found and fixed a real, live-confirmed bug while building this**: both
  `run_stt()` and `run_imagegen()` (and, without this fix, the new
  `run_audiogen()`) called `/api/v1/load` with no backend selector at all,
  under the assumption -- true for onnxruntime/kokoro, but NOT universal --
  that these recipes have exactly one backend. Confirmed live against a real
  instance: whispercpp (`cpu/metal/npu/rocm/vulkan`), sd-cpp
  (`cpu/cuda/metal/rocm/vulkan`), and acestep (`cuda/rocm/vulkan`) all report
  `"selectable_backend": true` with a `"default_backend": "cpu"` fallback --
  so asking to benchmark e.g. `whispercpp-vulkan` was silently running on
  CPU instead, mislabeling the leaderboard entry. Fixed by extracting
  `run_sweep`'s existing (correct) `resolve_backend()`-based load-kwargs
  logic into a shared `_resolve_backend_load_kwargs()` helper and using it in
  `run_stt`/`run_imagegen`/`run_audiogen`; added regression tests
  asserting the fake server actually receives the selector. The new
  server-verified device check (below) would have caught this class of bug
  after the fact too, but this fixes the root cause.

  **3D-gen (trellis) still open.** Same pattern, still needing its own
  live-verified endpoint contract and metric definition, and its own
  schema/leaderboard per the precedent above. Bigger lift than the others:
  takes a base64 *input image*, not text, which LemonMatrix has no plumbing
  for yet.
- **Server-verified `exclusive_run` -- done.** `_ExclusivityMonitor` in
  `bench.py` background-polls the profiled instance's own `/api/v1/health`
  during warmup and measured trials and overrides the caller's assertion to
  `false` if another loaded model reports `is_busy`/`is_streaming`; falls
  back to the caller's assertion (with a note) when every poll fails or the
  run finishes before the first poll fires. Confirmed live: `is_busy`
  genuinely flips `true` for the duration of an in-flight request. Used by
  every run_* pipeline (`run_sweep`, `run_classify`, `run_tts`, `run_stt`,
  `run_imagegen`, `run_audiogen`).
- **Server-verified `compute_engine` and `model_reload_free` -- done.**
  Deep-read the real Lemonade server source at a local checkout
  (`router.cpp`'s `get_all_loaded_models()`): every `all_models_loaded[]`
  entry also carries the *actual* `device` Lemonade used (a "cpu"/"gpu"/
  "npu" bitmask string -- confirmed it cannot distinguish integrated from
  discrete GPU) and `watchdog_reset` (true once Lemonade's own backend
  watchdog force-restarts the model's subprocess mid-run). `_ExclusivityMonitor`
  now also tracks these for the run's own model; a device that contradicts
  the claimed `compute_engine`, or a watchdog reset observed during
  measurement, invalidates the run with a note. `model_reload_free` was
  previously hardcoded `true` ("there is no unload here by design") --
  true of LemonMatrix's own calls, but not a guarantee against Lemonade
  restarting the backend on its own; it's now actually verified. Router runs
  are exempt from the device check (no single physical device to verify).
  Confirmed live end to end against a real instance (a real load's `device`
  correctly read back as `"gpu"`, `watchdog_reset: false`).
- **Classify/TTS dashboard pages -- done.** `/profiles/<name>/classify` and
  `/profiles/<name>/tts` each combine a run form (pre-filled from the
  profile's live models/backends, filtered to the right recipe/modality) and
  a results table, following `run_form.html`'s conventions. Verified live
  against a real instance end to end after fixing a real bug this surfaced
  (see below).
- **STT/imagegen/audiogen dashboard pages -- done.** `/profiles/<name>/stt`
  (multipart file upload for the WAV input, with a friendly error if the
  uploaded file isn't a real WAV), `/profiles/<name>/imagegen`, and
  `/profiles/<name>/audiogen` follow the classify/tts pages' exact
  conventions (pre-filled model/backend dropdowns filtered to the right
  recipe/modality, a results table below the form). Verified live against a
  real instance.
- **Bug found and fixed while building the classify page:**
  `capabilities.available_backends()`'s OS filter collapsed a recipe's
  `support` list into one entry per backend key, keeping only the *last*
  entry when a backend key appeared more than once (onnxruntime's `cpu`
  backend has three separate `support` entries -- one per device family:
  x86_64/windows, x86_64+arm64/linux, arm64/macos). Since the last entry in
  Lemonade's own list order happened to be macos-only, this silently hid an
  *installed* onnxruntime backend from a linux host. Fixed by unioning the
  OS lists across every entry sharing a backend key instead of keeping the
  last one; added a regression test. llamacpp never hit this because none of
  its backend keys repeat in its `support` list.

## 4. Ranking-integrity gaps closed in one pass

Went through idea.md's own "missing or not yet trustworthy" list item by item. Two turned out to be permanent, structural limitations of Lemonade's own API (not something to build), and the rest were genuinely implementable:

- **Thermal validity and `power_state` -- confirmed permanently unverifiable, not pending.** Checked Lemonade's own metrics source directly: no temperature sensor and no AC/battery status are exposed anywhere in the server. Documented as a structural disclosure in idea.md rather than left as an ambiguous TODO.
- **Model class and active parameter count -- same permanent limitation.** Confirmed against `model_info_to_json` in Lemonade's own source: no dedicated field for either exists to verify against.
- **Quantization -- done, partially.** `capabilities.parse_quantization()` (extracted from the dashboard's existing pre-fill heuristic, now shared) parses Lemonade's own checkpoint string for the loaded model; `run_sweep()` flags a run whose claimed `model.quantization` contradicts it. Best-effort (not every checkpoint string encodes a variant), but a real improvement over a pure user assertion. Confirmed live against a real instance with a deliberately wrong quant string.
- **Confidence intervals -- done.** `ci95_half_width` on decode/prefill/TTFT, using the actual Student's t critical value for the trial count (a small lookup table for df 1-30, falling back to the 1.96 normal approximation beyond that) rather than a fixed z-value that understates the interval for small samples.
- **Reproducibility -- done.** Every model/router result now carries `prompt_sha256`, so two runs can be confirmed to have used the identical prompt without embedding the (potentially long) text itself.
- **Cost model's amortized-hardware half -- done, and it's what actually makes the field usable.** `hardware_cost_usd`/`hardware_lifetime_hours` inputs (CLI flags, dashboard form fields) spread hardware cost over decode tokens/sec, independent of power data. Since the energy half is always `None` in practice (Lemonade never reports power draw, confirmed above), this is the first time `cost_per_1k_tokens_usd` populates on a real run at all.
- **Leaderboard default valid-only filter -- done.** The dashboard's `/` leaderboard now hides invalid runs unless a user explicitly clears the filter; a hidden form marker distinguishes "never filtered" from "explicitly cleared" so sort-link clicks (which carry forward whatever's already in the query string) don't silently reset it.
- **A second, unrelated packaging bug found and fixed while verifying the above.** Built an actual wheel to check the "not confirmed included in a built wheel" TODO item, and found `schema/*.json` at the repo root were independently-maintained files while `pyproject.toml`'s package-data only ever pulled from `src/lemonmatrix/schema/` -- only `result.schema.json` had ever been copied there by hand, so the five newer schemas were silently missing from every built wheel (confirmed: `validate_classify_result()` etc. raised `FileNotFoundError` from a real installed-package test). Fixed by making the repo-root copies symlinks into the package, eliminating the possibility of drift by construction, with a regression test.

Still open from idea.md's list: durable job cancellation/retry/cross-profile scheduling (see the job-engine note above), and everything under "hosted, multi-user" (public leaderboard is TODO item 1, still not started).

## 5. Embeddings and reranking -- done, CLI-first

`lemonmatrix embeddings`/`embeddings-results` (`POST /v1/embeddings`, llamacpp GGUF + FastFlowLM NPU) and `lemonmatrix rerank`/`rerank-results` (`POST /v1/rerank`, llamacpp GGUF only). Both are pure passthroughs of llama.cpp's own OpenAI-shaped responses (confirmed from `test/server_llm.py` in the local `lemonade` checkout) -- no timing field either, so latency is client-side wall-clock like every other non-text-modality pipeline. Metrics scale with batch size (embeddings) or document count (reranking), both recorded for reproducibility the same way size/steps are for image generation.

**Live-verified the contract, not yet the happy path.** Confirmed against the real instance that both endpoints are reachable and that requesting embeddings/reranking from a plain chat model returns a genuinely informative error: `"This server does not support embeddings/reranking. Start it with --embeddings/--reranking"` -- llama-server itself needs to be launched with that flag, which is Lemonade's own responsibility based on the loaded model's registered labels/type, not something LemonMatrix's `/api/v1/load` call controls. Exercising the actual happy path needs a real embedding- or reranking-labeled GGUF model pulled (a genuine multi-hundred-MB download not done in this pass) -- the request/response shapes themselves come from Lemonade's own upstream test suite, the same evidentiary bar used for classify/tts/stt/imagegen/audiogen, all of which had the same live-execution limitation for other reasons (glibc). No dashboard pages yet, same as the other non-text modalities initially.

## 6. Image edits/variations, and 3D mesh generation -- done, CLI-first

**Image edits/variations** extend the existing `ImageGenConfig`/`run_imagegen`/`imagegen_result.schema.json` pipeline (via a new `operation: "generate"|"edit"|"variation"` field) rather than adding new infrastructure, per the plan -- both endpoints run through the same model and produce the same metric shape as `/v1/images/generations`. Confirmed from Lemonade's own server source that both are multipart-only (400 if not), and that `/v1/images/variations` doesn't accept a prompt, steps, cfg_scale, or seed at all -- the schema makes those fields optional (not required) and `run_imagegen` omits them from the metrics for that operation rather than reporting values that were never actually sent. `lemonmatrix imagegen --operation edit --input-image ... --mask-image ...` / `--operation variation --input-image ...`.

**Image upscale is deliberately NOT implemented.** Confirmed against Lemonade's own server source (`handle_image_upscale`) that it shells out to a `sd-cli -M upscale` subprocess per request rather than going through the normal `auto_load_model_if_needed`/Router/`WrappedServer` path every other endpoint uses -- there is no persistent model residency, so it's invisible to `/api/v1/health` and there's nothing for this tool's exclusivity/device/reload-freedom checks to verify against. Forcing it into the same pipeline would mean silently fabricating those validity fields.

**3D generation** (`lemonmatrix meshgen`/`meshgen-results`, trellis recipe via `POST /v1/3d/generations`) is its own pipeline/schema (`meshgen_result.schema.json`) -- image-to-mesh via a base64-encoded input image, raw glTF-binary response (same no-JSON-envelope pattern as `/v1/audio/speech`), metric is meshes/sec. Confirmed live: unlike every other newer backend hit this session, `trellis-server` actually **runs** on this sandbox (no GLIBC_2.38 wall) -- it's apparently built like llamacpp rather than the newer onnxruntime/kokoro/whispercpp/sd-cpp/acestep binaries. Didn't exercise the full happy path live: the suggested test model (`TRELLIS-3D`, checkpoint `ilintar/trellis2-gguf`) is a 15.4 GB download, too large to justify pulling just for verification -- the request/response contract itself comes from Lemonade's own server source, the same evidentiary bar used everywhere else. No dashboard page yet.

## 7. Job-engine execution -- done, opt-in

Delegates a single combination's execution to Lemonade's own `POST /v1/jobs` (durable, crash-persistent, survives client disconnect and server restart -- confirmed against Lemonade's own source) instead of this process making N sequential direct HTTP calls. `run_sweep_via_job()` in `bench.py`; `--via-job-engine` on `lemonmatrix run` and `lemonmatrix sweep`; a checkbox on the dashboard's single-run form and sweep form; threaded through `sweep_batch.py`'s `_run_batch()` for durable dashboard batches.

**Live-verified end to end against a real instance**, including a real surprise: a job's `chat` step embeds `timings`/`usage` directly in its own output (accessible via `context[step_id]` after the job completes) -- no separate `/v1/stats` call needed at all, unlike the direct-HTTP path. Interleaving `system_stats` steps between trials gives the same per-trial memory/power sampling the direct path gets from `_resource_samples()`. Also confirmed a real, easy-to-guess-wrong detail from source: job `load`/`chat` step params use the key `"model"`, not `"model_name"` like the direct `/api/v1/load` endpoint.

**Deliberately additive, not a replacement.** Refactored `run_sweep()` to extract its statistics/validity/cost-model logic into a shared `_aggregate_sweep_result()` that both execution paths call, so the two produce byte-identical aggregated metrics from identical raw measurements (confirmed by a test running the same config through both and comparing) -- but `run_sweep()` itself, `SweepBatch`/`SweepStore`, and every default code path are untouched. Router runs aren't supported via the job engine (a router has no fixed backend/ctx_size for a job's `load` step); whole multi-combo sweeps still aren't expressed as a single job (no loop primitive exists server-side, confirmed against Lemonade's own job-system docs, so each combination is still its own job).
