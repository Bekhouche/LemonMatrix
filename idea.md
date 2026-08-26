# LemonMatrix Architecture

This document describes the core data model behind LemonMatrix: the profile abstraction, how profiles map to Lemonade instances, how configuration is auto-discovered, and how it all connects to the result schema.

## Challenge context

LemonMatrix is being developed for AMD's [Lemonade Developer Challenge](https://www.amd.com/en/developer/resources/technical-articles/2026/join-the-lemonade-developer-challenge.html), which began on February 15, 2026 and remains open while supplies last. The challenge accepts open-source Lemonade projects, including deep performance evaluations, and evaluates community impact, technical depth and quality, and creativity.

For submission, the project must remain accessible under an open-source license. The entrant must also join the AMD AI Developer Program, follow the Lemonade Discord `#AMDDevChallenge` channel, and submit the project through the member-site form. A recorded demo or other public walkthrough is encouraged.

## Core concept: one profile equals one Lemonade

A **profile** is LemonMatrix's representation of a single Lemonade instance. Every profile points at exactly one Lemonade endpoint, identified by its IP or URL. This one-to-one mapping is the spine of the whole system.

Because Lemonade exposes an OpenAI-compatible API and knows its own hardware, a profile does not need to be filled in by hand. When you add a profile, LemonMatrix connects to the Lemonade endpoint, reads its information, and populates the profile automatically.

## The control plane

LemonMatrix is a control plane that can hold many profiles at once. Those profiles can point at very different targets:

- **A single local Lemonade**, for example one running on `localhost` on your own machine.
- **Multiple local Lemonade instances**, for example a fleet of developer machines reachable over the LAN, each with different hardware.
- **A cloud Lemonade**, for example a remote endpoint running on hosted hardware.

This is what turns LemonMatrix from a local benchmark script into a benchmarking platform. One control plane can drive benchmarks across a room full of machines and remote endpoints, each profile carrying its own auto-discovered identity.

```
                     LemonMatrix
                 (benchmark control plane)
                /          |          \
         Profile A     Profile B     Profile C
       Strix Halo      HX 370         Remote
         (local)       (local)        (cloud)
             |             |              |
         Lemonade      Lemonade       Lemonade
        localhost      LAN host     cloud endpoint
```

One profile equals one Lemonade instance, local or cloud.

## Auto-discovery

When a profile is created from a Lemonade IP or URL, LemonMatrix queries that instance and fills in the fixed facts about the machine and its software stack. These typically include:

- Device model
- CPU
- Integrated GPU (iGPU)
- Discrete GPU (dGPU)
- NPU
- Memory size and type
- Operating system and version
- Driver and ROCm version
- Available backends
- Lemonade version
- Models the instance can serve

Nobody types their hardware details by hand. This is not only a convenience. It is also what keeps submissions honest and comparable, because the environment record comes straight from the machine rather than from a person filling in a form.

## Fixed facts versus swept settings

The benchmark axes split into two kinds, and the split matters for how runs are organized.

**Fixed facts** are properties of the Lemonade instance itself. They are discovered once and belong to the profile:

- Device, CPU, iGPU, dGPU, NPU
- Operating system
- Lemonade version

**Swept settings** are the things that can vary on the same instance. They are chosen per run, and one profile produces many result rows as you sweep across them:

- Backend (for example llama.cpp via ROCm, then llama.cpp via Vulkan)
- Power state (for example on battery, then plugged in, then power-capped)
- Model class (dense, then Mixture-of-Experts)
- Quantization

So within a single profile you might run the ROCm backend and then the Vulkan backend, on battery and then plugged in, across a dense model and then an MoE model. Each combination is one benchmark run and one result row.

A note on operating system. The OS is a property of the machine, so a different OS is a different Lemonade instance and therefore a different profile, not a setting swept inside one profile. Backends such as ROCm versus Vulkan are a true in-profile sweep, because one machine can run several. Keep this line clear so the data model stays consistent as submissions grow.

## How profiles map to the result schema

The profile model maps directly onto the result schema. A benchmark result has an `environment` block and a `config` block, and they correspond exactly to the two kinds of axes above.

- The **profile** fills the `environment` block. A profile is, in effect, a saved and auto-populated environment fingerprint.
- The **sweep** fills the `config` block. Each swept combination produces one result record with its own `config`.

This means the same fingerprint is reused across every run on that machine, and only the swept settings change from row to row. That is what makes any slice of the leaderboard comparable.

## Result schema

Every benchmark run produces one record that conforms to the schema below. Prefill and decode are always reported separately, because one is compute-bound and the other is memory-bandwidth-bound, and collapsing them hides the most important part of the story. The `validity` block carries the confounder controls, so a run that fails a gate is recorded and shown but never ranked.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://lemonmatrix.github.io/schema/result.schema.json",
  "title": "LemonMatrix Benchmark Result",
  "description": "A single benchmark run for one model on one configuration. Prefill and decode are always reported separately.",
  "type": "object",
  "required": ["schema_version", "run_id", "timestamp", "model", "config", "environment", "metrics", "validity"],
  "additionalProperties": false,
  "properties": {
    "schema_version": {
      "type": "string",
      "const": "0.1.0"
    },
    "run_id": {
      "type": "string",
      "description": "Unique identifier for this run (UUID recommended)."
    },
    "timestamp": {
      "type": "string",
      "format": "date-time",
      "description": "UTC timestamp when the run started."
    },
    "model": {
      "type": "object",
      "required": ["name", "class", "quantization", "context_length"],
      "additionalProperties": false,
      "properties": {
        "name": { "type": "string", "description": "e.g. Llama-3.1-8B-Instruct" },
        "class": { "type": "string", "enum": ["dense", "moe"] },
        "parameters_b": { "type": "number", "description": "Total parameter count in billions." },
        "active_parameters_b": { "type": "number", "description": "Active parameters per token for MoE models, in billions." },
        "quantization": { "type": "string", "description": "Exact quant string, e.g. Q4_K_M, INT4, AWQ. Never normalize or approximate." },
        "context_length": { "type": "integer", "description": "Context window used for the run." }
      }
    },
    "config": {
      "type": "object",
      "description": "The matrix axes for this run. Filled by the sweep.",
      "required": ["compute_engine", "backend", "os", "power_state"],
      "additionalProperties": false,
      "properties": {
        "compute_engine": { "type": "string", "enum": ["cpu", "igpu", "dgpu", "npu", "hybrid"] },
        "backend": {
          "type": "string",
          "description": "e.g. llama.cpp-vulkan, llama.cpp-rocm, cuda, fastflowlm-npu"
        },
        "os": { "type": "string", "enum": ["windows", "linux", "macos", "docker"] },
        "power_state": { "type": "string", "enum": ["plugged", "battery", "power_capped"] },
        "power_cap_w": { "type": "number", "description": "Applied power cap in watts, if power_state is power_capped." }
      }
    },
    "environment": {
      "type": "object",
      "description": "Fingerprint of the machine and software stack. Auto-discovered from the profile. Required for any run to be comparable.",
      "required": ["device_model", "cpu", "memory_gb", "os_version", "driver_version"],
      "additionalProperties": false,
      "properties": {
        "device_model": { "type": "string", "description": "e.g. HP Ryzen AI Max+ 395 (Strix Halo)" },
        "cpu": { "type": "string", "description": "e.g. AMD Ryzen AI 9 HX 370" },
        "igpu": { "type": "string" },
        "dgpu": { "type": "string" },
        "npu": { "type": "string", "description": "e.g. XDNA 2" },
        "memory_gb": { "type": "number" },
        "memory_type": { "type": "string", "description": "e.g. LPDDR5X-7500" },
        "os_version": { "type": "string" },
        "driver_version": { "type": "string" },
        "rocm_version": { "type": "string" },
        "backend_version": { "type": "string", "description": "Version of the underlying engine, e.g. llama.cpp build hash." },
        "lemonade_version": { "type": "string" }
      }
    },
    "metrics": {
      "type": "object",
      "description": "Prefill and decode are reported separately because their hardware profiles differ.",
      "required": ["prefill", "decode", "ttft_ms", "peak_memory_gb"],
      "additionalProperties": false,
      "properties": {
        "prefill": {
          "type": "object",
          "required": ["tokens_per_sec"],
          "additionalProperties": false,
          "properties": {
            "tokens_per_sec": { "type": "number" },
            "joules_per_token": { "type": "number" }
          }
        },
        "decode": {
          "type": "object",
          "required": ["tokens_per_sec"],
          "additionalProperties": false,
          "properties": {
            "tokens_per_sec": { "type": "number" },
            "inter_token_latency_ms": { "type": "number" },
            "joules_per_token": { "type": "number" }
          }
        },
        "ttft_ms": { "type": "number", "description": "Time to first token, milliseconds." },
        "peak_memory_gb": { "type": "number" },
        "cost_per_1k_tokens_usd": {
          "type": "number",
          "description": "Optional. Derived from the cost model: amortized hardware plus energy at local price."
        },
        "energy_price_usd_per_kwh": {
          "type": "number",
          "description": "Local electricity price used for the cost figure, if reported."
        }
      }
    },
    "validity": {
      "type": "object",
      "description": "Confounder controls. A run that fails a gate must set valid=false and record why.",
      "required": ["valid", "warmup_discarded", "thermal_ok", "exclusive_run"],
      "additionalProperties": false,
      "properties": {
        "valid": { "type": "boolean", "description": "False if any hard gate failed. Invalid runs are shown but never ranked." },
        "warmup_discarded": { "type": "boolean", "description": "True if warmup iterations were excluded from timing." },
        "thermal_ok": { "type": "boolean", "description": "False if temperature crossed the throttle threshold during measurement." },
        "peak_temp_c": { "type": "number" },
        "exclusive_run": { "type": "boolean", "description": "True if no competing GPU/NPU workload ran during measurement." },
        "model_reload_free": { "type": "boolean", "description": "True if the model stayed resident for the whole run (no unload tax)." },
        "notes": { "type": "string", "description": "Any flags, caveats, or reasons a gate failed." }
      }
    }
  }
}
```

## Summary

A profile is one Lemonade instance, discovered automatically and reused as the environment fingerprint for every run on that machine. The control plane holds many profiles across local, multi-local, and cloud targets. Fixed facts belong to the profile and fill the `environment` block. Swept settings vary per run and fill the `config` block. This separation is what keeps every row on the leaderboard comparable.

## Implementation status

Status reviewed on August 18, 2026 (post router integration).

### Implemented

- A Python package and `lemonmatrix` CLI for adding, detecting, listing, inspecting, refreshing, and deleting Lemonade profiles.
- Defensive environment discovery from Lemonade's health and system-information APIs, including local and opt-in bounded LAN discovery.
- A Lemonade API client covering health, hardware, model, backend, download, load, completion, statistics, and classification operations.
- Two first-class run types: **model runs** (single LLM) and **router runs** (`collection.router` routing policies). The CLI accepts `--run-type router` and auto-fills model class, quantization, engine, and backend.
- Per-run `route_trace: true` on router chat requests so that per-trial routing decisions (`route_to`, `matched_rule`, `default_used`) are captured and persisted to a trials sidecar.
- Dispersion statistics alongside every mean: standard deviation, 95th-percentile, and trial count for decode throughput, prefill throughput, and TTFT.
- Raw per-trial measurements saved to a separate trials sidecar (`results/<profile>/trials/<run_id>.json`); viewable via `lemonmatrix trials <profile> <run_id>`.
- Engine/backend compatibility validation: incompatible combinations are flagged and the run is marked invalid. Router runs skip this check.
- Persistent, durable sweep batch state: SQLite-backed `SweepStore` survives dashboard restarts; in-flight batches are marked `interrupted` on startup.
- Full CLI: `profile add/detect/list/show/refresh/delete/debug`, `run`, `sweep`, `export` (CSV), `trials`, `classify`, `classify-results`, `tts`, `tts-results`, `stt`, `stt-results`, `imagegen`, `imagegen-results`, `audiogen`, `audiogen-results`, `embeddings`, `embeddings-results`, `rerank`, `rerank-results`, `meshgen`, `meshgen-results`, `dashboard`.
- Eight further, fully separate run kinds, each with its own schema and results tree rather than bolted onto the model/router leaderboard (their metrics aren't comparable to LLM token throughput):
  - **Classification** (`lemonmatrix classify`, ONNX text-classifiers via Lemonade's onnxruntime recipe and `/v1/classify`). Latency is measured client-side (the response carries no timing field, confirmed against Lemonade's own server source).
  - **Text-to-speech** (`lemonmatrix tts`, kokoro/openmoss via `/v1/audio/speech`). Metric is real-time-factor: generated clip duration (read exactly from the response WAV's own header) divided by client-measured wall-clock generation time.
  - **Speech-to-text** (`lemonmatrix stt`, whispercpp/moonshine via a multipart `/v1/audio/transcriptions` -- confirmed NOT JSON, unlike every other endpoint). Metric is real-time-factor using the *input* clip's known duration.
  - **Image generation** (`lemonmatrix imagegen`, sd-cpp/thenoise). Covers three operations sharing one pipeline (`--operation generate|edit|variation`): `generate` (`/v1/images/generations`, text-to-image), `edit` (`/v1/images/edits`, prompt-guided edit of an input image, optional mask), and `variation` (`/v1/images/variations`, unguided variation -- Lemonade's own endpoint doesn't accept a prompt/steps/cfg_scale/seed at all for this one, confirmed against its server source, so those fields are omitted from the metrics rather than misreported). Metric is images/sec at a fixed size (and step count for generate/edit). Image *upscale* (`/v1/images/upscale`) is deliberately not covered: confirmed it shells out to a CLI subprocess per request rather than going through the normal load/Router path, so there's no persistent model residency for this tool's exclusivity/reload checks to verify against.
  - **Audio generation** (`lemonmatrix audiogen`, acestep/thinksound via `/v1/audio/generations` -- text/music/sound-effect generation, not speech). Same real-time-factor approach as TTS, but a separate schema/results tree since it's a different task with a different compute profile.
  - **Embeddings** (`lemonmatrix embeddings`, llamacpp GGUF or FastFlowLM NPU via `/v1/embeddings`, a pure passthrough of llama.cpp's own response). Metric is embeddings/sec over a batch of inputs (batch size required for reproducibility, since latency scales with it); also records the returned embedding dimensionality.
  - **Reranking** (`lemonmatrix rerank`, llamacpp GGUF only via `/v1/rerank`, also a pure passthrough). Metric is documents/sec over a fixed document set for one query (document count required for reproducibility). Live-confirmed the error path (`"This server does not support embeddings/reranking. Start it with --embeddings/--reranking"` when the loaded model isn't one) but not yet the full happy path, which needs a real embedding/reranking-labeled GGUF model pulled.
  - **3D generation** (`lemonmatrix meshgen`, trellis recipe via `/v1/3d/generations`, image-to-mesh). Base64 input image in, raw glTF-binary response (same no-JSON-envelope pattern as `/v1/audio/speech`) -- metric is meshes/sec. Confirmed live that `trellis-server` actually runs on a sandbox where every *other* newer backend (onnxruntime/kokoro/whispercpp/sd-cpp/acestep) hit a GLIBC_2.38 wall -- it's apparently built like llamacpp instead. Didn't exercise the full happy path live: the suggested test model is a 15.4 GB download.
- **Opt-in job-engine execution** (`run_sweep_via_job()`, `--via-job-engine` on `lemonmatrix run`/`sweep`, a checkbox on the dashboard's single-run and sweep forms): delegates one combination's execution to a single durable Lemonade job (`POST /v1/jobs` -- confirmed against Lemonade's own source that jobs are crash-persistent and survive server restart) instead of this process making N sequential direct HTTP calls, so the run survives LemonMatrix itself disconnecting or being killed mid-run. Live-verified end to end against a real instance, including the surprising bit: a job's `chat` step embeds `timings`/`usage` directly in its own output (`context[step_id]`), so no separate `/v1/stats` call is needed at all, and `system_stats` steps interleaved between trials give the same per-trial memory/power sampling the direct-HTTP path gets from `_resource_samples()`. Both execution paths share one aggregation function (`_aggregate_sweep_result`) so the statistics/validity/cost-model logic is written once and produces identical results from identical raw measurements (confirmed by a test running the same config through both paths and comparing). Deliberately additive and opt-in, not a replacement: `SweepBatch`/`SweepStore` remain the default durability mechanism for multi-combo dashboard batches, and router runs aren't supported via the job engine (no fixed backend/ctx_size for a job's load step).
  - All five text/audio/image modalities have Flask dashboard pages (`/profiles/<name>/classify`, `/tts`, `/stt`, `/imagegen`, `/audiogen`) with a pre-filled run form (model/backend dropdowns filtered to the right recipe/modality) and a results table; the STT page uses a multipart file upload for the WAV input, with a friendly error if the uploaded file isn't actually a WAV. Embeddings/reranking/meshgen are CLI-only for now.
- All run_* pipelines now server-verify `compute_engine` and `model_reload_free` in addition to `exclusive_run`, via the same `/api/v1/health` polling: a mismatch between the claimed engine and Lemonade's own reported `device` for the loaded model, or a mid-run watchdog-triggered backend restart, now invalidates a run with a note (see "Missing or not yet trustworthy" below for what changed).
- While wiring this up, found and fixed a real bug: `run_stt`/`run_imagegen` called `/api/v1/load` with no backend selector at all, so whispercpp/sd-cpp (both report multiple real backends and a `cpu` default, confirmed live) were silently benchmarked on CPU regardless of the requested backend. Fixed by reusing `run_sweep`'s existing correct backend-resolution logic.
- CSV export from both the CLI (`lemonmatrix export`) and a leaderboard button in the dashboard; includes `run_type`, `router_default_model`, and the correct `dgpu`/`igpu` GPU column.
- A local Flask dashboard with profile management (including refresh/delete UI buttons), result details, a sortable/filterable leaderboard with an Export CSV link, single runs, and durable background sweep batches.
- Package data includes templates, schema, and static files so the dashboard works correctly outside an editable checkout.
- 267 passing automated tests using a fake Lemonade server, including router runs, route_trace capture, trials sidecar isolation, SweepStore round-trip/interrupt, CSV field correctness, server-verified exclusive_run/compute_engine/model_reload_free/quantization (health-polling detection of competing models, device mismatches, watchdog resets, and checkpoint-implied quant mismatches, with correct fallback when polling can't be confirmed to have run, and correct exemption of router runs from the device/quantization checks), classification/TTS/STT/image-generation (all three operations)/audio-generation/embeddings/reranking/3D-mesh-generation runs (schema validation, engine/backend mismatch, exclusivity, and isolation from the model/router results tree), regression tests confirming the backend selector is actually sent to `/api/v1/load` for stt/imagegen/audiogen/embeddings/rerank/meshgen, all five non-text-modality dashboard pages end to end (including the STT page's multipart upload and its rejection of a non-WAV file), the leaderboard's default valid-only filter (including sort-link state persistence), confidence intervals and the hardware-amortization cost model, opt-in job-engine execution (including a fake job engine in the test server, a router-run rejection, cleanup-after-completion, and a test proving both execution paths aggregate identically), a regression test for an OS-filtering bug in `available_backends()`, and a packaging regression test guarding the schema-symlink fix.

### Deep-dive against Lemonade's own source (not just its public docs)

Read the actual `lemonade-sdk/lemonade` C++ server source at a local checkout to find gaps the public API docs don't surface. Beyond the compute_engine/model_reload_free verification and the whispercpp/sd-cpp backend-selector bug above, this surfaced several capabilities LemonMatrix still doesn't use:
- **A server-side job engine (`POST /v1/jobs`) -- integrated, opt-in.** See "Implemented" above (`run_sweep_via_job`).
- **Embeddings and reranking -- done, CLI-first.** See the "Implemented" section above.
- **Image edits/variations -- done** (extended the existing `imagegen` pipeline; see "Implemented" above). **3D generation -- done** (own pipeline, `meshgen`; see "Implemented" above). **Image upscale deliberately not implemented** -- see "Implemented" above for why.
- Confirmed definitively (reading `metrics_linux.cpp`/`metrics_windows.cpp`/`metrics_macos.cpp` directly, not just observing live instances) that Lemonade's `/api/v1/system-stats` implementation has no power/watts field on any platform -- upgrades the existing "no instance checked so far reports power" note to "the feature doesn't exist in the server at all."
- vLLM (ROCm, experimental) was already correctly covered by `capabilities.py`'s backend-key-keyed compatibility table -- confirmed as a non-gap, no action needed.

### Partially implemented or inferred

- Prefill throughput is estimated as prompt tokens divided by TTFT because Lemonade does not expose a direct prefill metric.
- Memory is sampled after each completion rather than measured as a true high-water mark. Host RAM is used, with a validity note, when VRAM is unavailable.
- Backend versions are recorded when Lemonade reports an installed engine build. ROCm and driver data depend on what the server exposes.
- Power and energy fields are supported, but confirmed (by reading Lemonade's own `metrics_linux.cpp`/`metrics_windows.cpp`/`metrics_macos.cpp` directly, not just observing live instances) that no platform's implementation has a power/watts field at all -- the energy half of the cost model is therefore permanently None in practice, not "usually." Thermal data (`thermal_ok`, `peak_temp_c`) and AC/battery `power_state` are the same kind of permanent, structural gap: no temperature sensor and no AC/battery status are exposed anywhere in the server source either, so both remain user assertions with no possible server-side verification path -- this is a property of Lemonade's own API surface, not something LemonMatrix has left unfinished.
- The cost model's hardware-amortization half is now implemented (`hardware_cost_usd`/`hardware_lifetime_hours` inputs, spread over decode tokens/sec) alongside the pre-existing energy half -- since energy is always None in practice (above), this is what actually makes `cost_per_1k_tokens_usd` populate on a real run for the first time.
- `model.quantization` is now server-verified on a best-effort basis: `capabilities.parse_quantization()` (shared with the dashboard run form's pre-fill) parses Lemonade's own checkpoint string for the loaded model and flags a run whose claimed quantization contradicts it. Lemonade has no dedicated quantization/model-class/parameter-count field at all (confirmed against `model_info_to_json` in its own source) -- model class (dense/moe) and active parameter count remain permanently unverifiable the same way thermal/power_state are, since there's nothing to parse them from.
- Dispersion reporting now includes a 95% confidence interval (`ci95_half_width`, using the Student's t critical value for the actual trial count rather than a fixed 1.96) alongside the existing stddev/p95, and every model/router result now carries `prompt_sha256` so two runs can be confirmed to have used the identical prompt without embedding the prompt text itself.
- Profiles persist the environment fingerprint and connection details. Models and backends are discovered live rather than stored as part of the profile.
- GPU classification uses a VRAM-size heuristic because Lemonade does not explicitly distinguish integrated and discrete AMD GPUs; users can override the result.
- Cartesian-product sweeps are available in the dashboard as durable background batches; the CLI `sweep` command runs the same combinations synchronously but does not yet persist state through `SweepStore`.
- `exclusive_run` is server-verified where possible: a background thread polls the profiled instance's own `/api/v1/health` during warmup and measured trials, and a competing model reported `is_busy`/`is_streaming` overrides the caller's assertion to `False`. This is a spot check at the polling interval, not continuous monitoring, so a competing request that starts and finishes entirely between two polls is not caught; when every poll fails (or the run finishes before the first poll fires), verification falls back to the caller's own assertion with a note explaining that it could not be confirmed.
- The same health-polling mechanism now also server-verifies `compute_engine` and `model_reload_free`, closing two previously-flagged trust gaps. Confirmed against Lemonade's own source (`router.cpp`'s `get_all_loaded_models()`): every entry in `all_models_loaded[]` carries the *actual* `device` Lemonade used (a "cpu"/"gpu"/"npu" bitmask string -- it cannot distinguish integrated from discrete GPU) and `watchdog_reset` (true once Lemonade's own backend watchdog force-restarts the model's subprocess mid-run). A device that contradicts the claimed `compute_engine`, or a watchdog reset observed during measurement, now invalidates the run with a note explaining why. Router runs have no single physical device to verify and are exempt from the device check, same as the existing engine/backend compatibility check.
- Dashboard sweep and run forms do not yet expose a router run type selector (routing policy runs are CLI-first for now).
- 3D generation still has no benchmarking support at all -- needs image-input plumbing LemonMatrix doesn't have anywhere yet.
- The schema permits `docker` and `hybrid`, but OS discovery does not identify Docker and capability discovery does not produce the hybrid engine.
- The leaderboard's "Valid only" filter now defaults to on (a hidden marker in the filter form distinguishes "never filtered" from "explicitly cleared," so sort-link clicks correctly preserve whichever state was last chosen) -- invalid runs are hidden unless a user explicitly asks to see them, closing a previously-flagged gap.
- **Packaging bug found and fixed**: `schema/*.json` at the repo root used to be independently-maintained files, while `pyproject.toml`'s package-data has only ever pulled from `src/lemonmatrix/schema/`. Only `result.schema.json` had ever been manually copied there -- the five newer schemas were silently missing from every built wheel (confirmed by actually building one and installing it into a fresh venv: `validate_classify_result()` etc. would raise `FileNotFoundError`). Fixed by making the repo-root files symlinks into the package instead of independent copies, so there is exactly one file on disk per schema and this class of drift is now structurally impossible; a test asserts the symlink relationship holds.

### Missing or not yet trustworthy for ranking

- Memory type is defined by the schema and architecture but is not populated by discovery.
- `power_state` and `power_cap_w` are recorded but LemonMatrix does not change or verify the target machine's power source or power cap -- and, per the confirmed finding above, never can via Lemonade's API. This is disclosed as a permanent, structural limitation, not a pending TODO.
- Thermal validity is not measured; `thermal_ok` is currently always true and peak temperature is not collected -- same permanent limitation as `power_state`, confirmed absent from Lemonade's own metrics source.
- Model class (dense/moe) and active parameter count are not verified from server metadata for every run, and never can be -- Lemonade has no field for either. Quantization *is* now best-effort server-verified (see above).
- Sweep jobs live only in process memory when run via the CLI's `sweep` command; the dashboard's sweep batches are durable (`SweepStore`), but there is still no cancellation, retry policy, or cross-profile scheduling.
- The leaderboard is local only. There is no hosted multi-user submission flow, result signing/provenance, duplicate detection, moderation, export, or cross-machine synchronization.
- The project still needs challenge-facing materials: a polished methodology, reproducible benchmark protocol, representative AMD results, screenshots or a recorded demo, contribution guidance, and the final AMD member-site submission.
- The job-engine path (`run_sweep_via_job`, `--via-job-engine`) is opt-in only, model runs only (a router has no fixed backend/ctx_size for a job's load step), and per-combination rather than whole-batch -- `SweepBatch`/`SweepStore` remain the default durability mechanism and still own multi-combo batch tracking, deliberately left untouched to avoid regression risk to an already-working, tested path. Making job-engine execution the default, or expressing a whole multi-combo sweep as a single job, was not attempted.
- 3D generation, embeddings, and reranking dashboard pages don't exist yet (CLI-only, same as the other modalities initially).

### Recommended next milestones

1. Run and publish a small reproducible AMD hardware matrix covering at least two backends or quantizations.
2. Add a concise methodology and demo focused on the challenge criteria: usefulness to the local-AI community, technical rigor, and a clear Lemonade integration.
3. Decide whether job-engine execution should become the default (or stay opt-in) once it's had more real-world mileage.
4. Add embeddings/reranking as new benchmarkable modalities, following the classify/tts/stt/imagegen/audiogen precedent.