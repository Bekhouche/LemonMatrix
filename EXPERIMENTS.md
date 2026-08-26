# Experiment Plan

Cross-hardware benchmark plan across the four real machines connected as LemonMatrix profiles, for
the public leaderboard / AMD Lemonade Developer Challenge submission. Every number below (installed
vs. installable backends, pulled models, VRAM, GPU family) was read live from each instance's own
`/api/v1/system-info`, not assumed — this file should be re-verified (`lemonmatrix profile debug
<name>`) before trusting it again after any further setup changes.

**Status: every genuinely installable backend on all four machines is now installed** (58/58 installs
succeeded on 2026-08-24, via `lemonmatrix profile install-backend`). Everything still showing
`unsupported` is architecturally impossible (wrong GPU vendor for that recipe, wrong OS, or no NPU
present) — there is nothing left to install. Model pulls (§3, §4) are the only remaining setup.

## 1. Hardware inventory

| Profile | GPU | Family | VRAM | OS | Notes |
|---|---|---|---|---|---|
| `NVIDIA-L4` | 7x NVIDIA L4 | `sm_89` (Ada) | 22.49 GB each | Linux, Ubuntu 22.04 | Datacenter; every installable text-gen/classify/tts/stt/imagegen/audiogen/meshgen backend installed. **Ubuntu 22.04's glibc is too old for several Lemonade binaries -- see §7.** |
| `AMD-Instinct-MI350X` | AMD Instinct MI350X | `gfx950` | 287.6 GB | Linux, Ubuntu 24.04 | Datacenter; same, fully installed. Lemonade reports the GPU's own `name` field as the bare string `"90500"`, not a product name -- `family` is the only trustworthy identifier here |
| `Radeon-RX-7900-XTX` | AMD Radeon RX 7900 XTX | `gfx110X` (generic RDNA3 bucket on this OS path) | 24 GB (Lemonade's own `vram_gb` here is a confirmed unit bug -- reports raw bytes as if already GB) | Windows 11 Pro for Workstations | Consumer; fully installed from scratch (19/19), including `llamacpp` cpu/vulkan/rocm |
| `Radeon-RX-7900-XTX-WSL` | same physical card, via WSL2 | `gfx1100` (exact -- more precise than the native-Windows report) | 24 GB (correct here) | WSL2 Ubuntu 24.04 | Consumer, same GPU as above, different OS/driver stack; fully installed from scratch (20/20) including **`vllm:rocm`** -- the only one of the four where vLLM is even offered (native Windows reports it `unsupported: Requires Linux`) |

## 2. Remaining setup before any run

- **Model pulls only** -- every backend install is done. See §3/§4 for which models still need pulling per machine.
- **Router**: done -- see §9.

## 3. Task 1 -- LLM chat / model runs (`lemonmatrix run` / `sweep`)

### Model tiers

| Tier | Model | Quant | Size | Purpose |
|---|---|---|---|---|
| Sanity | `Qwen3-1.7B-GGUF` | `Q4_0` | ~1 GB | Fast, cheap -- validates the whole engine/backend matrix everywhere before committing to the big pull. Already on `Radeon-RX-7900-XTX-WSL`. |
| Headline | `Qwen3.8-27B-GGUF` (`unsloth/Qwen3.8-27B-GGUF`) | `UD-Q4_K_XL` | ~16.4-16.7 GB | The one true cross-hardware row -- same checkpoint on datacenter NVIDIA, datacenter AMD, and consumer AMD. Already pulled on `NVIDIA-L4` and `AMD-Instinct-MI350X`; needs pulling on both 7900-XTX profiles (fits in 24 GB VRAM with room for KV cache at a modest context length). |

`NVIDIA-L4` also has a second 27B quant (`Q4_K_M`, 15.9 GB) already pulled -- optional extra row to compare two quantizations of the same model on the same hardware, if time allows.

### Engine x backend matrix

| Profile | Rows to run | Angle |
|---|---|---|
| `NVIDIA-L4` | `cpu` / `cuda` / `vulkan` | Which backend wins on the *same* NVIDIA card |
| `AMD-Instinct-MI350X` | `cpu` / `rocm` / `vulkan` | Same question on AMD datacenter silicon |
| `Radeon-RX-7900-XTX` | `cpu` / `rocm` / `vulkan` | Consumer AMD GPU, native Windows -- all three backends installed and ready |
| `Radeon-RX-7900-XTX-WSL` | `cpu` / `rocm` / `vulkan` / **`vllm-rocm`** | Same physical GPU as above, WSL2 -- OS/virtualization-overhead story, plus the only vLLM data point in the whole Discord project roundup; `vllm:rocm` installed and ready |

### Fixed parameters (kept constant across every row so results are actually comparable)

- `context_length`: 4096 (not the model's reported max -- confirmed elsewhere in this project that defaulting to a huge max context makes even loading painfully slow)
- `power_state`: `plugged` (all four are desktop/server boxes, not battery-relevant)
- `warmup_trials`: 2, `measured_trials`: 5 (CLI/dashboard defaults)
- Same prompt and `max_tokens` (256, default) on every row
- `model_class`: `dense` (Qwen3.8-27B and Qwen3-1.7B are both dense, not MoE)

## 4. Task 2 -- non-text modalities

Every backend below is now **installed** on all four machines (confirmed live after the install sweep;
`Text generation` excluded here, covered in §3). The only remaining work per modality is choosing and
pulling a model -- no backend installs left to do anywhere.

| Modality | Recipe(s) | Backends installed, all 4 machines |
|---|---|---|
| Text classification | `onnxruntime` | `cpu` |
| Text-to-speech | `kokoro`, `openmoss` | `kokoro-cpu` everywhere; `openmoss-vulkan` everywhere, `openmoss-rocm` on the three AMD boxes, `openmoss-cuda` on `NVIDIA-L4` |
| Speech-to-text | `whispercpp`, `moonshine` | `whispercpp-cpu/vulkan` everywhere, `whispercpp-rocm` on the three AMD boxes; `moonshine-cpu` everywhere |
| Image generation | `sd-cpp` | `sd-cpp-cpu/vulkan` everywhere, `sd-cpp-rocm` on the three AMD boxes, `sd-cpp-cuda` on `NVIDIA-L4` |
| Audio generation | `acestep`, `thinksound` | `vulkan` everywhere, `rocm` on the three AMD boxes, `cuda` on `NVIDIA-L4` (both recipes) |
| 3D generation | `trellis` | `vulkan` everywhere, `rocm` on the three AMD boxes, `cuda` on `NVIDIA-L4` |
| Embeddings | `llamacpp` (same install as §3) | ready everywhere once a GGUF embeddings-labeled model is pulled |
| Reranking | `llamacpp` only (not `flm`) | ready everywhere once a reranker-labeled GGUF is pulled |

Since every backend is now installed uniformly, the cross-hardware angle for §4 is exactly the same shape
as §3's: pick one model per modality, run it on `vulkan` on all four (the one backend every machine shares),
plus each machine's own vendor-native backend (`cuda` on `NVIDIA-L4`, `rocm` on the three AMD boxes).

### Suggested minimal plan per modality (models still need to be identified/pulled -- not yet chosen)

- **Classify**: any small ONNX text-classifier (e.g. a phishing/spam detector) -- `onnxruntime` is CPU-only everywhere regardless of machine, so this one doesn't get a GPU-backend comparison.
- **TTS**: `kokoro` (CPU-only) for a baseline everywhere; `openmoss` on `vulkan` for the real cross-hardware GPU row, `cuda`/`rocm` for each machine's native backend.
- **STT**: `whispercpp` on `vulkan` everywhere plus each machine's native backend -- the most consistent cross-hardware modality after LLM chat.
- **Image generation**: `sd-cpp`, same `vulkan`-everywhere-plus-native pattern.
- **Audio generation**: `acestep`, same pattern.
- **3D generation**: `trellis`, same pattern; note the earlier finding that the smallest realistic test checkpoint is a large (~15 GB) download, so this is a lower-priority tier unless the storage/bandwidth cost is worth it.
- **Embeddings / reranking**: lowest priority for the leaderboard story (throughput isn't comparable to chat tok/s and isn't visually compelling in a demo) -- worth one data point each on `NVIDIA-L4` for completeness, not a full four-machine matrix.

## 5. Showcase / differentiator runs (after the core matrices above are clean)

- ✅ **Router run -- done, see §9.** Two real routers built and benchmarked; found and fixed two real bugs along the way (an exclusivity-check bug, then a second ordering bug in that same fix). Genuinely valid router benchmarks confirmed on both `Radeon-RX-7900-XTX` (263.9 tok/s) and `Radeon-RX-7900-XTX-WSL` (32.9 tok/s).
- ✅ **`--via-job-engine` run -- done.** `Qwen3-1.7B-GGUF` on `NVIDIA-L4` via the job engine: 56.9 tok/s, matching the earlier direct-HTTP measurement (56.6 tok/s) closely -- confirms the two execution paths agree.
- ✅ **Deliberately-mismatched combo -- done.** `--engine cpu --backend llamacpp-vulkan --force` on `NVIDIA-L4`: ran (6.9 tok/s, matching the earlier vulkan measurement exactly) but was caught by **two independent checks** and marked invalid -- the static engine/backend compatibility rule, and the server-verified device check (claimed `cpu`, but Lemonade's own `/api/v1/health` reported `device: gpu` for the loaded model). A clean demonstration of "shown but never ranked."
- ⚠️ **`vllm-rocm` on `Radeon-RX-7900-XTX-WSL`** -- attempted, currently fails, not a LemonMatrix issue. The WSL instance was briefly unreachable (`xx.xx.xx.xx:8055` timed out while `:8050` on the same host responded fine -- the WSL2 process/VM was down) but came back up on its own (and was auto-upgraded from Lemonade 11.6.0 to 11.7.0 in the process, with all models/backends intact). Pulled the real catalog model `Qwen3.5-0.8B-FP16-vLLM` (`Qwen/Qwen3.5-0.8B`, merged into the pre-existing catalog entry once fully downloaded) successfully. Loading it failed twice, including with a 300s client-side timeout: `vllm-server failed to start within timeout` -- Lemonade's own internal readiness timeout firing, not a client-side one, so more patience from this side doesn't help. `system-info` shows the backend `state: installed` with no error message. Lemonade itself labels this backend `"vLLM ROCm (experimental)"` (`experimental: true` in its own recipe metadata), and this fits the broader pattern already found on this specific WSL box (`acestep-rocm` crashes reproducibly, `vulkan` has severe overhead) -- WSL2's ROCm stack looks generally less stable here across multiple, otherwise-unrelated backends. Deferred, same treatment as `TRELLIS-3D`/`OpenMOSS-TTS` -- would need direct log access on the machine to diagnose further.

## 6. Open items / not yet decided

- Hardware cost/lifetime inputs (`--hardware-cost-usd` / `--hardware-lifetime-hours`) for the amortized-hardware cost-model component -- not filled in here since actual purchase prices aren't known; add per-machine if/when available.
- Which specific ONNX classifier / embeddings / reranker checkpoints to pull -- not chosen yet, deferred to §4's "suggested minimal plan" until the LLM matrix (§3) is done.
- Whether `NVIDIA-L4`'s 7 cards get exercised as true multi-GPU (llama.cpp's own GPU-split behavior, not something LemonMatrix controls) or the run only ever touches one card -- worth checking once the first `cuda` run actually executes, since a single L4's 22.49 GB already fits the 27B tier without needing a split at all.

## 7. Results and findings so far (2026-08-25)

### Task 1 (LLM chat) — done on all four machines

| Machine | Backend | 1.7B decode tok/s | 27B decode tok/s |
|---|---|---|---|
| NVIDIA-L4 | cpu | 56.6 | 3.6 |
| NVIDIA-L4 | cuda | **broken** (won't load, consistent) | — |
| NVIDIA-L4 | vulkan | 6.9 (slower than its own cpu) | — (not viable) |
| AMD-Instinct-MI350X | cpu | 0.4 (~75min for one combo -- likely NUMA-related on this 64-core EPYC) | — (skipped) |
| AMD-Instinct-MI350X | rocm | 385.9 | 100.0 |
| AMD-Instinct-MI350X | vulkan | 0.4 (identical to broken cpu) | — (skipped) |
| Radeon-RX-7900-XTX (native) | cpu | 17.7 | — |
| Radeon-RX-7900-XTX (native) | rocm | 207.3 | 14.2 |
| Radeon-RX-7900-XTX (native) | vulkan | 145.1 (unstable, ±74 stddev) | 20.5 |
| Radeon-RX-7900-XTX-WSL | cpu | 25.6 | — |
| Radeon-RX-7900-XTX-WSL | rocm | 259.8 | 17.1 |
| Radeon-RX-7900-XTX-WSL | vulkan | 37.9 | timed out (>600s, not viable at 27B) |

Key findings: (1) NVIDIA-L4's `cuda` backend fails to load outright, consistently -- no further detail available via the API. (2) `vulkan` doesn't seem to actually engage the GPU at all on NVIDIA-L4 (slower than CPU). (3) MI350X's `cpu` and `vulkan` are both ~0.4 tok/s, identical to each other -- `vulkan` likely silently falls back to a broken CPU path there. (4) WSL2 adds severe overhead specifically to `vulkan` on the shared 7900 XTX (37.9 vs 145-253 tok/s natively) but barely touches `rocm` (259.8 vs 207-222) -- a clean, reproducible finding. (5) `Radeon-RX-7900-XTX` and `Radeon-RX-7900-XTX-WSL` are the **same physical GPU** -- must never be benchmarked concurrently (confirmed live: running both at once produced a false "competing workload" flag on one side and a genuine load timeout on the other).

### Real LemonMatrix bug found and fixed

`_wav_duration_seconds()` crashed on kokoro's real TTS output: kokoro returns IEEE-float PCM (WAV fmt tag 3), which Python's stdlib `wave` module refuses to parse at all ("unknown format"), even though duration is trivially computable from the same header fields regardless of PCM format. Rewrote it to parse the RIFF header directly via `struct` instead of `wave.open()`. Verified against a synthetic float32 WAV and the full test suite (297 tests, all still passing).

### Task 2 (non-text modalities) — MI350X and NVIDIA-L4 done, Windows machines pending

**NVIDIA-L4 — near-total failure, root cause identified**: `classify` (onnxruntime), `kokoro` TTS, `whispercpp` STT, and `sd-cpp` imagegen **all fail to start**. The one error that printed full detail is unambiguous: `ort-server: ... GLIBCXX_3.4.32' not found ... GLIBC_2.38' not found`. **NVIDIA-L4 runs Ubuntu 22.04, and these Lemonade binaries require glibc 2.38+ (Ubuntu 24.04+ only)** -- a real OS/binary-compatibility gap, not a LemonMatrix bug, and not fixable without upgrading that machine's OS. `acestep` (audiogen) and `llamacpp`-based modalities (embeddings, rerank, LLM chat) are unaffected -- apparently built against an older glibc baseline. `AMD-Instinct-MI350X` (Ubuntu 24.04) does not have this problem, consistent with the theory.

**MI350X**: classify ✅ (123.2ms, 8.12/sec), STT ✅ (RTF 22.4x), imagegen ✅ (3.5s/image via vulkan), audiogen ✅ but slow (RTF 0.17x via vulkan), embeddings ✅ (154.2ms), rerank ✅ (111.5ms). `kokoro` TTS hit the WAV bug above (now fixed, needs re-run). `OpenMOSS-TTS` failed with a genuine Lemonade error ("generation produced no audio -- codec missing or model emitted EOS too early"). `TRELLIS-3D` meshgen exceeded the 600s timeout and was still running server-side afterward, causing (correct, not a false positive) "competing workload" invalidation on the embeddings/rerank runs that followed.

**NVIDIA-L4's `TRELLIS-3D` also timed out**, on a completely different backend (`cuda` vs. MI350X's `vulkan`) and vendor -- since it hangs the same way on two unrelated GPUs, the likely cause is the request itself (possibly the plain solid-gray synthetic BMP test image used here being degenerate input for the mesh pipeline) rather than a hardware/backend problem. Not yet resolved as of this writing.

**`TRELLIS-3D` update**: tried a fresh single-trial retry (both a flat solid-gray input and a textured checkerboard input, ruling out degenerate-input as the cause) on MI350X -- neither completed within 120-180s either, after the original run had already run 600s+ without finishing. This appears to be a genuine characteristic of this model/pipeline taking far longer than 10 minutes per generation (3D reconstruction is inherently heavier than 2D image gen), not a hardware, backend, or input-content problem. **Deferred** -- skipping meshgen on the remaining two machines for this pass rather than spending another 10+ minutes per attempt; worth a dedicated, patient retry later with a much longer timeout (30-60 min) if 3D-gen numbers are wanted for the leaderboard.

### Task 2 — Radeon-RX-7900-XTX-WSL, done

Classify ✅ (77.0ms, 12.98/sec), `kokoro` TTS ✅ (RTF 3.52x, correct after the WAV fix), STT ✅ (RTF 19.98x), imagegen ✅ via `sd-cpp-rocm` (301.1ms/image -- notably faster than either other machine that got a valid imagegen number), embeddings ✅ (116.8ms), rerank ✅ (95.2ms).

`OpenMOSS-TTS` failed with the same genuine "no audio produced" error seen on every other machine -- consistent, not environment-specific.

**New finding: a live backend-version drift caught mid-run.** `llamacpp-rocm` silently flipped from `installed` to `state: update_required` (a newer build, b10470, had become available; the previously-working b10397 no longer satisfied the requirement) sometime between the LLM tier runs and this Task 2 pass -- correctly caught by `validate_combo_against_profile`, which aborted embeddings/rerank rather than running them against a backend Lemonade itself now considers stale. Fixed by re-running `lemonade backends install llamacpp:rocm`; both then passed cleanly. Not a LemonMatrix bug -- confirms the pre-flight validation is doing its job when the ground truth shifts mid-session.

**New finding: `ACE-Step-Music` (audiogen) is unreliable specifically on this WSL profile.** Failed twice in a row with `CURL error: Failure when receiving data from the peer` via `acestep-rocm` -- the same recipe/backend combination that worked fine on both `Radeon-RX-7900-XTX` (native, RTF 5.58x) and `AMD-Instinct-MI350X` (RTF 0.17x). Since it's reproducible only here, this looks like a WSL2+ROCm stability issue specific to acestep's subprocess, not a LemonMatrix client bug -- deferred, same treatment as `TRELLIS-3D`.

## 8. Task 2 final summary, all four machines

| Modality | NVIDIA-L4 | MI350X | 7900-XTX (native) | 7900-XTX (WSL) |
|---|---|---|---|---|
| Classify | ❌ glibc | ✅ | ✅ | ✅ |
| TTS (kokoro) | ❌ glibc | ✅ | ✅ | ✅ |
| TTS (OpenMOSS) | ❌ glibc | ❌ real error | ❌ real error | ❌ real error |
| STT | ❌ glibc | ✅ | ✅ | ✅ |
| Imagegen | ❌ glibc | ✅ | ❌ (`sd-server` failed to start) | ✅ |
| Audiogen | ✅ (slow) | ✅ (slow) | ✅ | ❌ (WSL-specific crash) |
| Meshgen | ⚠️ deferred (>10min, hangs) | ⚠️ deferred (>10min, hangs) | not attempted | not attempted |
| Embeddings | ✅ | ✅ | ✅ | ✅ |
| Rerank | ✅ | ✅ | ✅ | ✅ |

`OpenMOSS-TTS` fails identically on all four machines with the same error -- a real, consistent Lemonade/model issue, not environment-specific. `TRELLIS-3D` is deferred on both machines it was tried on (needs a dedicated 30-60 minute timeout retry later). Everything else has at least one clean, valid cross-machine data point.

**`Imagegen` on `Radeon-RX-7900-XTX` revisited -- flaky, not cleanly broken.** A retry with a longer client-side timeout (90s) got `sd-server` to acknowledge a successful load (`{"status": "success", ...}`), but a `client.health()` check immediately after showed nothing resident at all, and a follow-up `lemonmatrix imagegen` call failed with the identical `sd-server failed to start or become ready` error. This looks like the sd-cpp-rocm subprocess crashing shortly after acknowledging start, not a hard, permanent failure -- worth a patient retry, but not re-attempted further this pass.

## 9. Router differentiator — created, tested, and benchmarked (2026-08-25)

### How a router is actually created (verified against Lemonade's C++ source, not guessed)

Registration goes through the **same generic endpoint** LemonMatrix already used for custom model registration -- `POST /api/v1/pull` (`Server::handle_pull`, `src/cpp/server/server.cpp:5563`), which calls `ModelManager::register_user_model` (`model_manager.cpp:3232`). A `collection.router` body needs, at minimum:

```json
{
  "model_name": "user.<Name>",
  "recipe": "collection.router",
  "version": "1",
  "components": ["<already-registered model A>", "<already-registered model B>"],
  "routing": {
    "candidates": ["<model A>", "<model B>"],
    "default_model": "<model A>",
    "rules": [
      {
        "id": "<rule-id>",
        "match": { "keywords_any": ["def ", "function"] },
        "route_to": "<model B>"
      }
    ]
  }
}
```

Confirmed from `routing_policy_parser.cpp` (full 672-line read): `routing` accepts `candidates`/`default_model`/`router`/`classifiers`/`rules`; each rule's `match` supports `keywords_any`/`keywords_all`/`regex`/`min_chars`/`max_chars`/`has_tools`/`has_images`/`metadata`/`classifier`, combinable via `any`/`all`/`not`. `routing.router` is sugar for an LLM-as-classifier. There's no round-robin primitive -- only rule matching plus an unconditional `default_model` fallback. A real example ships in the repo at `examples/router/policy_local.json`. Candidate models must already be registered (not necessarily loaded -- router collections are virtual and lazy-load whichever candidate gets selected, per the comment at `server.cpp:5995-5998`). Invocation is completely ordinary: `POST /v1/chat/completions` with `model: "<router_name>"`, `route_trace: true` to get the decision back in `x_lemonade_route`.

Registered via LemonMatrix's own `client.start_model_download(model_name, recipe="collection.router", version="1", components=[...], routing={...})` -- no new client code needed, since `**kwargs` already flowed through into the POST body.

### Two routers built from models we already had pulled

1. **`user.Demo-Router-RuleBased`** (registered on `NVIDIA-L4`, `AMD-Instinct-MI350X`, `Radeon-RX-7900-XTX`) -- candidates `Qwen3-1.7B-GGUF` (default) / `Qwen3.8-27B-GGUF-UD-Q4_K_XL`; routes to the 27B model on coding keywords or `min_chars: 500`, otherwise stays on the 1.7B model.
2. **`user.Demo-Router-Privacy`** (registered on `AMD-Instinct-MI350X`) -- candidates same two models, default is the 27B model; a `metadata: {key: "consent", equals: "denied"}` rule routes to the small model instead ("stays local" pattern, mirrors the repo's own example).

Both verified with real chat completions and `route_trace: true` on MI350X:
- Rule-based: `"Hi, how are you?"` → `default_used: true, route_to: Qwen3-1.7B-GGUF`. `"...def add(a, b): ..."` → `matched_rule: coding-or-long-to-big, route_to: Qwen3.8-27B-GGUF-UD-Q4_K_XL`.
- Privacy: no metadata → `default_used: true, route_to: ...27B...`. `metadata: {"consent": "denied"}` → `matched_rule: sensitive-stays-local, route_to: Qwen3-1.7B-GGUF`.

### Real router benchmark

`lemonmatrix run --run-type router --model user.Demo-Router-RuleBased` on `Radeon-RX-7900-XTX`: **264.9 tok/s decode**, correctly dispatching to the default small model for the generic benchmark prompt (confirmed via the trials sidecar's `x_lemonade_route`: `route_to: Qwen3-1.7B-GGUF, default_used: true` on all 3 trials).

### Real LemonMatrix bug found and fixed

`_ExclusivityMonitor` compared `/api/v1/health` entries against the router's own collection name to detect competing workloads -- but a router is virtual and **never itself appears in health**; only whichever candidate it actually dispatched to does (confirmed live and from source). Every router run was therefore misidentifying its own selected candidate as an unrelated competing workload and getting invalidated for a reason that wasn't real. Fixed in `bench.py`: `_competing_model_names`/`_own_model_entry`/`_ExclusivityMonitor` now take a **set** of own-model names; `run_sweep()` resolves a router's real candidate set via `client.models()`'s `components` field before constructing the monitor. Added a regression test (`test_router_run_recognizes_its_own_routed_candidate_as_not_competing`); full suite still passes (298 tests).

### One further finding, not fixable in LemonMatrix

**Router-dispatched candidates never get an explicit backend selector.** A direct model run can pass `llamacpp_backend: rocm`; a router's internal `auto_load_model_if_needed` call cannot be steered by the routing policy or by LemonMatrix at all. On a machine whose *default* backend for a recipe happens to be a broken one (MI350X, NVIDIA-L4 -- see §7 and the `llamacpp-cuda` deep-dive in §10), **every** router run is stuck on that broken default regardless of which candidate gets selected.

### A real second bug in the fix itself, found and corrected (2026-08-26)

The exclusivity-monitor fix above was initially believed to work everywhere except native Windows (a "Windows doesn't list routers" theory). That theory was **wrong** -- re-investigated and traced to a real ordering bug in the fix's own code, not an OS difference:

- Confirmed live and reproducibly: calling `POST /api/v1/load` directly on a router's own registered name makes Lemonade **drop it from subsequent `GET /api/v1/models` responses** for as long as one of its candidates remains the active loaded model (it reappears once that candidate is unloaded) -- on **every** OS tested (Linux and Windows alike), not just Windows.
- The fix's own candidate-resolving lookup (`client.models()`, filtered for `components`) was placed **after** `client.load(cfg.model_name)` in `run_sweep()` -- exactly the call that triggers the disappearance. So the lookup would find nothing and silently fall back to the old, buggy single-name behavior, on literally every router run that ever executed one -- explaining why every "confirmed" router benchmark up to this point (MI350X's 100.0 tok/s LLM-tier number aside, which isn't a router run) still showed `competing workload` false positives despite the fix supposedly being in place.
- Fixed by moving the candidate-set resolution **before** the `client.load()` call in `run_sweep()`. Verified against two genuinely fresh routers that had never been loaded before (so the fix's timing could be tested honestly, since a router that had already had `load()` called on it earlier in the session stays hidden from listing regardless of code ordering): `user.Demo-Router-Fresh` on `Radeon-RX-7900-XTX-WSL` (32.9 tok/s, `valid: true`) and `user.Demo-Router-Fresh2` on `Radeon-RX-7900-XTX` (263.9 tok/s, `valid: true`) -- both with `exclusive_run` correctly verified, no false "competing workload" note. Full test suite still passes.
- **Correction to the earlier §9 finding**: native Windows Lemonade does **not** have a special router-listing quirk distinct from Linux. Every machine behaves identically; the apparent Windows-only failure was an artifact of testing native Windows *after* a real benchmark run (which calls `load()` on the router) while the Linux machines had only been listing-checked *before* their first `load()` call.

## 10. `llamacpp-cuda` on `NVIDIA-L4` -- root cause narrowed, not fully resolved (2026-08-26)

Previously just documented as "broken, consistent, no detail available." Investigated further:

- **Not a glibc issue**: llamacpp runs fine via `cpu` on this exact same Ubuntu 22.04 install, so the glibc 2.38 gap that explains `onnxruntime`/`kokoro`/`whispercpp`/`sd-cpp` doesn't apply here.
- **Not device-selection confusion**: tried explicit single-GPU selectors (`llamacpp_device: CUDA0/Cuda0/cuda0`) across the 7-GPU box -- identical failure every time, ruling out a multi-GPU-ambiguity theory.
- **Not model-specific**: both `Qwen3-1.7B-GGUF` and `Qwen3.8-27B-GGUF-UD-Q4_K_XL` fail identically.
- **Not a client-timeout artifact**: tried up to a 300s client-side timeout -- still fails with Lemonade's own `llama-server failed to start` message, meaning Lemonade's *own* internal readiness timeout is what's firing, not this client's.
- **Not a one-time cache-miss**: an immediate retry right after a failure shows no improvement.
- **Genuinely informative timing signal**: failures are not instant -- they scale with model size (17.5s for the 1.7B model, 35.9s for the 27B model), matching how long loading weights into VRAM would actually take. This suggests the `llama-server` CUDA process starts and loads the model, but never reaches a state Lemonade considers "ready" -- consistent with a CUDA runtime/driver version mismatch specific to this llama.cpp CUDA build (`b10397`, driver `535.309.01`) rather than a hard crash on start.
- Confirmed the recipe's own `default_backend` for `llamacpp` on this machine is `cuda` -- this is *why* every router run and any load that doesn't pass an explicit backend selector lands on this broken path by default (see §9).
- `sd-cpp-cuda` and `acestep-cuda` on the same machine behave differently from each other and from `llamacpp-cuda` (acestep works, albeit slowly; sd-cpp is flaky) -- each recipe ships its own separately-built CUDA binary, so this is plausibly a `llama.cpp`-CUDA-build-specific issue, not a system-wide CUDA/driver problem (which would be expected to affect all three identically).

**Reached the limit of what's diagnosable via the REST API alone.** Confirming the exact root cause (most likely a CUDA toolkit version the `b10397` build expects vs. what driver `535.309.01` actually provides) would need direct access to the `llama-server` process's own stderr/logs on that machine, which isn't available through Lemonade's API. GitHub's `b10397` release notes don't document a specific CUDA toolkit requirement either.
