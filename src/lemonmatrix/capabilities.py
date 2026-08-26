"""What can actually be chosen when running a sweep against a profile.

Derived from the same /api/v1/system-info response discovery already reads:
which compute engines are physically present (from "devices"), and which
backends Lemonade can run on this instance (from "recipes"), filtered to
states that can actually run ("installed" or "installable" -- "unsupported"
never can) and to this host's OS. Backend strings here are Lemonade's own
"<recipe_key>-<backend_key>" (e.g. "llamacpp-vulkan") confirmed against a
real response -- NOT idea.md's example strings ("llama.cpp-vulkan"), since
the real recipe key is "llamacpp", with no dot, and there is no 1:1 mapping
back from the pretty example strings to arbitrary recipe keys in general.
backend_version.py's alias table is kept separately for those example
strings, since CLI users may still type them by hand.
"""

from __future__ import annotations

import re

from .backend_version import resolve_backend
from .profile import IGPU_VRAM_THRESHOLD_GB


def parse_quantization(model_id: str, checkpoint: str) -> str:
    """Best-effort quantization string for a model Lemonade already knows
    about, parsed from the checkpoint's ":VARIANT" suffix (the same
    convention /v1/pull uses, confirmed live) when present, else from a
    trailing GGUF-quant-looking segment of the model's own id (Q4_K_M, Q8_0,
    IQ4_XS, F16, BF16, ...) -- deliberately narrow, since a looser pattern
    would also match a non-quant suffix like "...-GGUF" itself and silently
    mislabel the file format as if it were the quantization.

    Neither path is guaranteed by the API, so this is best-effort, not
    ground truth -- used both for the dashboard run form's presentation-only
    pre-fill and, in bench.py, to flag (not silently override) a run whose
    claimed model.quantization contradicts what Lemonade's own checkpoint
    string implies. Returns "" if nothing could be parsed.
    """
    quant = checkpoint.rsplit(":", 1)[-1] if ":" in checkpoint else ""
    if not quant:
        match = re.search(r"-((?:Q\d|IQ\d|F16|F32|BF16)[A-Za-z0-9_]*)$", model_id)
        quant = match.group(1) if match else ""
    return quant

# Backend key (the segment after the last '-' in "recipe-key", e.g. "vulkan" from
# "llamacpp-vulkan") → compute engines that are physically consistent with that backend.
# "system" means llama.cpp's auto-detect; "hybrid" is explicitly multi-device.
_BACKEND_KEY_ENGINES: dict[str, frozenset[str]] = {
    "cpu":    frozenset({"cpu", "hybrid"}),
    "system": frozenset({"cpu", "igpu", "dgpu", "hybrid"}),
    "vulkan": frozenset({"igpu", "dgpu", "hybrid"}),
    "rocm":   frozenset({"igpu", "dgpu", "hybrid"}),
    "cuda":   frozenset({"dgpu", "hybrid"}),
    "metal":  frozenset({"igpu", "dgpu", "hybrid"}),
    "opencl": frozenset({"igpu", "dgpu", "hybrid"}),
    "npu":    frozenset({"npu", "hybrid"}),
}

# Human-readable description of what each backend key needs.
_BACKEND_KEY_DESCRIPTION: dict[str, str] = {
    "cpu":    "CPU-only execution — no GPU or NPU required",
    "system": "llama.cpp auto-select — works on any compute device",
    "vulkan": "Vulkan GPU compute — requires an iGPU or dGPU",
    "rocm":   "AMD ROCm GPU compute — requires an AMD iGPU or dGPU",
    "cuda":   "NVIDIA CUDA compute — requires a discrete NVIDIA GPU (dGPU)",
    "metal":  "Apple Metal GPU compute — requires an Apple GPU",
    "opencl": "OpenCL GPU compute — requires an iGPU or dGPU",
    "npu":    "AMD XDNA NPU compute — requires an AMD NPU",
}

# What to suggest when a user picks the wrong engine for a backend.
_BACKEND_KEY_ENGINE_SUGGESTIONS: dict[str, str] = {
    "vulkan": "use --engine igpu or --engine dgpu (Vulkan needs a GPU)",
    "rocm":   "use --engine igpu or --engine dgpu (ROCm is an AMD GPU backend)",
    "cuda":   "use --engine dgpu (CUDA requires a discrete NVIDIA GPU)",
    "metal":  "use --engine igpu or --engine dgpu (Metal needs an Apple GPU)",
    "opencl": "use --engine igpu or --engine dgpu (OpenCL needs a GPU)",
    "npu":    "use --engine npu (this backend runs on the AMD XDNA NPU only)",
    "cpu":    "use --engine cpu (this backend only runs on the CPU)",
}


def engine_backend_compatible(compute_engine: str, backend: str) -> tuple[bool, str]:
    """Return (True, "") or (False, human-readable reason + fix suggestion).

    A run declared as "cpu" with a GPU-only backend (rocm, vulkan, cuda) is a
    mislabel that produces invalid leaderboard entries — and will fail when
    Lemonade tries to load the model with the wrong device.  Catch it here so
    callers can abort or warn before a single trial is wasted.

    The backend string is the full "recipe-key" form (e.g. "llamacpp-rocm",
    "flm-npu").  We extract the backend key as the segment after the last
    hyphen and look it up in the table above.  Unknown backend keys (anything
    not in the table) are considered compatible to avoid blocking future
    Lemonade backends that haven't been enumerated yet.
    """
    _, _, backend_key = backend.rpartition("-")
    allowed = _BACKEND_KEY_ENGINES.get(backend_key)
    if allowed is None:
        return True, ""  # unknown key — don't block it
    if compute_engine not in allowed:
        desc = _BACKEND_KEY_DESCRIPTION.get(backend_key, f"requires {', '.join(sorted(allowed))}")
        suggestion = _BACKEND_KEY_ENGINE_SUGGESTIONS.get(backend_key, f"use one of: {', '.join(sorted(allowed))}")
        return False, (
            f"engine '{compute_engine}' + backend '{backend}' is an impossible combination.\n"
            f"  {backend_key}: {desc}.\n"
            f"  Fix: {suggestion}."
        )
    return True, ""


def compatible_engines_for_backend(backend: str) -> list[str]:
    """Return the compute engines that can legally pair with this backend string."""
    _, _, backend_key = backend.rpartition("-")
    allowed = _BACKEND_KEY_ENGINES.get(backend_key)
    return sorted(allowed) if allowed else []


def validate_combo_against_profile(
    compute_engine: str,
    backend: str,
    system_info: dict,
    environment: dict,
    modality: str | None = "Text generation",
) -> list[str]:
    """Check an engine/backend combo against the live profile and return a list of issues.

    Checks performed (all purely from Lemonade's API data — no OS sensing):
    1. Engine/backend key compatibility.
    2. Whether the chosen compute engine is actually present on the hardware.
    3. Whether the chosen backend is installed (not just installable) on the instance.

    `modality` must match the recipe's own declared modality (e.g. "Text
    classification" for onnxruntime, "Speech-to-text" for whispercpp) --
    defaulting to "Text generation" is only correct for LLM chat/router runs.
    Passing the wrong modality here would make check 3 silently compare
    against the wrong recipe family's installed/installable sets: a real
    onnxruntime-cpu install would show as "not found" because
    available_backends() filtered it out under the "Text generation"
    modality before this function ever saw it.
    """
    issues: list[str] = []

    # 1. Logical compatibility.
    ok, reason = engine_backend_compatible(compute_engine, backend)
    if not ok:
        issues.append(reason)

    # 2. Hardware presence.
    present_engines = available_compute_engines(system_info)
    if compute_engine not in ("hybrid", "router") and compute_engine not in present_engines:
        detected = ", ".join(present_engines) if present_engines else "none detected"
        issues.append(
            f"engine '{compute_engine}' is not present on this hardware "
            f"(detected: {detected})."
        )

    # 3. Backend installed state.
    # Normalise the backend string: alias form ("llama.cpp-vulkan") → canonical
    # form ("llamacpp-vulkan") so user-typed aliases compare correctly against
    # what Lemonade reports.
    resolved = resolve_backend(system_info, backend)
    canonical_backend = f"{resolved[0]}-{resolved[1]}" if resolved else backend

    installed_backends = {
        b["backend"]
        for b in available_backends(system_info, environment, modality=modality)
        if b["state"] == "installed"
    }
    installable_backends = {
        b["backend"]
        for b in available_backends(system_info, environment, modality=modality)
        if b["state"] == "installable"
    }
    if canonical_backend not in ("collection.router",) and installed_backends:
        # Only warn when we have live data (installed set non-empty means API call succeeded).
        if canonical_backend not in installed_backends:
            if canonical_backend in installable_backends:
                issues.append(
                    f"backend '{backend}' is not installed on this instance "
                    f"(it is installable — install it in Lemonade first)."
                )
            else:
                issues.append(
                    f"backend '{backend}' was not found on this instance "
                    f"(installed: {', '.join(sorted(installed_backends)) or 'none'})."
                )

    return issues


def host_os(environment: dict) -> str | None:
    os_version = (environment.get("os_version") or "").lower()
    if "linux" in os_version:
        return "linux"
    if "windows" in os_version:
        return "windows"
    if "darwin" in os_version or "macos" in os_version:
        return "macos"
    return None


def _as_list(value) -> list:
    if value is None:
        return []
    return [value] if isinstance(value, dict) else value


def available_compute_engines(system_info: dict) -> list[str]:
    """Which of cpu/igpu/dgpu/npu are physically present, in schema order."""
    devices = system_info.get("devices") if isinstance(system_info.get("devices"), dict) else system_info
    engines = []

    if (devices.get("cpu") or {}).get("available"):
        engines.append("cpu")

    amd_gpus = _as_list(devices.get("amd_gpu"))
    has_igpu = any(g.get("available") and (g.get("vram_gb") or 0) < IGPU_VRAM_THRESHOLD_GB for g in amd_gpus)
    has_amd_dgpu = any(g.get("available") and (g.get("vram_gb") or 0) >= IGPU_VRAM_THRESHOLD_GB for g in amd_gpus)
    has_nvidia_dgpu = any(g.get("available") for g in _as_list(devices.get("nvidia_gpu")))

    if has_igpu:
        engines.append("igpu")
    if has_amd_dgpu or has_nvidia_dgpu:
        engines.append("dgpu")

    npu_entries = _as_list(devices.get("amd_npu"))
    if any(n.get("available") for n in npu_entries):
        engines.append("npu")

    return engines


def available_backends(
    system_info: dict, environment: dict | None = None, modality: str | None = "Text generation"
) -> list[dict]:
    """Backends Lemonade could run here right now.

    Each entry: {backend, recipe, recipe_display, backend_key, state, version}.
    Excludes "unsupported" entries outright (can never run here) and, when
    the host OS is known, entries whose declared OS support list excludes it.

    Defaults to "Text generation" recipes only -- Lemonade also serves audio,
    image, speech, and 3D-generation recipes (acestep, kokoro, sd-cpp, ...),
    but LemonMatrix's schema (tokens_per_sec, prefill/decode) only makes sense
    for LLM inference. Pass modality=None to see every recipe regardless.
    """
    detected_os = host_os(environment or {})
    recipes = system_info.get("recipes") or {}
    results = []

    for recipe_key, recipe in recipes.items():
        if modality and recipe.get("modality") != modality:
            continue

        backends = recipe.get("backends") or {}
        # A backend key can appear in multiple `support` entries -- e.g.
        # onnxruntime's "cpu" backend has separate entries per device family
        # (x86_64 on windows/linux, arm64 on macos). Confirmed live against a
        # real instance: collapsing to one entry per key (keeping only the
        # last) silently dropped linux from onnxruntime's effective OS
        # support, since its *last* "cpu" entry in the list is macos-only.
        # Union the OS lists across every entry for a key instead.
        support_by_backend: dict[str, set[str]] = {}
        for s in recipe.get("support") or []:
            support_by_backend.setdefault(s.get("backend"), set()).update(s.get("os") or [])

        for backend_key, entry in backends.items():
            state = entry.get("state")
            if state not in ("installed", "installable"):
                continue

            allowed_os = support_by_backend.get(backend_key)
            if detected_os and allowed_os and detected_os not in allowed_os:
                continue

            results.append(
                {
                    "backend": f"{recipe_key}-{backend_key}",
                    "recipe": recipe_key,
                    "recipe_display": recipe.get("display_name", recipe_key),
                    "backend_key": backend_key,
                    "state": state,
                    "version": entry.get("version"),
                }
            )

    results.sort(key=lambda r: (r["recipe_display"], r["backend_key"]))
    return results


# ---------------------------------------------------------------------------
# Router model detection
# ---------------------------------------------------------------------------

def is_router_model(model: dict) -> bool:
    """True if this entry from /api/v1/models is a Lemonade collection.router.

    Lemonade registers routers with recipe "collection.router" (or a prefix
    of "collection").  The model id may also contain "collection." as a
    namespace prefix.  Both indicators are checked so that slight API
    variations across Lemonade versions are handled gracefully.
    """
    recipe = (model.get("recipe") or "").lower()
    model_id = (model.get("id") or "").lower()
    return recipe.startswith("collection") or model_id.startswith("collection.")


def available_routers(models: list[dict]) -> list[dict]:
    """Return the subset of /api/v1/models entries that are router models."""
    return [m for m in models if is_router_model(m)]
