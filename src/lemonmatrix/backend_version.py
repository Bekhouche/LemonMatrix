"""Looks up the installed engine build version for a run's --backend string.

Lemonade has no single "CUDA version" or "ROCm version" field (confirmed
against a real system-info response) -- what it has is a per-recipe,
per-backend entry under system_info["recipes"][recipe]["backends"][key],
each with its own "version" (e.g. llama.cpp's own build tag "b10397") and a
"state" of "installed" / "installable" / "unsupported". Only "installed"
means that version is what's actually running; "installable" just means it
could be downloaded, so it must not be reported as if it were active.

The mapping from a free-text --backend string (there is no enum for it in
the schema) to a (recipe, backend_key) pair is inherently a guess for
hand-typed strings -- BACKEND_ALIASES covers IDEA.md's example strings
("llama.cpp-vulkan") confirmed against the real recipes tree. capabilities.py
instead generates canonical "<recipe_key>-<backend_key>" strings directly
from Lemonade's own keys (e.g. "llamacpp-vulkan", no dot) for the dashboard's
backend dropdown -- recipe keys can contain hyphens themselves (e.g.
"ryzenai-llm", "sd-cpp"), but backend keys never do, so splitting on the
*last* hyphen recovers both correctly. Unrecognized strings return None
rather than guessing further.
"""

from __future__ import annotations

BACKEND_ALIASES: dict[str, tuple[str, str]] = {
    "llama.cpp-cpu": ("llamacpp", "cpu"),
    "llama.cpp-vulkan": ("llamacpp", "vulkan"),
    "llama.cpp-rocm": ("llamacpp", "rocm"),
    "llama.cpp-cuda": ("llamacpp", "cuda"),
    "llama.cpp-metal": ("llamacpp", "metal"),
    "llama.cpp-system": ("llamacpp", "system"),
    "fastflowlm-npu": ("flm", "npu"),
    "ryzenai-npu": ("ryzenai-llm", "npu"),
}


def resolve_backend(system_info: dict, backend: str) -> tuple[str, str] | None:
    """Split a --backend string into (recipe_key, backend_key), or None if
    unrecognized. Used both to look up the installed engine version here and,
    in bench.py, to actually tell Lemonade which backend to load (the
    "<recipe>_backend" parameter on POST /api/v1/load) -- without this, a
    load call has no backend selector at all and Lemonade silently falls
    back to the recipe's configured default_backend, which may not be
    installed on this host even when the one the user picked is.
    """
    if backend in BACKEND_ALIASES:
        return BACKEND_ALIASES[backend]

    # Canonical "<recipe_key>-<backend_key>" form: split on the last hyphen
    # and confirm the recipe_key half actually exists, since backend keys
    # never contain hyphens but some recipe keys do.
    recipe_key, _, backend_key = backend.rpartition("-")
    if recipe_key and recipe_key in (system_info.get("recipes") or {}):
        return recipe_key, backend_key
    return None


def lookup_backend_version(system_info: dict, backend: str) -> str | None:
    """The installed engine build version for `backend`, or None if unknown
    or not actually installed (state != "installed")."""
    resolved = resolve_backend(system_info, backend)
    if resolved is None:
        return None
    recipe_key, backend_key = resolved

    recipe = (system_info.get("recipes") or {}).get(recipe_key) or {}
    entry = (recipe.get("backends") or {}).get(backend_key) or {}
    if entry.get("state") != "installed":
        return None
    return entry.get("version")
