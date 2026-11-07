"""Verified against the real recipes/devices tree from a live Lemonade 11.6.0
instance (Ubuntu 22.04, 7x NVIDIA L4, no AMD hardware) -- see
test_profile_real_shape.py for the matching discovery-side fixture."""

from lemonmatrix.capabilities import available_backends, available_compute_engines

SYSTEM_INFO = {
    "devices": {
        "amd_gpu": [],
        "amd_npu": {"available": False, "name": "Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz"},
        "cpu": {"available": True, "name": "Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz"},
        "nvidia_gpu": [{"available": True, "name": "NVIDIA L4", "vram_gb": 22.49} for _ in range(7)],
    },
    "recipes": {
        "llamacpp": {
            "display_name": "Llama.cpp GPU",
            "modality": "Text generation",
            "backends": {
                "cpu": {"state": "installable", "version": "b10375"},
                "cuda": {"state": "installable", "version": "b10397"},
                "vulkan": {"state": "installable", "version": "b10375"},
                "rocm": {"state": "unsupported"},
            },
            "support": [
                {"backend": "cpu", "os": ["linux", "windows"]},
                {"backend": "cuda", "os": ["linux", "windows"]},
                {"backend": "vulkan", "os": ["linux", "windows"]},
            ],
        },
        "flm": {
            "display_name": "FastFlowLM NPU",
            "modality": "Text generation",
            "backends": {"npu": {"state": "unsupported"}},
            "support": [{"backend": "npu", "os": ["windows"]}],
        },
        "kokoro": {
            "display_name": "Kokoro",
            "modality": "Text-to-speech",
            "backends": {"cpu": {"state": "installable", "version": "b17"}},
            "support": [{"backend": "cpu", "os": ["linux", "windows"]}],
        },
    },
}

ENVIRONMENT = {"os_version": "Linux-6.8.0-117-generic (Ubuntu 22.04)"}


def test_available_compute_engines_reports_cpu_and_dgpu_only():
    assert available_compute_engines(SYSTEM_INFO) == ["cpu", "dgpu"]


def test_available_backends_defaults_to_text_generation_only():
    backends = available_backends(SYSTEM_INFO, ENVIRONMENT)
    assert {b["backend"] for b in backends} == {"llamacpp-cpu", "llamacpp-cuda", "llamacpp-vulkan"}


def test_available_backends_excludes_unsupported_state():
    backends = available_backends(SYSTEM_INFO, ENVIRONMENT)
    assert "flm-npu" not in {b["backend"] for b in backends}


def test_available_backends_excludes_wrong_os():
    # flm-npu is unsupported anyway; force it "installable" to isolate the
    # OS filter specifically (its support list only claims windows).
    system_info = {
        **SYSTEM_INFO,
        "recipes": {
            **SYSTEM_INFO["recipes"],
            "flm": {**SYSTEM_INFO["recipes"]["flm"], "backends": {"npu": {"state": "installable"}}},
        },
    }
    backends = available_backends(system_info, ENVIRONMENT)
    assert "flm-npu" not in {b["backend"] for b in backends}


def test_available_backends_modality_none_includes_everything():
    backends = available_backends(SYSTEM_INFO, ENVIRONMENT, modality=None)
    assert "kokoro-cpu" in {b["backend"] for b in backends}


def test_available_backends_unions_os_across_duplicate_support_entries():
    # Regression test: confirmed live against a real instance that
    # onnxruntime's "cpu" backend has THREE separate support entries (one
    # per device family: x86_64/windows, x86_64-arm64/linux, arm64/macos),
    # not one. Collapsing them by backend key and keeping only the last
    # entry silently dropped "linux" (the last entry in Lemonade's own list
    # order is macos-only), hiding an installed backend from a linux host.
    system_info = {
        **SYSTEM_INFO,
        "recipes": {
            **SYSTEM_INFO["recipes"],
            "onnxruntime": {
                "display_name": "ONNX Runtime",
                "modality": "Text classification",
                "backends": {"cpu": {"state": "installed", "version": "0.3.7"}},
                "support": [
                    {"backend": "cpu", "os": ["windows"]},
                    {"backend": "cpu", "os": ["linux"]},
                    {"backend": "cpu", "os": ["macos"]},
                ],
            },
        },
    }
    backends = available_backends(system_info, ENVIRONMENT, modality="Text classification")
    assert {b["backend"] for b in backends} == {"onnxruntime-cpu"}


def test_new_recipes_work_with_zero_recipe_specific_code():
    """Confirms LemonMatrix's chat-model pipeline (available_compute_engines,
    available_backends, validate_combo_against_profile) is recipe-name-agnostic
    -- it works off each recipe's own declared "modality" and "backends", not a
    hardcoded allowlist -- so a brand-new recipe Lemonade adds shows up and
    validates correctly with no LemonMatrix code changes at all.

    "vllm" and an AMD dGPU under ROCm are used here specifically: verified
    against Lemonade's own C++ source (src/cpp/include/lemon/backends/vllm/vllm.h)
    that vllm is a real, currently-shipping recipe -- modality "Text generation",
    ROCm-only, chat-completions-only (no embeddings/reranking interface) -- ahead
    of a real switch to AMD GPU hardware where it would actually be exercised.
    """
    system_info = {
        "devices": {
            "amd_gpu": [{"name": "AMD Radeon RX 7900 XTX", "vram_gb": 24, "available": True}],
            "cpu": {"available": True, "name": "AMD Ryzen 9 7950X"},
        },
        "recipes": {
            "vllm": {
                "display_name": "vLLM",
                "modality": "Text generation",
                "backends": {"rocm": {"state": "installed", "version": "0.6.0"}},
                "support": [{"backend": "rocm", "os": ["linux"]}],
            },
        },
    }
    environment = {"os_version": "Linux-6.8.0 (Ubuntu 24.04)"}

    # A GPU that size clears IGPU_VRAM_THRESHOLD_GB -> classified dgpu, same
    # AMD-device-list-based logic used for every other AMD GPU, regardless of
    # which backend (rocm vs vulkan) happens to be active for it.
    assert available_compute_engines(system_info) == ["cpu", "dgpu"]

    backends = available_backends(system_info, environment, modality="Text generation")
    assert {b["backend"] for b in backends} == {"vllm-rocm"}
    assert backends[0]["recipe_display"] == "vLLM"
    assert backends[0]["state"] == "installed"

    from lemonmatrix.capabilities import validate_combo_against_profile

    assert validate_combo_against_profile("dgpu", "vllm-rocm", system_info, environment) == []
    # cpu can't run a rocm backend -- still correctly rejected for a recipe
    # validate_combo_against_profile has never heard of by name.
    assert validate_combo_against_profile("cpu", "vllm-rocm", system_info, environment) != []
