from lemonmatrix.backend_version import lookup_backend_version

SYSTEM_INFO = {
    "recipes": {
        "llamacpp": {
            "backends": {
                "vulkan": {"state": "installed", "version": "b10375"},
                "cuda": {"state": "installable", "version": "b10397"},
                "rocm": {"state": "unsupported"},
            }
        },
        "flm": {"backends": {"npu": {"state": "installed", "version": "v1.0.1"}}},
    }
}


def test_returns_version_when_installed():
    assert lookup_backend_version(SYSTEM_INFO, "llama.cpp-vulkan") == "b10375"
    assert lookup_backend_version(SYSTEM_INFO, "fastflowlm-npu") == "v1.0.1"


def test_returns_none_when_only_installable():
    assert lookup_backend_version(SYSTEM_INFO, "llama.cpp-cuda") is None


def test_returns_none_when_unsupported():
    assert lookup_backend_version(SYSTEM_INFO, "llama.cpp-rocm") is None


def test_returns_none_for_unknown_backend_string():
    assert lookup_backend_version(SYSTEM_INFO, "some-custom-backend") is None


def test_returns_none_when_recipes_missing_entirely():
    assert lookup_backend_version({}, "llama.cpp-vulkan") is None


def test_resolves_canonical_recipe_backend_form():
    # capabilities.py generates "<recipe_key>-<backend_key>" directly, e.g.
    # "llamacpp-vulkan" (no dot) -- must resolve without an explicit alias.
    assert lookup_backend_version(SYSTEM_INFO, "llamacpp-vulkan") == "b10375"


def test_canonical_form_handles_hyphenated_recipe_keys():
    system_info = {"recipes": {"sd-cpp": {"backends": {"cuda": {"state": "installed", "version": "master-726"}}}}}
    assert lookup_backend_version(system_info, "sd-cpp-cuda") == "master-726"
