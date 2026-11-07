"""Regression tests against the real /api/v1/system-info shape captured from a
live Lemonade 11.6.0 instance (Ubuntu 22.04, dual-Xeon-class host, 7x NVIDIA L4).

That instance exposed device info nested under "devices" (not flat, as the
docs-derived guess in test_profile.py's fake server assumed), had no
"OEM System" field at all, and gave each GPU its own driver_version rather
than one global field. This is what caught the original mapping bug.
"""

from lemonmatrix.profile import discover_environment


class _StubClient:
    """Duck-types just enough of LemonadeClient for discover_environment."""

    def __init__(self, system_info: dict, health: dict):
        self._system_info = system_info
        self._health = health

    def system_info(self) -> dict:
        return self._system_info

    def health(self) -> dict:
        return self._health


REAL_SYSTEM_INFO = {
    "OS Version": "Linux-6.8.0-117-generic (Ubuntu 22.04)",
    "Physical Memory": "251.50 GB",
    "Processor": "Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz",
    "devices": {
        "amd_gpu": [],
        "amd_npu": {
            "available": False,
            "error": "No NPU device found with amdxdna driver",
            "family": "",
            # Lemonade reports the CPU name here when no NPU exists -- a
            # quirk of the API, not something discovery should surface.
            "name": "Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz",
            "utilization": 0.0,
        },
        "cpu": {
            "available": True,
            "cores": 32,
            "family": "x86_64",
            "name": "Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz",
            "threads": 64,
        },
        "nvidia_gpu": [
            {
                "available": True,
                "compute_capability": "8.9",
                "driver_version": "535.309.01",
                "family": "sm_89",
                "index": i,
                "name": "NVIDIA L4",
                "uuid": f"GPU-{i}",
                "vram_gb": 22.494140625,
            }
            for i in range(7)
        ],
    },
}

REAL_HEALTH = {"status": "ok", "version": "11.6.0", "model_loaded": None, "all_models_loaded": []}


def test_discover_environment_reads_nested_devices_block():
    environment, gaps = discover_environment(_StubClient(REAL_SYSTEM_INFO, REAL_HEALTH))

    assert environment["cpu"] == "Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz"
    assert environment["memory_gb"] == 251.5
    assert environment["os_version"] == "Linux-6.8.0-117-generic (Ubuntu 22.04)"
    assert environment["lemonade_version"] == "11.6.0"


def test_discover_environment_summarizes_multiple_identical_gpus():
    environment, _ = discover_environment(_StubClient(REAL_SYSTEM_INFO, REAL_HEALTH))
    assert environment["dgpu"] == "7x NVIDIA L4"
    assert "igpu" not in environment


def test_discover_environment_falls_back_to_cpu_name_for_device_model():
    # No "OEM System" field at all on this real response.
    environment, gaps = discover_environment(_StubClient(REAL_SYSTEM_INFO, REAL_HEALTH))
    assert environment["device_model"] == "Intel(R) Xeon(R) Silver 4314 CPU @ 2.40GHz"
    assert "device_model" not in gaps


def test_discover_environment_uses_per_gpu_driver_version():
    # No global driver_version field -- falls back to the GPUs' own field.
    environment, gaps = discover_environment(_StubClient(REAL_SYSTEM_INFO, REAL_HEALTH))
    assert environment["driver_version"] == "535.309.01"
    assert "driver_version" not in gaps


def test_discover_environment_ignores_unavailable_npu_with_bogus_name():
    # amd_npu.available is False here, but Lemonade still fills its "name"
    # field with the CPU name -- must not be mistaken for a real NPU.
    environment, _ = discover_environment(_StubClient(REAL_SYSTEM_INFO, REAL_HEALTH))
    assert "npu" not in environment
