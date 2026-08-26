from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import Profile, discover_environment


def test_discover_environment_maps_fake_server(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, gaps = discover_environment(client)

    assert environment["device_model"] == "HP Ryzen AI Max+ 395 (Strix Halo)"
    assert environment["cpu"] == "AMD Ryzen AI 9 HX 370"
    assert environment["memory_gb"] == 128.0
    assert environment["os_version"] == "Windows 11 Pro 24H2"
    assert environment["lemonade_version"] == "8.1.0"
    assert environment["npu"] == "XDNA 2"
    # Below the 4 GB threshold -> classified as integrated, not discrete;
    # the fake profile also carries a discrete NVIDIA card (added so
    # acestep/trellis's cuda-only backends have a real "dgpu" to validate
    # against) which nvidia_gpu entries always classify as discrete.
    assert environment["igpu"] == "AMD Radeon 8060S"
    assert environment["dgpu"] == "NVIDIA RTX 4090"
    # Each AMD GPU's own ROCm architecture codename, read straight off its
    # "family" field -- the fake nvidia_gpu entry carries no such field
    # (matches every real nvidia_gpu response seen so far), so only igpu_family
    # populates here.
    assert environment["igpu_family"] == "gfx1150"
    assert "dgpu_family" not in environment

    # driver_version has no field in the fake server's system-info -> a gap.
    assert environment["driver_version"] == "unknown"
    assert "driver_version" in gaps
    assert "device_model" not in gaps


def test_discover_environment_respects_overrides(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, gaps = discover_environment(
        client, driver_version="24.10.1", igpu_override="Custom iGPU", dgpu_override="Custom dGPU"
    )
    assert environment["driver_version"] == "24.10.1"
    assert environment["igpu"] == "Custom iGPU"
    assert environment["dgpu"] == "Custom dGPU"
    assert gaps == []


def test_profile_round_trips_through_disk(tmp_path):
    prof = Profile(name="my-strix-halo", base_url="http://localhost:8000", environment={"cpu": "x"})
    prof.save(tmp_path)

    loaded = Profile.load("my-strix-halo", tmp_path)
    assert loaded.base_url == prof.base_url
    assert loaded.environment == prof.environment

    all_profiles = Profile.list_all(tmp_path)
    assert [p.name for p in all_profiles] == ["my-strix-halo"]


def test_discover_environment_reads_gpu_family_even_when_name_is_not_a_product_name():
    """Live-verified against a real AMD Instinct MI350X Lemonade instance:
    that GPU's own "name" field is the bare string "90500" (not a product
    name) and it carries no "driver_version" field at all -- but "family":
    "gfx950" is still a correct, identifiable ROCm architecture codename.
    discover_environment() doesn't call Lemonade over HTTP in this test (no
    fake-server support for a synthetic system-info payload exists here), so
    it exercises _classify_gpus directly through the same devices-block shape
    a real /api/v1/system-info response has.
    """
    from lemonmatrix.profile import _classify_gpus

    devices = {
        "amd_gpu": [
            {
                "available": True,
                "family": "gfx950",
                "integrated": False,
                "name": "90500",
                "virtual_mem_gb": 125.8,
                "vram_gb": 287.6,
            }
        ],
        "nvidia_gpu": [{"available": False, "error": "No NVIDIA discrete GPU found", "name": ""}],
    }
    igpu, dgpu, igpu_family, dgpu_family, driver_version = _classify_gpus(devices)
    assert igpu is None
    assert dgpu == "90500"
    assert igpu_family is None
    assert dgpu_family == "gfx950"
    assert driver_version is None
