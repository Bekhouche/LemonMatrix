"""Profile discovery and local storage.

A profile is LemonMatrix's record of one Lemonade instance: where to reach it,
plus the auto-discovered environment fingerprint (idea.md's "fixed facts").

Confirmed against a real /api/v1/system-info response (Ubuntu 22.04, 7x NVIDIA
L4, Lemonade 11.6.0): per-device info (cpu, amd_gpu, nvidia_gpu, amd_npu) lives
nested under a top-level "devices" key, while "OS Version" and "Physical
Memory" are flat top-level strings. There is no "OEM System" field on Linux at
all -- device_model falls back to the CPU name there. Lemonade also has no
single global driver_version; NVIDIA GPU entries carry their own per-GPU
driver_version, which is used as a fallback. None of this is formally spec'd
anywhere, so discovery stays defensive: it also checks the flat (undevices-
wrapped) shape from an earlier, unverified reading of the docs, in case some
other Lemonade version or platform uses it, and falls back to "unknown"
rather than crashing. Callers can override anything discovery got wrong via
CLI flags. Distinguishing an integrated GPU from a discrete one is a genuine
ambiguity in the amd_gpu list (both can appear as same-shaped entries) --
see _classify_gpus below for the heuristic and its escape hatch.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .client import LemonadeClient

DEFAULT_PROFILE_DIR = Path.home() / ".lemonmatrix" / "profiles"

# Below this VRAM size, a GPU entry is assumed integrated rather than discrete.
IGPU_VRAM_THRESHOLD_GB = 4.0


def _get_first(d: dict, *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default


def _parse_memory_gb(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"([\d.]+)\s*(GB|GiB|MB)?", str(value), re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "GB").upper()
    return number / 1024 if unit == "MB" else number


def _summarize_names(names: list[str]) -> str | None:
    """Collapse a list of GPU names into one fingerprint string, e.g. 7 identical
    NVIDIA L4s -> "7x NVIDIA L4". The schema's igpu/dgpu fields are single
    strings -- a fingerprint, not a hardware inventory -- so multi-GPU boxes
    (common on cloud/server profiles) get summarized rather than truncated to
    just the first card, which would silently under-report the hardware."""
    if not names:
        return None
    counts = Counter(names)
    return ", ".join(f"{count}x {name}" if count > 1 else name for name, count in counts.items())


def _classify_gpus(devices: dict) -> tuple[str | None, str | None, str | None, str | None, str | None]:
    """Return (igpu_summary, dgpu_summary, igpu_family, dgpu_family, driver_version)
    from the devices block.

    nvidia_gpu entries are always discrete. amd_gpu entries are split by a
    VRAM-size heuristic (see IGPU_VRAM_THRESHOLD_GB) because the API doesn't
    otherwise say which is integrated. Override with --igpu/--dgpu if wrong.
    Lemonade has no single global driver_version field; per-GPU entries (at
    least nvidia_gpu ones) carry their own, used here as a fallback.

    `family` (e.g. "gfx950", "gfx1151") is each GPU's own ROCm architecture
    codename -- confirmed live against a real AMD Instinct MI350X instance
    that this can be materially more useful than `name`: that card's own
    "name" field was the bare string "90500" (not a product name), while
    "family": "gfx950" was still correct and identifiable.
    """
    igpu_names: list[str] = []
    dgpu_names: list[str] = []
    igpu_families: list[str] = []
    dgpu_families: list[str] = []
    driver_version = None

    amd_gpus = devices.get("amd_gpu") or []
    if isinstance(amd_gpus, dict):
        amd_gpus = [amd_gpus]
    for gpu in amd_gpus:
        name = gpu.get("name")
        if not gpu.get("available", True) or not name:
            continue
        vram = gpu.get("vram_gb")
        family = gpu.get("family") or None
        if vram is not None and vram >= IGPU_VRAM_THRESHOLD_GB:
            dgpu_names.append(name)
            if family:
                dgpu_families.append(family)
        else:
            igpu_names.append(name)
            if family:
                igpu_families.append(family)
        driver_version = driver_version or gpu.get("driver_version")

    nvidia_gpus = devices.get("nvidia_gpu") or []
    if isinstance(nvidia_gpus, dict):
        nvidia_gpus = [nvidia_gpus]
    for gpu in nvidia_gpus:
        name = gpu.get("name")
        if not gpu.get("available", True) or not name:
            continue
        dgpu_names.append(name)
        family = gpu.get("family") or None
        if family:
            dgpu_families.append(family)
        driver_version = driver_version or gpu.get("driver_version")

    return (
        _summarize_names(igpu_names),
        _summarize_names(dgpu_names),
        _summarize_names(igpu_families),
        _summarize_names(dgpu_families),
        driver_version,
    )


@dataclass
class Profile:
    """One Lemonade instance: connection info plus its environment fingerprint."""

    name: str
    base_url: str
    environment: dict = field(default_factory=dict)
    api_key: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Profile":
        return cls(**data)

    def path(self, directory: Path | None = None) -> Path:
        return (directory or DEFAULT_PROFILE_DIR) / f"{self.name}.json"

    def save(self, directory: Path | None = None) -> Path:
        directory = directory or DEFAULT_PROFILE_DIR
        directory.mkdir(parents=True, exist_ok=True)
        path = self.path(directory)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, name: str, directory: Path | None = None) -> "Profile":
        path = (directory or DEFAULT_PROFILE_DIR) / f"{name}.json"
        return cls.from_dict(json.loads(path.read_text()))

    @classmethod
    def list_all(cls, directory: Path | None = None) -> list["Profile"]:
        directory = directory or DEFAULT_PROFILE_DIR
        if not directory.exists():
            return []
        return [cls.load(p.stem, directory) for p in sorted(directory.glob("*.json"))]


def discover_environment(
    client: LemonadeClient,
    *,
    driver_version: str | None = None,
    igpu_override: str | None = None,
    dgpu_override: str | None = None,
) -> tuple[dict, list[str]]:
    """Build the schema's `environment` block by querying a live Lemonade instance.

    Returns (environment, gaps) where `gaps` lists required fields that could
    not be discovered and were filled with "unknown" -- callers should surface
    this to the user rather than silently submitting a guessed fingerprint.
    """
    system_info = client.system_info()
    health = client.health()

    # Confirmed shape nests per-device info under "devices"; fall back to the
    # flat top-level shape from the (unverified) docs example in case another
    # Lemonade version/platform uses that instead.
    devices = system_info.get("devices") if isinstance(system_info.get("devices"), dict) else system_info

    igpu, dgpu, igpu_family, dgpu_family, gpu_driver_version = _classify_gpus(devices)
    npu = devices.get("amd_npu") or {}
    if isinstance(npu, list):
        npu = npu[0] if npu else {}

    cpu = devices.get("cpu") or {}
    cpu_name = _get_first(cpu, "name") if isinstance(cpu, dict) else (str(cpu) if cpu else None)
    cpu_name = cpu_name or _get_first(system_info, "Processor")

    environment = {
        "device_model": _get_first(system_info, "OEM System", "device_model", default=cpu_name or "unknown"),
        "cpu": cpu_name or "unknown",
        "memory_gb": _parse_memory_gb(_get_first(system_info, "Physical Memory", "memory_gb")),
        "os_version": _get_first(system_info, "OS Version", "os_version", default="unknown"),
        "driver_version": driver_version
        or gpu_driver_version
        or _get_first(system_info, "Driver Version", "GPU Driver", "driver_version", default="unknown"),
        "lemonade_version": health.get("version"),
    }

    igpu = igpu_override or igpu
    dgpu = dgpu_override or dgpu
    if igpu:
        environment["igpu"] = igpu
    if igpu_family:
        environment["igpu_family"] = igpu_family
    if dgpu:
        environment["dgpu"] = dgpu
    if dgpu_family:
        environment["dgpu_family"] = dgpu_family
    if npu.get("available") and npu.get("name"):
        environment["npu"] = npu["name"]

    rocm_version = _get_first(system_info, "ROCm Version", "rocm_version")
    if rocm_version:
        environment["rocm_version"] = rocm_version

    # Required by the schema but not guaranteed by the API; keep the caller
    # informed rather than silently shipping a plausible-looking fabrication.
    gaps = [
        required
        for required in ("device_model", "cpu", "memory_gb", "os_version", "driver_version")
        if environment.get(required) in (None, "unknown")
    ]

    return environment, gaps


DEFAULT_PORT = 13305


def build_url(host: str, port: int = DEFAULT_PORT, scheme: str = "http") -> str:
    return f"{scheme}://{host}:{port}"


def connect_and_save(
    name: str,
    base_url: str,
    api_key: str | None = None,
    *,
    driver_version: str | None = None,
    igpu_override: str | None = None,
    dgpu_override: str | None = None,
) -> tuple[Profile, list[str]]:
    """Connect to `base_url`, discover its environment, and save it as a profile.

    Shared by the CLI and the dashboard so "add a profile" means one thing
    everywhere. Raises ConnectionError if the instance never answers /live.
    """
    client = LemonadeClient(base_url, api_key=api_key)
    if not client.live():
        raise ConnectionError(f"{base_url} did not respond to /live. Is Lemonade running there?")

    environment, gaps = discover_environment(
        client, driver_version=driver_version, igpu_override=igpu_override, dgpu_override=dgpu_override
    )
    prof = Profile(name=name, base_url=base_url, environment=environment, api_key=api_key)
    prof.save()
    return prof, gaps
