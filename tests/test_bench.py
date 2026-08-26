import hashlib
import time

import pytest
from conftest import get_fake_server

from lemonmatrix.bench import (SweepConfig, _ci95_half_width, _competing_model_names,
                                _device_matches_engine, _ExclusivityMonitor, run_sweep)
from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.validate import validate_result


def _cfg(**overrides) -> SweepConfig:
    defaults = dict(
        model_name="Llama-3.1-8B-Instruct-GGUF",
        model_class="dense",
        quantization="Q4_K_M",
        context_length=4096,
        compute_engine="igpu",
        backend="llama.cpp-vulkan",
        os="windows",
        power_state="plugged",
        warmup_trials=1,
        measured_trials=2,
    )
    defaults.update(overrides)
    return SweepConfig(**defaults)


def test_run_sweep_produces_schema_conformant_result(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, trials = run_sweep(client, environment, _cfg())

    validate_result(result)  # raises on any schema violation

    assert result["metrics"]["decode"]["tokens_per_sec"] == 42.0
    assert result["metrics"]["ttft_ms"] == 180.0
    # prompt_tokens=40, ttft=0.18s -> 40 / 0.18 prefill tok/s
    assert round(result["metrics"]["prefill"]["tokens_per_sec"], 1) == round(40 / 0.18, 1)
    assert result["metrics"]["peak_memory_gb"] == 9.5
    assert result["metrics"]["trial_count"] == 2
    assert result["validity"]["model_reload_free"] is True
    assert result["validity"]["warmup_discarded"] is True
    # llamacpp/vulkan is "installed" in the fake server's recipes tree.
    assert result["environment"]["backend_version"] == "b10375"
    # raw trials are returned separately -- 2 measured trials
    assert len(trials) == 2
    assert all("ttft_ms" in t and "decode_tokens_per_sec" in t for t in trials)


def test_run_sweep_falls_back_to_host_memory_when_vram_unavailable(fake_lemonade, monkeypatch):
    # Confirmed live: llamacpp's Vulkan backend on a real NVIDIA host
    # reported vram_gb as null on every /api/v1/system-stats poll, even
    # mid-inference -- without this fallback, peak_memory_gb stayed None
    # and every such run failed schema validation outright.
    import conftest

    monkeypatch.setitem(conftest.FAKE_SYSTEM_STATS, "vram_gb", None)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg())

    validate_result(result)
    assert result["metrics"]["peak_memory_gb"] == conftest.FAKE_SYSTEM_STATS["memory_gb"]
    assert "peak_memory_gb is host RAM, not VRAM" in result["validity"]["notes"]


def test_run_sweep_tells_lemonade_which_backend_to_load(fake_lemonade):
    # Regression test: without threading cfg.backend through to /api/v1/load,
    # Lemonade has no backend selector at all and silently falls back to the
    # recipe's own default_backend -- which is what actually caused a real
    # 500 on a live instance when the selected backend (llamacpp:cpu) wasn't
    # installed but the recipe's default (cuda) also wasn't, so the load
    # failed regardless of what the UI showed as "selected".
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    run_sweep(client, environment, _cfg(backend="llama.cpp-cuda"))

    payload = get_fake_server(fake_lemonade).last_load_payload
    assert payload["llamacpp_backend"] == "cuda"


def test_run_sweep_notes_and_invalidates_when_backend_unresolvable(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(backend="totally-not-a-real-backend-string"))

    validate_result(result)
    assert result["validity"]["valid"] is False
    assert "could not resolve backend" in result["validity"]["notes"]
    # /api/v1/load still got called, just without a backend selector.
    payload = get_fake_server(fake_lemonade).last_load_payload
    assert not any(k.endswith("_backend") for k in payload)


def test_run_sweep_omits_backend_version_when_not_installed(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    # llamacpp/cuda is only "installable" in the fake server's recipes tree,
    # not "installed" -- reporting its version would misrepresent what's
    # actually running this benchmark.
    result, _ = run_sweep(client, environment, _cfg(backend="llama.cpp-cuda"))

    validate_result(result)
    assert "backend_version" not in result["environment"]


def test_run_sweep_marks_invalid_when_not_exclusive(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(exclusive_run=False))

    validate_result(result)
    assert result["validity"]["valid"] is False
    assert result["validity"]["exclusive_run"] is False


def test_run_sweep_marks_invalid_when_engine_backend_incompatible(fake_lemonade):
    # cpu engine + rocm backend is a physical contradiction: ROCm is a GPU
    # framework and cannot run on a CPU-only path.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(compute_engine="cpu", backend="llamacpp-rocm"))

    validate_result(result)
    assert result["validity"]["valid"] is False
    assert "engine/backend mismatch" in result["validity"]["notes"]


def test_run_sweep_applies_cost_model_when_price_given(fake_lemonade, monkeypatch):
    # Power must come from the profiled instance's own /api/v1/system-stats,
    # never a local hardware tool (a profile can point at any machine, not
    # necessarily the one this process runs on) -- so this drives it through
    # the fake server's stats response, the same path a real Lemonade
    # instance would use if it ever reports a watts-like field.
    import conftest

    monkeypatch.setitem(conftest.FAKE_SYSTEM_STATS, "power_w", 100.0)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(energy_price_usd_per_kwh=0.15))

    assert "cost_per_1k_tokens_usd" in result["metrics"]
    # 100W at 42 decode tok/s -> 100/42 J/token -> kWh/token -> $/1000 tokens.
    kwh_per_token = (100.0 / 42.0) / 3600 / 1000
    expected_cost = kwh_per_token * 0.15 * 1000
    assert result["metrics"]["cost_per_1k_tokens_usd"] == pytest.approx(expected_cost)
    assert result["metrics"]["decode"]["joules_per_token"] == pytest.approx(100.0 / 42.0)


def test_run_sweep_omits_cost_model_when_power_unavailable(fake_lemonade):
    # FAKE_SYSTEM_STATS has no power field by default -- matching every real
    # Lemonade instance checked so far, which don't report power draw at all.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(energy_price_usd_per_kwh=0.15))

    assert "cost_per_1k_tokens_usd" not in result["metrics"]
    assert "joules_per_token" not in result["metrics"]["decode"]
    assert "doesn't report power draw" in result["validity"]["notes"]


def test_competing_model_names_ignores_own_model_and_idle_models():
    health = {
        "all_models_loaded": [
            {"model_name": "own-model", "is_busy": True, "is_streaming": False},
            {"model_name": "idle-other", "is_busy": False, "is_streaming": False},
            {"model_name": "busy-other", "is_busy": True, "is_streaming": False},
            {"model_name": "streaming-other", "is_busy": False, "is_streaming": True},
        ]
    }
    assert _competing_model_names(health, "own-model") == ["busy-other", "streaming-other"]


def test_competing_model_names_handles_missing_or_empty_list():
    assert _competing_model_names({}, "own-model") == []
    assert _competing_model_names({"all_models_loaded": []}, "own-model") == []


@pytest.mark.parametrize(
    "poll_count,poll_failures,expected",
    [
        (0, 0, True),  # run finished before the first poll ever fired
        (3, 3, True),  # every poll failed
        (3, 0, False),  # every poll succeeded
        (3, 1, False),  # a mix of success and failure is still "reachable"
    ],
)
def test_exclusivity_monitor_health_unreachable(poll_count, poll_failures, expected):
    monitor = _ExclusivityMonitor(client=None, own_model_name="own-model")
    monitor.poll_count = poll_count
    monitor.poll_failures = poll_failures
    assert monitor.health_unreachable is expected


def test_exclusivity_monitor_detects_competing_model_via_fake_server(fake_lemonade, monkeypatch):
    import conftest

    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "other-model", "is_busy": True, "is_streaming": False}],
    )
    client = LemonadeClient(fake_lemonade)

    with _ExclusivityMonitor(client, "own-model", interval_s=0.05) as monitor:
        time.sleep(0.2)

    assert monitor.poll_count > 0
    assert monitor.poll_failures == 0
    assert monitor.competing_models == {"other-model"}
    assert monitor.health_unreachable is False


def test_run_sweep_verifies_exclusive_run_via_health_polling(fake_lemonade):
    # Default FAKE_HEALTH has no other loaded models, so run_sweep's own
    # multiple real HTTP round trips (load, per-trial chat/stats/system-stats)
    # give the background monitor time to poll cleanly at least once.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg())

    validate_result(result)
    assert result["validity"]["exclusive_run"] is True
    assert "verified via /api/v1/health polling" in result["validity"]["notes"]


def test_run_sweep_detects_competing_model_via_health_polling(fake_lemonade, monkeypatch):
    import conftest

    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "someone-elses-model", "is_busy": True, "is_streaming": False}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(exclusive_run=True))

    validate_result(result)
    assert result["validity"]["exclusive_run"] is False
    assert result["validity"]["valid"] is False
    assert "competing workload detected during measurement: someone-elses-model" in result["validity"]["notes"]


def test_run_sweep_falls_back_to_assertion_when_health_polling_fails(fake_lemonade, monkeypatch):
    # Every health poll fails (not just a slow/absent one) -- verification
    # itself didn't run, so the caller's own exclusive_run assertion must be
    # trusted rather than silently reporting a false "verified clean".
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    def _broken_health(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(LemonadeClient, "health", _broken_health)

    result, _ = run_sweep(client, environment, _cfg(exclusive_run=True))

    assert result["validity"]["exclusive_run"] is True
    assert "could not be verified" in result["validity"]["notes"]


@pytest.mark.parametrize(
    "device_str,compute_engine,expected",
    [
        (None, "cpu", None),  # never observed -- unverified, not a mismatch
        ("cpu", "cpu", True),
        ("gpu", "cpu", False),
        ("gpu", "igpu", True),
        ("gpu", "dgpu", True),
        ("cpu", "igpu", False),
        ("npu", "npu", True),
        ("cpu", "npu", False),
        ("cpu|gpu", "cpu", False),  # more devices actually used than claimed
    ],
)
def test_device_matches_engine(device_str, compute_engine, expected):
    assert _device_matches_engine(device_str, compute_engine) is expected


def test_run_sweep_marks_invalid_when_device_contradicts_engine_claim(fake_lemonade, monkeypatch):
    import conftest

    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "Llama-3.1-8B-Instruct-GGUF", "is_busy": False, "is_streaming": False, "device": "cpu"}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(compute_engine="dgpu"))

    validate_result(result)
    assert result["validity"]["valid"] is False
    assert "compute_engine claimed 'dgpu' but Lemonade's own /api/v1/health reported device 'cpu'" in result["validity"]["notes"]


def test_run_sweep_valid_when_device_confirms_engine_claim(fake_lemonade, monkeypatch):
    import conftest

    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "Llama-3.1-8B-Instruct-GGUF", "is_busy": False, "is_streaming": False, "device": "gpu"}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(compute_engine="igpu"))

    validate_result(result)
    assert result["validity"]["valid"] is True
    assert "notes" not in result["validity"] or "compute_engine claimed" not in result["validity"].get("notes", "")


def test_run_sweep_marks_invalid_when_watchdog_reset_detected(fake_lemonade, monkeypatch):
    import conftest

    monkeypatch.setitem(
        conftest.FAKE_HEALTH,
        "all_models_loaded",
        [{"model_name": "Llama-3.1-8B-Instruct-GGUF", "is_busy": False, "is_streaming": False, "watchdog_reset": True}],
    )
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg())

    validate_result(result)
    assert result["validity"]["valid"] is False
    assert result["validity"]["model_reload_free"] is False
    assert "backend watchdog force-restarted the model process" in result["validity"]["notes"]


def test_run_sweep_marks_invalid_when_quantization_contradicts_checkpoint(fake_lemonade):
    # The fake server's "Qwen3.8-27B-GGUF-Q4_K_M" model has checkpoint
    # "unsloth/Qwen3.8-27B-GGUF:Q4_K_M" -- claiming a different quant than
    # the ":VARIANT" suffix implies must be caught.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(
        client, environment,
        _cfg(model_name="Qwen3.8-27B-GGUF-Q4_K_M", quantization="Q8_0", backend="llama.cpp-cpu", compute_engine="cpu"),
    )

    validate_result(result)
    assert result["validity"]["valid"] is False
    assert "quantization claimed 'Q8_0' but Lemonade's own checkpoint metadata implies 'Q4_K_M'" in result["validity"]["notes"]


def test_run_sweep_valid_when_quantization_matches_checkpoint(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(
        client, environment,
        _cfg(model_name="Qwen3.8-27B-GGUF-Q4_K_M", quantization="Q4_K_M", backend="llama.cpp-cpu", compute_engine="cpu"),
    )

    validate_result(result)
    assert result["validity"]["valid"] is True
    assert "quantization claimed" not in result["validity"].get("notes", "")


def test_run_sweep_quantization_unverified_when_checkpoint_has_no_variant_suffix(fake_lemonade):
    # "Llama-3.1-8B-Instruct-GGUF" has no checkpoint field and no quant-like
    # id suffix in the fake fixture -- nothing to check against, so an
    # invented quantization string must not be flagged as a mismatch.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(quantization="totally-made-up"))

    validate_result(result)
    assert "quantization claimed" not in result["validity"].get("notes", "")


def test_ci95_half_width_uses_t_distribution_not_fixed_1_96():
    # 5 values -> df=4 -> t=2.776, not the normal approximation's 1.96.
    values = [10.0, 11.0, 9.0, 10.5, 9.5]
    sd = 0.7905694150420949  # statistics.stdev(values)
    expected = 2.776 * sd / (5 ** 0.5)
    assert _ci95_half_width(values) == pytest.approx(expected, rel=1e-4)


def test_ci95_half_width_none_below_two_trials():
    assert _ci95_half_width([]) is None
    assert _ci95_half_width([10.0]) is None


def test_run_sweep_includes_prompt_sha256(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg())

    expected = hashlib.sha256(_cfg().prompt.encode()).hexdigest()
    assert result["metrics"]["prompt_sha256"] == expected


def test_run_sweep_includes_ci95_alongside_stddev_and_p95(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(measured_trials=3))

    validate_result(result)
    assert "ci95_half_width" in result["metrics"]["decode"]
    assert "ci95_half_width" in result["metrics"]["prefill"]


def test_run_sweep_applies_hardware_amortization_without_any_power_data(fake_lemonade):
    # Confirmed against Lemonade's own metrics source: no real instance
    # reports power draw, so the energy half of the cost model is always
    # None in practice. Hardware amortization needs no power data at all --
    # this must populate cost_per_1k_tokens_usd on its own.
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(client, environment, _cfg(hardware_cost_usd=2000.0, hardware_lifetime_hours=26280.0))

    validate_result(result)
    decode_tps = result["metrics"]["decode"]["tokens_per_sec"]
    hardware_cost_per_hour = 2000.0 / 26280.0
    expected = (hardware_cost_per_hour / 3600 / decode_tps) * 1000
    assert result["metrics"]["hardware_cost_per_1k_tokens_usd"] == pytest.approx(expected)
    assert result["metrics"]["cost_per_1k_tokens_usd"] == pytest.approx(expected)
    assert "energy_cost_per_1k_tokens_usd" not in result["metrics"]


def test_run_sweep_combines_hardware_and_energy_cost_when_both_available(fake_lemonade, monkeypatch):
    import conftest

    monkeypatch.setitem(conftest.FAKE_SYSTEM_STATS, "power_w", 100.0)
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    result, _ = run_sweep(
        client, environment,
        _cfg(hardware_cost_usd=2000.0, hardware_lifetime_hours=26280.0, energy_price_usd_per_kwh=0.15),
    )

    validate_result(result)
    metrics = result["metrics"]
    assert metrics["cost_per_1k_tokens_usd"] == pytest.approx(
        metrics["energy_cost_per_1k_tokens_usd"] + metrics["hardware_cost_per_1k_tokens_usd"]
    )
