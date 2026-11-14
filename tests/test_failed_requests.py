"""A failed measured request is left out instead of ending the run; above 5%
of measured requests failing, the run is not ranked; if nothing works, the run
still fails."""

import pytest
import requests

import lemonmatrix.bench as bench_mod
from lemonmatrix.bench import SweepConfig, run_sweep
from lemonmatrix.client import LemonadeAPIError, LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.validate import validate_result


def _cfg(**overrides) -> SweepConfig:
    values = dict(
        model_name="Llama-3.1-8B-Instruct-GGUF", model_class="dense", quantization="Q4_K_M",
        context_length=4096, compute_engine="igpu", backend="llama.cpp-vulkan", os="windows",
        power_state="plugged", warmup_trials=1, measured_trials=40,
    )
    values.update(overrides)
    return SweepConfig(**values)


def _fail_on(monkeypatch, calls_to_fail: set[int], exc_factory):
    """Make the given _run_one calls (0-based, warmups included) raise."""
    original = bench_mod._run_one
    count = {"n": -1}

    def run_one(client, cfg):
        count["n"] += 1
        if count["n"] in calls_to_fail:
            raise exc_factory()
        return original(client, cfg)

    monkeypatch.setattr(bench_mod, "_run_one", run_one)


def _server_error():
    return LemonadeAPIError("[json.exception.type_error.302] type must be string, but is null")


def _env(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)
    return client, environment


def test_one_failed_request_is_left_out_and_the_run_stays_ranked(fake_lemonade, monkeypatch):
    _fail_on(monkeypatch, {10}, _server_error)
    client, environment = _env(fake_lemonade)

    result, trials = run_sweep(client, environment, _cfg())

    validate_result(result)
    assert len(trials) == 39
    assert result["validity"]["valid"] is True
    assert "1 of 40 measured requests failed" in result["validity"]["notes"]
    assert "type must be string" in result["validity"]["notes"]


def test_more_than_five_percent_failed_is_not_ranked(fake_lemonade, monkeypatch):
    _fail_on(monkeypatch, {5, 10, 15}, _server_error)
    client, environment = _env(fake_lemonade)

    result, trials = run_sweep(client, environment, _cfg())

    assert len(trials) == 37
    assert result["validity"]["valid"] is False
    assert "3 of 40 measured requests failed" in result["validity"]["notes"]
    assert "not ranked" in result["validity"]["notes"]


def test_a_failed_warmup_does_not_count(fake_lemonade, monkeypatch):
    _fail_on(monkeypatch, {0}, _server_error)
    client, environment = _env(fake_lemonade)

    result, trials = run_sweep(client, environment, _cfg(measured_trials=5))

    assert len(trials) == 5
    assert "failed" not in result["validity"].get("notes", "")


def test_a_broken_setup_still_fails_the_run(fake_lemonade, monkeypatch):
    _fail_on(monkeypatch, set(range(1, 10)), _server_error)  # every measured request
    client, environment = _env(fake_lemonade)

    with pytest.raises(LemonadeAPIError):
        run_sweep(client, environment, _cfg())


def test_connection_errors_are_recorded_without_the_server_address(fake_lemonade, monkeypatch):
    _fail_on(monkeypatch, {3}, lambda: requests.ConnectionError("HTTPConnectionPool(host='10.1.2.3', port=13305)"))
    client, environment = _env(fake_lemonade)

    result, _ = run_sweep(client, environment, _cfg())

    assert "first: ConnectionError" in result["validity"]["notes"]
    assert "10.1.2.3" not in result["validity"]["notes"]
