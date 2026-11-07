import pytest
import requests

from lemonmatrix.client import LemonadeAPIError, LemonadeClient


def test_install_backend_success(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    result = client.install_backend("llamacpp", "cuda")
    assert result["status"] == "success"


def test_install_backend_raises_on_invalid_backend(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    with pytest.raises(requests.HTTPError):
        client.install_backend("llamacpp", "does-not-exist")


def test_install_backend_raises_when_fields_missing(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    with pytest.raises(requests.HTTPError):
        client.install_backend("", "")


def test_search_registry_returns_results(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    data = client.search_registry("llama-3.2-1b")
    assert data["results"][0]["repository_id"] == "bartowski/Llama-3.2-1B-Instruct-GGUF"


def test_pull_variants_returns_recipe_and_variants(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    data = client.pull_variants("bartowski/Llama-3.2-1B-Instruct-GGUF")
    assert data["recipe"] == "llamacpp"
    assert {v["name"] for v in data["variants"]} == {"Q4_K_M", "Q8_0"}


def test_pull_variants_raises_for_unknown_checkpoint(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    with pytest.raises(requests.HTTPError):
        client.pull_variants("nonexistent/repo")


def test_pull_model_success(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    result = client.pull_model(
        "user.Llama-3.2-1B-Instruct-GGUF-Q4_K_M",
        recipe="llamacpp",
        checkpoint="bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
    )
    assert result["status"] == "success"


def test_pull_model_reports_error_status_without_raising(fake_lemonade):
    # Per Lemonade's docs, a failed pull comes back as {"status": "error", ...}
    # rather than a raised exception -- callers must check the status field
    # themselves. Not independently confirmed against a real failing pull
    # (avoided triggering any real download attempt to test this).
    client = LemonadeClient(fake_lemonade)
    result = client.pull_model(
        "user.Bad", recipe="llamacpp", checkpoint="does-not-exist/repo:Q4_K_M"
    )
    assert result["status"] == "error"


def test_start_model_download_returns_immediately_with_job_snapshot(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    job = client.start_model_download(
        "user.Llama-3.2-1B-Instruct-GGUF-Q4_K_M",
        recipe="llamacpp",
        checkpoint="bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
    )
    assert job["id"] == "model:user.Llama-3.2-1B-Instruct-GGUF-Q4_K_M"
    assert job["status"] == "downloading"


def test_list_downloads_reflects_started_jobs(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    assert client.list_downloads() == []

    client.start_model_download("user.A", recipe="llamacpp", checkpoint="x/y:Q4_K_M")
    jobs = client.list_downloads()
    assert len(jobs) == 1
    assert jobs[0]["model_name"] == "user.A"


def test_control_download_pause_and_cancel(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    job = client.start_model_download("user.A", recipe="llamacpp", checkpoint="x/y:Q4_K_M")

    paused = client.control_download(job["id"], "pause")
    assert paused["status"] == "paused"

    cancelled = client.control_download(job["id"], "cancel")
    assert cancelled["status"] == "cancelled"


def test_control_download_remove(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    job = client.start_model_download("user.A", recipe="llamacpp", checkpoint="x/y:Q4_K_M")

    result = client.control_download(job["id"], "remove")
    assert result["status"] == "ok"
    assert client.list_downloads() == []


def test_control_download_remove_missing_job(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    result = client.control_download("model:does-not-exist", "remove")
    assert result == {"status": "ok", "missing": True}


def test_load_surfaces_lemonades_nested_error_message(fake_lemonade):
    # Confirmed live against a real CUDA backend that couldn't start: the
    # raised error's message is Lemonade's own explanation, not a generic
    # "500 Server Error: ... for url: ...".
    client = LemonadeClient(fake_lemonade)
    with pytest.raises(LemonadeAPIError) as exc_info:
        client.load("trigger-load-failure")
    assert "llama-server failed to start" in str(exc_info.value)
    assert isinstance(exc_info.value, requests.HTTPError)  # existing except/pytest.raises(HTTPError) still catch it


def test_install_backend_surfaces_bare_string_error_message(fake_lemonade):
    client = LemonadeClient(fake_lemonade)
    with pytest.raises(LemonadeAPIError) as exc_info:
        client.install_backend("", "")
    assert "recipe" in str(exc_info.value) and "backend" in str(exc_info.value)
