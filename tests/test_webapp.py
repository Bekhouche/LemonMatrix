import json
from urllib.parse import urlparse

import pytest

from lemonmatrix.webapp import create_app


def _host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    return parsed.hostname, parsed.port


@pytest.fixture
def client(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path / "profiles")
    app = create_app(results_dir=tmp_path / "results")
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client, fake_lemonade, tmp_path


def _write_result(tmp_path, profile: str, run_id: str, valid: bool, model_name: str = "Llama-3.1-8B-Instruct-GGUF") -> None:
    out_dir = tmp_path / "results" / profile
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "model": {"name": model_name, "class": "dense", "quantization": "Q4_K_M", "context_length": 4096},
        "config": {"compute_engine": "igpu", "backend": "llamacpp-vulkan", "os": "windows", "power_state": "plugged"},
        "environment": {
            "device_model": "Test Device", "cpu": "Test CPU", "memory_gb": 32,
            "os_version": "Windows 11", "driver_version": "1.0",
        },
        "metrics": {"prefill": {"tokens_per_sec": 100}, "decode": {"tokens_per_sec": 50}, "ttft_ms": 100, "peak_memory_gb": 8},
        "validity": {"valid": valid, "warmup_discarded": True, "thermal_ok": True, "exclusive_run": True},
    }
    (out_dir / f"{run_id}.json").write_text(json.dumps(result))


def test_index_renders_with_no_data(client):
    test_client, _, _ = client
    resp = test_client.get("/")
    assert resp.status_code == 200
    assert b"No results yet" in resp.data


def test_index_hides_invalid_runs_by_default(client):
    test_client, _, tmp_path = client
    _write_result(tmp_path, "demo", "valid-run", valid=True, model_name="Valid-Model")
    _write_result(tmp_path, "demo", "invalid-run", valid=False, model_name="Invalid-Model")

    resp = test_client.get("/")

    assert b"Valid-Model" in resp.data
    assert b"Invalid-Model" not in resp.data
    # The checkbox itself must render checked to reflect the applied default.
    assert b'name="valid_only" id="valid_only" value="1" checked' in resp.data


def test_index_shows_invalid_runs_when_filter_explicitly_cleared(client):
    test_client, _, tmp_path = client
    _write_result(tmp_path, "demo", "valid-run", valid=True, model_name="Valid-Model")
    _write_result(tmp_path, "demo", "invalid-run", valid=False, model_name="Invalid-Model")

    # Submitting the filter form with the checkbox unchecked omits
    # "valid_only" entirely, same as a real unchecked checkbox would.
    resp = test_client.get("/?filtered=1")

    assert b"Valid-Model" in resp.data
    assert b"Invalid-Model" in resp.data


def test_index_sort_link_preserves_explicitly_cleared_filter(client):
    test_client, _, tmp_path = client
    _write_result(tmp_path, "demo", "valid-run", valid=True, model_name="Valid-Model")
    _write_result(tmp_path, "demo", "invalid-run", valid=False, model_name="Invalid-Model")

    # Simulates clicking a column-sort link after having cleared the filter --
    # sort links carry forward whatever was already in the query string,
    # including "filtered", so the cleared state must persist.
    resp = test_client.get("/?filtered=1&sort=model.name&dir=asc")

    assert b"Valid-Model" in resp.data
    assert b"Invalid-Model" in resp.data


def test_index_empty_state_mentions_hidden_invalid_runs(client):
    test_client, _, tmp_path = client
    _write_result(tmp_path, "demo", "invalid-run", valid=False, model_name="Invalid-Model")

    resp = test_client.get("/")

    assert b"Invalid-Model" not in resp.data
    assert b"hidden by the" in resp.data


def test_profiles_page_renders(client):
    test_client, _, _ = client
    resp = test_client.get("/profiles")
    assert resp.status_code == 200
    assert b"No profiles yet" in resp.data


def test_add_profile_via_form(client):
    test_client, fake_url, _ = client
    resp = test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)
    assert resp.status_code == 200
    assert b"demo" in resp.data
    assert b"HP Ryzen AI Max" in resp.data


def test_add_profile_via_host_port(client):
    test_client, fake_url, _ = client
    host, port = _host_port(fake_url)
    resp = test_client.post(
        "/profiles/add",
        data={"name": "demo2", "host": host, "port": str(port)},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"demo2" in resp.data


def test_detect_finds_fake_server(client):
    test_client, fake_url, _ = client
    _, port = _host_port(fake_url)
    resp = test_client.post("/profiles/detect", data={"ports": str(port)})
    assert resp.status_code == 200
    assert fake_url.encode() in resp.data


def test_run_form_and_submission_populate_leaderboard(client):
    test_client, fake_url, _ = client

    add_resp = test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)
    assert add_resp.status_code == 200

    form_resp = test_client.get("/profiles/demo/run")
    assert form_resp.status_code == 200
    assert b"Run a sweep" in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/run",
        data={
            "model_name": "Llama-3.1-8B-Instruct-GGUF",
            "model_class": "dense",
            "quantization": "Q4_K_M",
            "context_length": "4096",
            "compute_engine": "igpu",
            "backend": "llama.cpp-vulkan",
            "os": "windows",
            "power_state": "plugged",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"Llama-3.1-8B-Instruct-GGUF" in run_resp.data

    leaderboard_resp = test_client.get("/")
    assert b"Llama-3.1-8B-Instruct-GGUF" in leaderboard_resp.data
    assert b"llama.cpp-vulkan" in leaderboard_resp.data


def test_run_form_applies_hardware_amortization(client):
    test_client, fake_url, tmp_path = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/run")
    assert b"Hardware cost" in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/run",
        data={
            "model_name": "Llama-3.1-8B-Instruct-GGUF",
            "model_class": "dense",
            "quantization": "Q4_K_M",
            "context_length": "4096",
            "compute_engine": "igpu",
            "backend": "llama.cpp-vulkan",
            "os": "windows",
            "power_state": "plugged",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
            "hardware_cost_usd": "2000",
            "hardware_lifetime_hours": "26280",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200

    written = list((tmp_path / "results" / "demo").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert "hardware_cost_per_1k_tokens_usd" in result["metrics"]
    assert result["metrics"]["cost_per_1k_tokens_usd"] == pytest.approx(result["metrics"]["hardware_cost_per_1k_tokens_usd"])


def test_run_form_executes_via_job_engine(client):
    test_client, fake_url, tmp_path = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/run")
    assert b"Lemonade's job engine" in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/run",
        data={
            "model_name": "Llama-3.1-8B-Instruct-GGUF",
            "model_class": "dense",
            "quantization": "Q4_K_M",
            "context_length": "4096",
            "compute_engine": "igpu",
            "backend": "llama.cpp-vulkan",
            "os": "windows",
            "power_state": "plugged",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
            "via_job_engine": "on",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"Llama-3.1-8B-Instruct-GGUF" in run_resp.data

    written = list((tmp_path / "results" / "demo").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["metrics"]["decode"]["tokens_per_sec"] == 42.0


def test_classify_form_and_submission(client):
    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/classify")
    assert form_resp.status_code == 200
    assert b"Classification benchmark" in form_resp.data
    assert conftest.FAKE_CLASSIFY_MODEL_ID.encode() in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/classify",
        data={
            "model_name": conftest.FAKE_CLASSIFY_MODEL_ID,
            "compute_engine": "cpu",
            "backend": "onnxruntime-cpu",
            "power_state": "plugged",
            "input_text": "Please verify your account now.",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"Classify run complete" in run_resp.data
    assert conftest.FAKE_CLASSIFY_MODEL_ID.encode() in run_resp.data

    # Never appears on the model/router leaderboard.
    leaderboard_resp = test_client.get("/")
    assert conftest.FAKE_CLASSIFY_MODEL_ID.encode() not in leaderboard_resp.data


def test_tts_form_and_submission(client):
    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/tts")
    assert form_resp.status_code == 200
    assert b"Text-to-speech benchmark" in form_resp.data
    assert conftest.FAKE_TTS_MODEL_ID.encode() in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/tts",
        data={
            "model_name": conftest.FAKE_TTS_MODEL_ID,
            "compute_engine": "cpu",
            "backend": "kokoro-cpu",
            "power_state": "plugged",
            "input_text": "Lemonade can speak",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"TTS run complete" in run_resp.data
    assert conftest.FAKE_TTS_MODEL_ID.encode() in run_resp.data

    leaderboard_resp = test_client.get("/")
    assert conftest.FAKE_TTS_MODEL_ID.encode() not in leaderboard_resp.data


def test_stt_form_and_submission(client):
    import io

    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/stt")
    assert form_resp.status_code == 200
    assert b"Speech-to-text benchmark" in form_resp.data
    assert conftest.FAKE_STT_MODEL_ID.encode() in form_resp.data

    wav_bytes = conftest._make_wav_bytes(2.0)
    run_resp = test_client.post(
        "/profiles/demo/stt",
        data={
            "model_name": conftest.FAKE_STT_MODEL_ID,
            "compute_engine": "cpu",
            "backend": "whispercpp-cpu",
            "power_state": "plugged",
            "audio_file": (io.BytesIO(wav_bytes), "sample.wav"),
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"STT run complete" in run_resp.data
    assert conftest.FAKE_STT_MODEL_ID.encode() in run_resp.data

    leaderboard_resp = test_client.get("/")
    assert conftest.FAKE_STT_MODEL_ID.encode() not in leaderboard_resp.data


def test_stt_form_rejects_non_wav_file(client):
    import io

    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    run_resp = test_client.post(
        "/profiles/demo/stt",
        data={
            "model_name": conftest.FAKE_STT_MODEL_ID,
            "compute_engine": "cpu",
            "backend": "whispercpp-cpu",
            "power_state": "plugged",
            "audio_file": (io.BytesIO(b"not a wav file"), "sample.wav"),
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"Couldn&#39;t read" in run_resp.data or b"Couldn't read" in run_resp.data


def test_imagegen_form_and_submission(client):
    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/imagegen")
    assert form_resp.status_code == 200
    assert b"Image generation benchmark" in form_resp.data
    assert conftest.FAKE_IMAGEGEN_MODEL_ID.encode() in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/imagegen",
        data={
            "model_name": conftest.FAKE_IMAGEGEN_MODEL_ID,
            "compute_engine": "cpu",
            "backend": "sd-cpp-cpu",
            "power_state": "plugged",
            "prompt": "A red circle",
            "image_size": "256x256",
            "steps": "2",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"Image-gen run complete" in run_resp.data
    assert conftest.FAKE_IMAGEGEN_MODEL_ID.encode() in run_resp.data

    leaderboard_resp = test_client.get("/")
    assert conftest.FAKE_IMAGEGEN_MODEL_ID.encode() not in leaderboard_resp.data


def test_audiogen_form_and_submission(client):
    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/audiogen")
    assert form_resp.status_code == 200
    assert b"Audio generation benchmark" in form_resp.data
    assert conftest.FAKE_AUDIOGEN_MODEL_ID.encode() in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/audiogen",
        data={
            "model_name": conftest.FAKE_AUDIOGEN_MODEL_ID,
            "compute_engine": "dgpu",
            "backend": "acestep-cuda",
            "power_state": "plugged",
            "prompt": "An upbeat acoustic guitar riff",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"Audio-gen run complete" in run_resp.data
    assert conftest.FAKE_AUDIOGEN_MODEL_ID.encode() in run_resp.data

    leaderboard_resp = test_client.get("/")
    assert conftest.FAKE_AUDIOGEN_MODEL_ID.encode() not in leaderboard_resp.data


def test_result_detail_404s_for_unknown_run(client):
    test_client, _, _ = client
    resp = test_client.get("/results/nope/nope")
    assert resp.status_code == 404


def test_failed_run_is_persisted_and_shown_in_failure_log(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    run_resp = test_client.post(
        "/profiles/demo/run",
        data={
            "model_name": "trigger-load-failure",
            "model_class": "dense",
            "quantization": "Q4_K_M",
            "context_length": "4096",
            "compute_engine": "igpu",
            "backend": "llama.cpp-vulkan",
            "power_state": "plugged",
            "warmup_trials": "1",
            "measured_trials": "1",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"saved to failure log" in run_resp.data
    assert b"llama-server failed to start" in run_resp.data

    failures_resp = test_client.get("/profiles/demo/failures")
    assert failures_resp.status_code == 200
    assert b"trigger-load-failure" in failures_resp.data
    assert b"llama-server failed to start" in failures_resp.data
    assert b"llama.cpp-vulkan" in failures_resp.data


def test_profile_detail_links_to_failures_only_when_present(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo")
    assert b"Failures (" not in resp.data

    test_client.post(
        "/profiles/demo/run",
        data={
            "model_name": "trigger-load-failure",
            "model_class": "dense",
            "quantization": "Q4_K_M",
            "context_length": "4096",
            "compute_engine": "igpu",
            "backend": "llama.cpp-vulkan",
            "power_state": "plugged",
        },
    )

    resp = test_client.get("/profiles/demo")
    assert b"Failures (1)" in resp.data


def test_failures_page_empty_state(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/failures")
    assert resp.status_code == 200
    assert b"No failures recorded yet" in resp.data


def test_sweep_batch_failure_is_persisted_to_failure_log(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    # Drives sweep_batch's pieces directly (rather than through the
    # /profiles/<name>/sweep route, which resolves quant variants against
    # actually-pulled models) since "trigger-load-failure" isn't a real
    # pulled model id -- this test only needs to confirm _run_batch persists
    # a failure record, not the route's own model-resolution logic.
    from lemonmatrix.sweep_batch import SweepBatch, _run_batch

    batch = SweepBatch("demo", [{"model_name": "trigger-load-failure", "quantization": "Q4_K_M", "context_length": 4096, "compute_engine": "igpu", "backend": "llama.cpp-vulkan", "power_state": "plugged"}])
    _run_batch(
        batch,
        base_url=fake_url,
        api_key=None,
        environment={},
        results_dir=test_client.application.config["RESULTS_DIR"],
        model_class="dense",
        os_name="linux",
        power_cap_w=None,
        warmup_trials=1,
        measured_trials=1,
        max_tokens=32,
        exclusive_run=True,
        energy_price_usd_per_kwh=None,
    )
    assert batch.failed_count == 1

    failures_resp = test_client.get("/profiles/demo/failures")
    assert b"trigger-load-failure" in failures_resp.data
    assert b"sweep_item" in failures_resp.data
    assert batch.id.encode() in failures_resp.data


def test_profile_detail_shows_available_backends_and_engines(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo")
    assert resp.status_code == 200
    assert b"Available to run" in resp.data
    # llamacpp/vulkan is "installed"; cuda is only "installable" -- both text-gen.
    assert b"llamacpp-vulkan" in resp.data
    assert b"llamacpp-cuda" in resp.data
    assert b"igpu" in resp.data  # amd_gpu entry is below the iGPU VRAM threshold


def test_profile_detail_shows_install_button_only_for_installable(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo")
    html = resp.data.decode()
    # cuda is "installable" -> gets an Install button; vulkan is already
    # "installed" -> no button for it.
    assert 'value="cuda"' in html
    assert html.count("Install</button>") == 1


def test_install_backend_success_flashes_and_redirects(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.post(
        "/profiles/demo/backends/install",
        data={"recipe": "llamacpp", "backend_key": "cuda"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Installed llamacpp:cuda" in resp.data


def test_install_backend_failure_flashes_error(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.post(
        "/profiles/demo/backends/install",
        data={"recipe": "llamacpp", "backend_key": "does-not-exist"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Install failed" in resp.data


def test_install_backend_requires_recipe_and_backend(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.post("/profiles/demo/backends/install", data={}, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Missing recipe/backend" in resp.data


def test_run_form_populates_backend_dropdown_from_capabilities(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/run")
    assert resp.status_code == 200
    assert b'id="backend"' in resp.data and b'name="backend"' in resp.data
    assert b"llamacpp-vulkan" in resp.data
    assert b"Llama.cpp GPU" in resp.data
    # /api/v1/models returned models -- offered as a real <select> of base
    # models, with a dependent quantization <select> and model_name derived
    # via JS, not free text.
    assert b'<select id="model_base" name="model_base_display"' in resp.data
    assert b'<select id="quantization" name="quantization"' in resp.data
    assert b"Llama-3.1-8B-Instruct-GGUF" in resp.data
    assert b"Qwen3.8-27B-GGUF" in resp.data


def test_run_form_shows_os_as_read_only_fact_not_a_choice(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    # FAKE_SYSTEM_INFO's "OS Version" is "Windows 11 Pro 24H2" -- shown as
    # plain text, not a <select>, since it's a fixed profile fact.
    resp = test_client.get("/profiles/demo/run")
    assert b'name="os"' not in resp.data
    assert b"windows" in resp.data
    assert b"fixed by this profile" in resp.data


def test_run_form_prefills_quantization_and_context_from_first_model(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/run")
    html = resp.data.decode()
    # First model (Llama-3.1-8B-Instruct-GGUF) has no checkpoint and its id's
    # only trailing segment is "-GGUF" -- correctly not mistaken for a quant,
    # so its quantization option shows the "(unknown)" placeholder.
    assert '<option value="" data-model-id="Llama-3.1-8B-Instruct-GGUF" data-context="4096">' in html
    assert 'id="context_length" name="context_length" required value="4096"' in html
    assert 'id="model_name" name="model_name" value="Llama-3.1-8B-Instruct-GGUF"' in html


def test_run_form_switching_model_gets_its_own_quant_variants_in_js_data(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/run")
    html = resp.data.decode()
    # The JS lookup table must carry both base models' variant lists so
    # switching the Model dropdown can repopulate Quantization client-side.
    assert '"Llama-3.1-8B-Instruct-GGUF": [{' in html
    assert '"Qwen3.8-27B-GGUF": [{' in html
    assert '"id": "Qwen3.8-27B-GGUF-Q4_K_M"' in html


def test_run_form_falls_back_to_free_text_when_no_models_pulled(client, monkeypatch):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    # Exercise the empty-model-list fallback path distinctly from the
    # populated one above, without needing a second fake server.
    monkeypatch.setattr("lemonmatrix.client.LemonadeClient.models", lambda self: [])

    resp = test_client.get("/profiles/demo/run")
    assert resp.status_code == 200
    assert b'<input type="text" id="model_name" name="model_name" required' in resp.data
    assert b"add one first" in resp.data


def test_profile_detail_links_to_add_a_model(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo")
    assert b"Add a model" in resp.data
    assert b"/profiles/demo/models/add" in resp.data


def test_models_add_page_with_no_query_shows_empty_search_form(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/models/add")
    assert resp.status_code == 200
    assert b"Search Hugging Face" in resp.data


def test_models_add_search_shows_results(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/models/add?q=llama-3.2-1b")
    assert resp.status_code == 200
    assert b"bartowski/Llama-3.2-1B-Instruct-GGUF" in resp.data
    assert b"View variants" in resp.data


def test_models_add_variants_shows_quantizations_and_sizes(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get(
        "/profiles/demo/models/add?q=llama-3.2-1b&checkpoint=bartowski/Llama-3.2-1B-Instruct-GGUF"
    )
    assert resp.status_code == 200
    assert b"Q4_K_M" in resp.data
    assert b"Q8_0" in resp.data
    assert b"0.81 GB" in resp.data  # 807694464 bytes


def test_pull_model_starts_background_job_and_redirects_to_downloads(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.post(
        "/profiles/demo/models/pull",
        data={
            "model_name": "user.Llama-3.2-1B-Instruct-GGUF-Q4_K_M",
            "recipe": "llamacpp",
            "checkpoint": "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Started downloading user.Llama-3.2-1B-Instruct-GGUF-Q4_K_M" in resp.data
    assert b"Downloads for demo" in resp.data
    assert b"downloading" in resp.data


def test_pull_model_requires_model_name_and_checkpoint(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.post("/profiles/demo/models/pull", data={}, follow_redirects=True)
    assert resp.status_code == 200
    assert b"Missing model name or checkpoint" in resp.data


def test_pull_model_flashes_error_when_instance_unreachable(client):
    test_client, _, _ = client
    # Save a profile pointing at a port nothing listens on (privileged/
    # unassigned), bypassing `profile add`'s own liveness check, to exercise
    # the pull route's error handling specifically.
    from lemonmatrix.profile import Profile

    Profile(name="dead", base_url="http://127.0.0.1:1").save()
    resp = test_client.post(
        "/profiles/dead/models/pull",
        data={"model_name": "user.X", "recipe": "llamacpp", "checkpoint": "x/y:Q4_K_M"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"start download of user.X" in resp.data


def test_downloads_page_empty_state(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/downloads")
    assert resp.status_code == 200
    assert b"No downloads yet" in resp.data


def test_downloads_page_shows_progress_bar_for_active_job(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)
    test_client.post(
        "/profiles/demo/models/pull",
        data={"model_name": "user.A", "recipe": "llamacpp", "checkpoint": "x/y:Q4_K_M"},
    )

    resp = test_client.get("/profiles/demo/downloads")
    assert resp.status_code == 200
    assert b"user.A" in resp.data
    assert b"progress-track" in resp.data
    assert b"Pause" in resp.data
    assert b"Cancel" in resp.data


def test_downloads_control_pause_then_cancel(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)
    test_client.post(
        "/profiles/demo/models/pull",
        data={"model_name": "user.A", "recipe": "llamacpp", "checkpoint": "x/y:Q4_K_M"},
    )
    job_id = "model:user.A"

    paused_resp = test_client.post(
        "/profiles/demo/downloads/control", data={"id": job_id, "action": "pause"}, follow_redirects=True
    )
    assert b"paused" in paused_resp.data

    cancelled_resp = test_client.post(
        "/profiles/demo/downloads/control", data={"id": job_id, "action": "cancel"}, follow_redirects=True
    )
    assert b"cancelled" in cancelled_resp.data


def test_downloads_control_remove_clears_job(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)
    test_client.post(
        "/profiles/demo/models/pull",
        data={"model_name": "user.A", "recipe": "llamacpp", "checkpoint": "x/y:Q4_K_M"},
    )

    resp = test_client.post(
        "/profiles/demo/downloads/control",
        data={"id": "model:user.A", "action": "remove"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"No downloads yet" in resp.data


def test_downloads_control_requires_valid_action(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.post(
        "/profiles/demo/downloads/control", data={"id": "model:x", "action": "not-a-real-action"}, follow_redirects=True
    )
    assert resp.status_code == 200


def _wait_for_batch_done(test_client, timeout=10.0):
    import time

    batches = test_client.application.config["SWEEP_BATCHES"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(b.status == "done" for b in batches.values()):
            return
        time.sleep(0.1)
    raise AssertionError("sweep batch did not finish in time")


def test_sweep_form_renders_checkboxes_from_capabilities(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/sweep")
    assert resp.status_code == 200
    assert b'name="quantizations"' in resp.data
    assert b'name="engines"' in resp.data
    assert b'name="backends"' in resp.data
    assert b'name="power_states"' in resp.data
    assert b"llamacpp-vulkan" in resp.data


def test_sweep_post_starts_batch_and_redirects_to_status(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    # Use two valid engine+backend combos: igpu+vulkan and cpu+cpu (cuda is
    # installable only and igpu+cuda would be invalid, so we avoid those).
    resp = test_client.post(
        "/profiles/demo/sweep",
        data={
            "model_base": "Llama-3.1-8B-Instruct-GGUF",
            "quantizations": [""],
            "engines": ["igpu", "cpu"],
            "backends": ["llamacpp-vulkan", "llamacpp-cpu"],
            "power_states": ["plugged"],
            "model_class": "dense",
            "warmup_trials": "1",
            "measured_trials": "1",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200, resp.data.decode()
    # igpu+vulkan and cpu+cpu are valid; igpu+cpu and cpu+vulkan are dropped → 2 combos.
    assert b"Started a sweep of 2 combination" in resp.data
    assert b"Sweep" in resp.data

    _wait_for_batch_done(test_client)
    batch = next(iter(test_client.application.config["SWEEP_BATCHES"].values()))
    assert batch.total_count == 2
    assert batch.completed_count == 2


def test_sweep_status_page_shows_progress_table(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)
    test_client.post(
        "/profiles/demo/sweep",
        data={
            "model_base": "Llama-3.1-8B-Instruct-GGUF",
            "quantizations": [""],
            "engines": ["igpu"],
            "backends": ["llamacpp-vulkan"],
            "power_states": ["plugged"],
            "model_class": "dense",
            "warmup_trials": "1",
            "measured_trials": "1",
        },
    )
    _wait_for_batch_done(test_client)
    batch_id = next(iter(test_client.application.config["SWEEP_BATCHES"].keys()))

    resp = test_client.get(f"/profiles/demo/sweeps/{batch_id}")
    assert resp.status_code == 200
    assert b"llamacpp-vulkan" in resp.data
    assert b"completed" in resp.data


def test_sweep_blocks_second_concurrent_batch_for_same_profile(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    from lemonmatrix.sweep_batch import SweepBatch

    # Inject an already-running batch directly, bypassing timing races with
    # the real background thread, to deterministically test the guard.
    fake_batch = SweepBatch("demo", [{"model_name": "x"}])
    test_client.application.config["SWEEP_BATCHES"][fake_batch.id] = fake_batch

    resp = test_client.post(
        "/profiles/demo/sweep",
        data={
            "model_base": "Llama-3.1-8B-Instruct-GGUF",
            "quantizations": [""],
            "engines": ["igpu"],
            "backends": ["llamacpp-vulkan"],
            "power_states": ["plugged"],
            "model_class": "dense",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"already running" in resp.data
    # Only the injected fake batch exists -- no new one was started.
    assert len(test_client.application.config["SWEEP_BATCHES"]) == 1


def test_sweep_requires_at_least_one_selection_per_axis(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.post(
        "/profiles/demo/sweep",
        data={"model_base": "Llama-3.1-8B-Instruct-GGUF", "quantizations": [""], "model_class": "dense"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Pick at least one" in resp.data
    assert len(test_client.application.config["SWEEP_BATCHES"]) == 0


def test_sweep_enforces_max_combinations_cap(client, monkeypatch):
    monkeypatch.setattr("lemonmatrix.webapp.MAX_SWEEP_COMBINATIONS", 1)
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    # Use 2 valid combos so the cap check fires after filtering.
    resp = test_client.post(
        "/profiles/demo/sweep",
        data={
            "model_base": "Llama-3.1-8B-Instruct-GGUF",
            "quantizations": [""],
            "engines": ["igpu", "cpu"],
            "backends": ["llamacpp-vulkan", "llamacpp-cpu"],
            "power_states": ["plugged"],
            "model_class": "dense",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"exceeds the 1 limit" in resp.data
    assert len(test_client.application.config["SWEEP_BATCHES"]) == 0


def test_run_form_shows_router_option_and_router_run_completes(client):
    import conftest

    test_client, fake_url, tmp_path = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/run")
    assert form_resp.status_code == 200
    assert b"Run type" in form_resp.data
    assert conftest.FAKE_ROUTER_MODEL_ID.encode() in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/run",
        data={
            "run_type": "router",
            "router_model": conftest.FAKE_ROUTER_MODEL_ID,
            "context_length": "4096",
            "power_state": "plugged",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200, run_resp.data.decode()
    assert conftest.FAKE_ROUTER_MODEL_ID.encode() in run_resp.data

    written = list((tmp_path / "results" / "demo").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["run_type"] == "router"
    assert result["config"]["compute_engine"] == "router"
    assert result["config"]["backend"] == "collection.router"

    # Router runs are model/router-leaderboard citizens, unlike the other
    # bolt-on modalities -- confirm it shows up there.
    leaderboard_resp = test_client.get("/")
    assert conftest.FAKE_ROUTER_MODEL_ID.encode() in leaderboard_resp.data


def test_run_form_blocks_job_engine_for_router(client):
    import conftest

    test_client, fake_url, tmp_path = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    run_resp = test_client.post(
        "/profiles/demo/run",
        data={
            "run_type": "router",
            "router_model": conftest.FAKE_ROUTER_MODEL_ID,
            "context_length": "4096",
            "power_state": "plugged",
            "via_job_engine": "on",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200
    assert b"doesn&#39;t support router runs" in run_resp.data or b"doesn't support router runs" in run_resp.data
    assert not list((tmp_path / "results" / "demo").glob("*.json"))


def test_sweep_form_shows_router_option(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/sweep")
    assert resp.status_code == 200
    assert b'name="router_models"' in resp.data


def test_sweep_router_form_and_submission(client):
    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.post(
        "/profiles/demo/sweep",
        data={
            "run_type": "router",
            "router_models": [conftest.FAKE_ROUTER_MODEL_ID],
            "router_context_length": "4096",
            "power_states": ["plugged"],
            "warmup_trials": "1",
            "measured_trials": "1",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200, resp.data.decode()
    assert b"Started a sweep of 1 combination" in resp.data

    _wait_for_batch_done(test_client)
    batch = next(iter(test_client.application.config["SWEEP_BATCHES"].values()))
    assert batch.total_count == 1
    assert batch.completed_count == 1


def test_sweep_router_form_blocks_job_engine(client):
    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.post(
        "/profiles/demo/sweep",
        data={
            "run_type": "router",
            "router_models": [conftest.FAKE_ROUTER_MODEL_ID],
            "power_states": ["plugged"],
            "via_job_engine": "on",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"doesn&#39;t support router runs" in resp.data or b"doesn't support router runs" in resp.data
    assert len(test_client.application.config["SWEEP_BATCHES"]) == 0


def test_embeddings_form_and_submission(client):
    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/embeddings")
    assert form_resp.status_code == 200
    assert b"Embeddings benchmark" in form_resp.data
    assert conftest.FAKE_EMBED_MODEL_ID.encode() in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/embeddings",
        data={
            "model_name": conftest.FAKE_EMBED_MODEL_ID,
            "compute_engine": "igpu",
            "backend": "llamacpp-vulkan",
            "power_state": "plugged",
            "input_texts": "hello world\na second sentence",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200, run_resp.data.decode()
    assert b"Embeddings run complete" in run_resp.data
    assert conftest.FAKE_EMBED_MODEL_ID.encode() in run_resp.data

    leaderboard_resp = test_client.get("/")
    assert conftest.FAKE_EMBED_MODEL_ID.encode() not in leaderboard_resp.data


def test_rerank_form_and_submission(client):
    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/rerank")
    assert form_resp.status_code == 200
    assert b"Reranking benchmark" in form_resp.data
    assert conftest.FAKE_RERANK_MODEL_ID.encode() in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/rerank",
        data={
            "model_name": conftest.FAKE_RERANK_MODEL_ID,
            "compute_engine": "igpu",
            "backend": "llamacpp-vulkan",
            "power_state": "plugged",
            "query": "capital of France",
            "documents": "Paris is the capital of France.\nBerlin is the capital of Germany.",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
        },
        follow_redirects=True,
    )
    assert run_resp.status_code == 200, run_resp.data.decode()
    assert b"Rerank run complete" in run_resp.data
    assert conftest.FAKE_RERANK_MODEL_ID.encode() in run_resp.data

    leaderboard_resp = test_client.get("/")
    assert conftest.FAKE_RERANK_MODEL_ID.encode() not in leaderboard_resp.data


def test_meshgen_form_and_submission(client):
    import io

    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    form_resp = test_client.get("/profiles/demo/meshgen")
    assert form_resp.status_code == 200
    assert b"3D mesh-generation benchmark" in form_resp.data
    assert conftest.FAKE_MESHGEN_MODEL_ID.encode() in form_resp.data

    run_resp = test_client.post(
        "/profiles/demo/meshgen",
        data={
            "model_name": conftest.FAKE_MESHGEN_MODEL_ID,
            "compute_engine": "dgpu",
            "backend": "trellis-cuda",
            "power_state": "plugged",
            "input_image": (io.BytesIO(b"fake png bytes"), "cat.png"),
            "resolution": "512",
            "warmup_trials": "1",
            "measured_trials": "1",
            "exclusive_run": "on",
        },
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert run_resp.status_code == 200, run_resp.data.decode()
    assert b"Mesh-gen run complete" in run_resp.data
    assert conftest.FAKE_MESHGEN_MODEL_ID.encode() in run_resp.data

    leaderboard_resp = test_client.get("/")
    assert conftest.FAKE_MESHGEN_MODEL_ID.encode() not in leaderboard_resp.data


def test_profile_debug_page_shows_raw_json(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/debug")
    assert resp.status_code == 200
    assert b"/api/v1/system-info" in resp.data
    assert b"/api/v1/health" in resp.data
    assert b"recipes" in resp.data


def test_result_detail_links_to_trials_when_sidecar_exists(client):
    test_client, fake_url, tmp_path = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    _write_result(tmp_path, "demo", "run1", valid=True)
    no_trials_resp = test_client.get("/results/demo/run1")
    assert no_trials_resp.status_code == 200
    assert b"View raw trials" not in no_trials_resp.data

    from lemonmatrix.results_store import save_trials

    save_trials(tmp_path / "results", "demo", "run1", [{"decode_tokens_per_sec": 42.0}])
    with_trials_resp = test_client.get("/results/demo/run1")
    assert b"View raw trials" in with_trials_resp.data

    trials_resp = test_client.get("/results/demo/run1/trials")
    assert trials_resp.status_code == 200
    assert b"decode_tokens_per_sec" in trials_resp.data


def test_trials_page_404s_when_no_sidecar(client):
    test_client, fake_url, tmp_path = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)
    _write_result(tmp_path, "demo", "run1", valid=True)

    resp = test_client.get("/results/demo/run1/trials")
    assert resp.status_code == 404


def test_profile_detail_links_to_new_modalities_and_debug(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo")
    assert resp.status_code == 200
    assert b'href="/profiles/demo/embeddings"' in resp.data
    assert b'href="/profiles/demo/rerank"' in resp.data
    assert b'href="/profiles/demo/meshgen"' in resp.data
    assert b'href="/profiles/demo/debug"' in resp.data


def test_profiles_page_shows_online_status_for_reachable_profile(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles")
    assert resp.status_code == 200
    assert b'class="status online"' in resp.data
    assert b">online<" in resp.data


def test_profiles_page_shows_offline_status_for_unreachable_profile(client):
    test_client, fake_url, tmp_path = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    # Point the saved profile file at a port nothing is listening on, without
    # going through connect_and_save (which would refuse to save an
    # unreachable instance) -- simulates a previously-good profile whose
    # instance has since gone offline.
    from lemonmatrix.profile import Profile

    prof = Profile.load("demo")
    prof.base_url = "http://127.0.0.1:1"
    prof.save()

    resp = test_client.get("/profiles")
    assert resp.status_code == 200
    assert b'class="status offline"' in resp.data
    assert b">offline<" in resp.data


def test_queue_form_renders(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles/demo/queue")
    assert resp.status_code == 200
    assert b"Run queue" in resp.data
    assert b"Add a run" in resp.data


def test_queue_runs_heterogeneous_combinations_sequentially(client):
    import conftest

    test_client, fake_url, tmp_path = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    queue_items = json.dumps(
        [
            {
                "run_type": "model",
                "model_name": "Llama-3.1-8B-Instruct-GGUF",
                "model_class": "dense",
                "quantization": "Q4_K_M",
                "context_length": 4096,
                "compute_engine": "igpu",
                "backend": "llamacpp-vulkan",
                "power_state": "plugged",
            },
            {
                "run_type": "model",
                "model_name": "Llama-3.1-8B-Instruct-GGUF",
                "model_class": "dense",
                "quantization": "Q4_K_M",
                "context_length": 4096,
                "compute_engine": "cpu",
                "backend": "llamacpp-cpu",
                "power_state": "plugged",
            },
            {
                "run_type": "router",
                "model_name": conftest.FAKE_ROUTER_MODEL_ID,
                "context_length": 4096,
                "power_state": "plugged",
            },
        ]
    )
    resp = test_client.post(
        "/profiles/demo/queue",
        data={"queue_json": queue_items, "warmup_trials": "1", "measured_trials": "1"},
        follow_redirects=True,
    )
    assert resp.status_code == 200, resp.data.decode()
    assert b"Started a queue of 3 run" in resp.data

    _wait_for_batch_done(test_client)
    batch = next(iter(test_client.application.config["SWEEP_BATCHES"].values()))
    assert batch.total_count == 3
    assert batch.completed_count == 3

    written = list((tmp_path / "results" / "demo").glob("*.json"))
    assert len(written) == 3
    run_types = {json.loads(p.read_text()).get("run_type", "model") for p in written}
    assert "router" in run_types


def test_queue_drops_incompatible_items_and_keeps_valid_ones(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    queue_items = json.dumps(
        [
            {
                "run_type": "model",
                "model_name": "Llama-3.1-8B-Instruct-GGUF",
                "model_class": "dense",
                "quantization": "Q4_K_M",
                "context_length": 4096,
                "compute_engine": "igpu",
                "backend": "llamacpp-vulkan",
                "power_state": "plugged",
            },
            {
                # cpu engine + a GPU-only backend key is an impossible pair.
                "run_type": "model",
                "model_name": "Llama-3.1-8B-Instruct-GGUF",
                "model_class": "dense",
                "quantization": "Q4_K_M",
                "context_length": 4096,
                "compute_engine": "cpu",
                "backend": "llamacpp-vulkan",
                "power_state": "plugged",
            },
        ]
    )
    resp = test_client.post(
        "/profiles/demo/queue",
        data={"queue_json": queue_items, "warmup_trials": "1", "measured_trials": "1"},
        follow_redirects=True,
    )
    assert resp.status_code == 200, resp.data.decode()
    assert b"Dropped 1 queued run" in resp.data
    assert b"Started a queue of 1 run" in resp.data

    _wait_for_batch_done(test_client)
    batch = next(iter(test_client.application.config["SWEEP_BATCHES"].values()))
    assert batch.total_count == 1


def test_queue_requires_at_least_one_item(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.post(
        "/profiles/demo/queue",
        data={"queue_json": "[]"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Add at least one run" in resp.data
    assert len(test_client.application.config["SWEEP_BATCHES"]) == 0


def test_queue_blocks_job_engine_when_router_item_present(client):
    import conftest

    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    queue_items = json.dumps(
        [{"run_type": "router", "model_name": conftest.FAKE_ROUTER_MODEL_ID, "context_length": 4096, "power_state": "plugged"}]
    )
    resp = test_client.post(
        "/profiles/demo/queue",
        data={"queue_json": queue_items, "via_job_engine": "on"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"doesn&#39;t support router runs" in resp.data or b"doesn't support router runs" in resp.data
    assert len(test_client.application.config["SWEEP_BATCHES"]) == 0


def test_queue_blocked_while_another_batch_is_running(client):
    test_client, fake_url, _ = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    from lemonmatrix.sweep_batch import SweepBatch

    fake_batch = SweepBatch("demo", [{"model_name": "x"}])
    test_client.application.config["SWEEP_BATCHES"][fake_batch.id] = fake_batch

    resp = test_client.get("/profiles/demo/queue")
    assert b"already running" in resp.data

    resp = test_client.post(
        "/profiles/demo/queue",
        data={"queue_json": json.dumps([{"run_type": "router", "model_name": "x", "power_state": "plugged"}])},
        follow_redirects=True,
    )
    assert b"already running" in resp.data
    assert len(test_client.application.config["SWEEP_BATCHES"]) == 1


def test_classify_form_blocks_incompatible_engine_before_running(client):
    import conftest

    test_client, fake_url, tmp_path = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    # onnxruntime only declares a "cpu" backend on this fake profile -- "npu"
    # is an impossible pairing that must be blocked before any trial runs,
    # not merely marked invalid afterward.
    resp = test_client.post(
        "/profiles/demo/classify",
        data={
            "model_name": conftest.FAKE_CLASSIFY_MODEL_ID,
            "compute_engine": "npu",
            "backend": "onnxruntime-cpu",
            "power_state": "plugged",
            "input_text": "hello",
            "warmup_trials": "1",
            "measured_trials": "1",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Fix the engine/backend combination" in resp.data
    assert not list((tmp_path / "results" / "demo" / "classify").glob("*.json"))


def test_profiles_page_shows_gpu_column_preferring_dgpu_over_igpu(client):
    test_client, fake_url, tmp_path = client
    test_client.post("/profiles/add", data={"name": "demo", "url": fake_url}, follow_redirects=True)

    resp = test_client.get("/profiles")
    assert resp.status_code == 200
    assert b">GPU<" in resp.data
    assert b">Device<" not in resp.data
    # The fake profile's fixture carries both an igpu (AMD Radeon 8060S) and a
    # dgpu (NVIDIA RTX 4090) -- the discrete card should win the single GPU
    # column, since that's the more relevant accelerator for benchmarking.
    assert b"NVIDIA RTX 4090" in resp.data
    assert b"AMD Radeon 8060S" not in resp.data
