import json
from urllib.parse import urlparse

from click.testing import CliRunner

from lemonmatrix.cli import cli


def _host_port(url: str) -> tuple[str, int]:
    parsed = urlparse(url)
    return parsed.hostname, parsed.port


def test_cli_profile_add_and_list(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output
    assert "Saved profile 'strix-halo'" in add_result.output

    list_result = runner.invoke(cli, ["profile", "list"])
    assert list_result.exit_code == 0, list_result.output
    assert "strix-halo" in list_result.output


def test_cli_profile_debug_prints_raw_endpoints(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    debug_result = runner.invoke(cli, ["profile", "debug", "strix-halo"])
    assert debug_result.exit_code == 0, debug_result.output
    assert "system-info" in debug_result.output
    assert "amd_gpu" in debug_result.output
    assert "\"version\": \"8.1.0\"" in debug_result.output


def test_cli_run_writes_valid_result(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    run_result = runner.invoke(
        cli,
        [
            "run",
            "--profile", "strix-halo",
            "--model", "Llama-3.1-8B-Instruct-GGUF",
            "--model-class", "dense",
            "--quant", "Q4_K_M",
            "--context-length", "4096",
            "--engine", "igpu",
            "--backend", "llama.cpp-vulkan",
            "--power-state", "plugged",
            "--warmup", "1",
            "--trials", "1",
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    assert "Wrote" in run_result.output

    written = list((tmp_path / "results" / "strix-halo").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["model"]["name"] == "Llama-3.1-8B-Instruct-GGUF"
    assert result["config"]["backend"] == "llama.cpp-vulkan"
    # No --os flag exists -- derived from the profile's own environment
    # (FAKE_SYSTEM_INFO's "OS Version" is "Windows 11 Pro 24H2").
    assert result["config"]["os"] == "windows"


def test_cli_run_via_job_engine_writes_valid_result(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    run_result = runner.invoke(
        cli,
        [
            "run",
            "--profile", "strix-halo",
            "--model", "Llama-3.1-8B-Instruct-GGUF",
            "--model-class", "dense",
            "--quant", "Q4_K_M",
            "--context-length", "4096",
            "--engine", "igpu",
            "--backend", "llama.cpp-vulkan",
            "--power-state", "plugged",
            "--warmup", "1",
            "--trials", "1",
            "--via-job-engine",
        ],
    )
    assert run_result.exit_code == 0, run_result.output
    assert "durable Lemonade job" in run_result.output
    assert "Wrote" in run_result.output

    written = list((tmp_path / "results" / "strix-halo").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["model"]["name"] == "Llama-3.1-8B-Instruct-GGUF"


def test_cli_run_via_job_engine_rejects_router(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])

    result = runner.invoke(
        cli,
        [
            "run",
            "--profile", "strix-halo",
            "--model", "my-collection-router",
            "--run-type", "router",
            "--context-length", "4096",
            "--power-state", "plugged",
            "--via-job-engine",
        ],
    )
    assert result.exit_code != 0
    assert "does not support --run-type router" in result.output


def test_cli_classify_writes_valid_result(fake_lemonade, tmp_path, monkeypatch):
    import conftest

    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    classify_result = runner.invoke(
        cli,
        [
            "classify",
            "--profile", "strix-halo",
            "--model", conftest.FAKE_CLASSIFY_MODEL_ID,
            "--engine", "cpu",
            "--backend", "onnxruntime-cpu",
            "--power-state", "plugged",
            "--input-text", "Please verify your account now.",
            "--warmup", "1",
            "--trials", "1",
        ],
    )
    assert classify_result.exit_code == 0, classify_result.output
    assert "Wrote" in classify_result.output

    written = list((tmp_path / "results" / "strix-halo" / "classify").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["run_type"] == "classify"
    assert result["model"]["name"] == conftest.FAKE_CLASSIFY_MODEL_ID

    # Not on the model/router leaderboard's results tree.
    assert list((tmp_path / "results" / "strix-halo").glob("*.json")) == []

    list_result = runner.invoke(cli, ["classify-results", "--profile", "strix-halo"])
    assert list_result.exit_code == 0, list_result.output
    assert conftest.FAKE_CLASSIFY_MODEL_ID in list_result.output


def test_cli_classify_requires_input_text_or_file(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])

    result = runner.invoke(
        cli,
        [
            "classify",
            "--profile", "strix-halo",
            "--model", "some-classifier",
            "--engine", "cpu",
            "--backend", "onnxruntime-cpu",
            "--power-state", "plugged",
        ],
    )
    assert result.exit_code != 0
    assert "Missing required option" in result.output


def test_cli_tts_writes_valid_result(fake_lemonade, tmp_path, monkeypatch):
    import conftest

    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    tts_result = runner.invoke(
        cli,
        [
            "tts",
            "--profile", "strix-halo",
            "--model", conftest.FAKE_TTS_MODEL_ID,
            "--engine", "cpu",
            "--backend", "kokoro-cpu",
            "--power-state", "plugged",
            "--input-text", "Lemonade can speak",
            "--warmup", "1",
            "--trials", "1",
        ],
    )
    assert tts_result.exit_code == 0, tts_result.output
    assert "Wrote" in tts_result.output

    written = list((tmp_path / "results" / "strix-halo" / "tts").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["run_type"] == "tts"
    assert result["model"]["name"] == conftest.FAKE_TTS_MODEL_ID

    # Not on the model/router leaderboard's results tree.
    assert list((tmp_path / "results" / "strix-halo").glob("*.json")) == []

    list_result = runner.invoke(cli, ["tts-results", "--profile", "strix-halo"])
    assert list_result.exit_code == 0, list_result.output
    assert conftest.FAKE_TTS_MODEL_ID in list_result.output


def test_cli_stt_writes_valid_result(fake_lemonade, tmp_path, monkeypatch):
    import conftest

    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(conftest._make_wav_bytes(2.0))

    stt_result = runner.invoke(
        cli,
        [
            "stt",
            "--profile", "strix-halo",
            "--model", conftest.FAKE_STT_MODEL_ID,
            "--engine", "cpu",
            "--backend", "whispercpp-cpu",
            "--power-state", "plugged",
            "--audio-file", str(audio_path),
            "--warmup", "1",
            "--trials", "1",
        ],
    )
    assert stt_result.exit_code == 0, stt_result.output
    assert "Wrote" in stt_result.output

    written = list((tmp_path / "results" / "strix-halo" / "stt").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["run_type"] == "stt"
    assert result["model"]["name"] == conftest.FAKE_STT_MODEL_ID

    # Not on the model/router leaderboard's results tree.
    assert list((tmp_path / "results" / "strix-halo").glob("*.json")) == []

    list_result = runner.invoke(cli, ["stt-results", "--profile", "strix-halo"])
    assert list_result.exit_code == 0, list_result.output
    assert conftest.FAKE_STT_MODEL_ID in list_result.output


def test_cli_imagegen_writes_valid_result(fake_lemonade, tmp_path, monkeypatch):
    import conftest

    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    imagegen_result = runner.invoke(
        cli,
        [
            "imagegen",
            "--profile", "strix-halo",
            "--model", conftest.FAKE_IMAGEGEN_MODEL_ID,
            "--engine", "cpu",
            "--backend", "sd-cpp-cpu",
            "--power-state", "plugged",
            "--prompt", "A red circle",
            "--size", "256x256",
            "--steps", "2",
            "--warmup", "1",
            "--trials", "1",
        ],
    )
    assert imagegen_result.exit_code == 0, imagegen_result.output
    assert "Wrote" in imagegen_result.output

    written = list((tmp_path / "results" / "strix-halo" / "imagegen").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["run_type"] == "imagegen"
    assert result["model"]["name"] == conftest.FAKE_IMAGEGEN_MODEL_ID
    assert result["metrics"]["steps"] == 2

    # Not on the model/router leaderboard's results tree.
    assert list((tmp_path / "results" / "strix-halo").glob("*.json")) == []

    list_result = runner.invoke(cli, ["imagegen-results", "--profile", "strix-halo"])
    assert list_result.exit_code == 0, list_result.output
    assert conftest.FAKE_IMAGEGEN_MODEL_ID in list_result.output


def test_cli_imagegen_edit_operation(fake_lemonade, tmp_path, monkeypatch):
    import conftest

    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])

    input_image = tmp_path / "cat.png"
    input_image.write_bytes(b"not a real png, the fake server never decodes it")

    result = runner.invoke(
        cli,
        [
            "imagegen",
            "--profile", "strix-halo",
            "--model", conftest.FAKE_IMAGEGEN_MODEL_ID,
            "--engine", "cpu",
            "--backend", "sd-cpp-cpu",
            "--power-state", "plugged",
            "--operation", "edit",
            "--prompt", "add a hat",
            "--input-image", str(input_image),
            "--warmup", "1",
            "--trials", "1",
        ],
    )
    assert result.exit_code == 0, result.output

    written = list((tmp_path / "results" / "strix-halo" / "imagegen").glob("*.json"))
    assert len(written) == 1
    saved = json.loads(written[0].read_text())
    assert saved["metrics"]["operation"] == "edit"


def test_cli_imagegen_edit_requires_input_image(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])

    result = runner.invoke(
        cli,
        [
            "imagegen",
            "--profile", "strix-halo",
            "--model", "some-model",
            "--engine", "cpu",
            "--backend", "sd-cpp-cpu",
            "--power-state", "plugged",
            "--operation", "edit",
            "--prompt", "add a hat",
        ],
    )
    assert result.exit_code != 0
    assert "--input-image is required" in result.output


def test_cli_audiogen_writes_valid_result(fake_lemonade, tmp_path, monkeypatch):
    import conftest

    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    audiogen_result = runner.invoke(
        cli,
        [
            "audiogen",
            "--profile", "strix-halo",
            "--model", conftest.FAKE_AUDIOGEN_MODEL_ID,
            "--engine", "dgpu",
            "--backend", "acestep-cuda",
            "--power-state", "plugged",
            "--prompt", "An upbeat acoustic guitar riff",
            "--warmup", "1",
            "--trials", "1",
        ],
    )
    assert audiogen_result.exit_code == 0, audiogen_result.output
    assert "Wrote" in audiogen_result.output

    written = list((tmp_path / "results" / "strix-halo" / "audiogen").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["run_type"] == "audiogen"
    assert result["model"]["name"] == conftest.FAKE_AUDIOGEN_MODEL_ID

    # Not on the model/router leaderboard's results tree.
    assert list((tmp_path / "results" / "strix-halo").glob("*.json")) == []

    list_result = runner.invoke(cli, ["audiogen-results", "--profile", "strix-halo"])
    assert list_result.exit_code == 0, list_result.output
    assert conftest.FAKE_AUDIOGEN_MODEL_ID in list_result.output


def test_cli_embeddings_writes_valid_result(fake_lemonade, tmp_path, monkeypatch):
    import conftest

    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    embeddings_result = runner.invoke(
        cli,
        [
            "embeddings",
            "--profile", "strix-halo",
            "--model", conftest.FAKE_EMBED_MODEL_ID,
            "--engine", "cpu",
            "--backend", "llama.cpp-cpu",
            "--power-state", "plugged",
            "--input", "hello world",
            "--input", "a second sentence",
            "--warmup", "1",
            "--trials", "1",
        ],
    )
    assert embeddings_result.exit_code == 0, embeddings_result.output
    assert "Wrote" in embeddings_result.output

    written = list((tmp_path / "results" / "strix-halo" / "embeddings").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["run_type"] == "embeddings"
    assert result["model"]["name"] == conftest.FAKE_EMBED_MODEL_ID
    assert result["metrics"]["batch_size"] == 2

    # Not on the model/router leaderboard's results tree.
    assert list((tmp_path / "results" / "strix-halo").glob("*.json")) == []

    list_result = runner.invoke(cli, ["embeddings-results", "--profile", "strix-halo"])
    assert list_result.exit_code == 0, list_result.output
    assert conftest.FAKE_EMBED_MODEL_ID in list_result.output


def test_cli_rerank_writes_valid_result(fake_lemonade, tmp_path, monkeypatch):
    import conftest

    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    rerank_result = runner.invoke(
        cli,
        [
            "rerank",
            "--profile", "strix-halo",
            "--model", conftest.FAKE_RERANK_MODEL_ID,
            "--engine", "cpu",
            "--backend", "llama.cpp-cpu",
            "--power-state", "plugged",
            "--query", "capital of France",
            "--document", "Paris is the capital of France.",
            "--document", "Berlin is the capital of Germany.",
            "--warmup", "1",
            "--trials", "1",
        ],
    )
    assert rerank_result.exit_code == 0, rerank_result.output
    assert "Wrote" in rerank_result.output

    written = list((tmp_path / "results" / "strix-halo" / "rerank").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["run_type"] == "rerank"
    assert result["model"]["name"] == conftest.FAKE_RERANK_MODEL_ID
    assert result["metrics"]["document_count"] == 2

    # Not on the model/router leaderboard's results tree.
    assert list((tmp_path / "results" / "strix-halo").glob("*.json")) == []

    list_result = runner.invoke(cli, ["rerank-results", "--profile", "strix-halo"])
    assert list_result.exit_code == 0, list_result.output
    assert conftest.FAKE_RERANK_MODEL_ID in list_result.output


def test_cli_meshgen_writes_valid_result(fake_lemonade, tmp_path, monkeypatch):
    import conftest

    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    runner = CliRunner()
    add_result = runner.invoke(cli, ["profile", "add", "strix-halo", "--url", fake_lemonade])
    assert add_result.exit_code == 0, add_result.output

    input_image = tmp_path / "cat.png"
    input_image.write_bytes(b"not a real png, the fake server never decodes it")

    meshgen_result = runner.invoke(
        cli,
        [
            "meshgen",
            "--profile", "strix-halo",
            "--model", conftest.FAKE_MESHGEN_MODEL_ID,
            "--engine", "dgpu",
            "--backend", "trellis-cuda",
            "--power-state", "plugged",
            "--input-image", str(input_image),
            "--resolution", "512",
            "--warmup", "1",
            "--trials", "1",
        ],
    )
    assert meshgen_result.exit_code == 0, meshgen_result.output
    assert "Wrote" in meshgen_result.output

    written = list((tmp_path / "results" / "strix-halo" / "meshgen").glob("*.json"))
    assert len(written) == 1
    result = json.loads(written[0].read_text())
    assert result["run_type"] == "meshgen"
    assert result["model"]["name"] == conftest.FAKE_MESHGEN_MODEL_ID
    assert result["metrics"]["resolution"] == "512"

    # Not on the model/router leaderboard's results tree.
    assert list((tmp_path / "results" / "strix-halo").glob("*.json")) == []

    list_result = runner.invoke(cli, ["meshgen-results", "--profile", "strix-halo"])
    assert list_result.exit_code == 0, list_result.output
    assert conftest.FAKE_MESHGEN_MODEL_ID in list_result.output


def test_cli_profile_add_from_host_and_port(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    host, port = _host_port(fake_lemonade)

    runner = CliRunner()
    result = runner.invoke(cli, ["profile", "add", "by-host", "--host", host, "--port", str(port)])
    assert result.exit_code == 0, result.output

    show_result = runner.invoke(cli, ["profile", "show", "by-host"])
    assert json.loads(show_result.output)["base_url"] == f"http://{host}:{port}"


def test_cli_profile_detect_finds_and_saves(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    _, port = _host_port(fake_lemonade)

    runner = CliRunner()
    detect_result = runner.invoke(cli, ["profile", "detect", "--ports", str(port), "--save", "detected"])
    assert detect_result.exit_code == 0, detect_result.output
    assert "Found 1 instance" in detect_result.output

    show_result = runner.invoke(cli, ["profile", "show", "detected"])
    assert show_result.exit_code == 0, show_result.output
    assert f":{port}" in show_result.output


def _add_profile(runner: CliRunner, fake_url: str, name: str = "demo"):
    result = runner.invoke(cli, ["profile", "add", name, "--url", fake_url])
    assert result.exit_code == 0, result.output


def test_cli_install_backend_success(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    runner = CliRunner()
    _add_profile(runner, fake_lemonade)

    result = runner.invoke(cli, ["profile", "install-backend", "demo", "llamacpp", "cuda"])
    assert result.exit_code == 0, result.output
    assert "Installed llamacpp:cuda" in result.output


def test_cli_install_backend_failure(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    runner = CliRunner()
    _add_profile(runner, fake_lemonade)

    result = runner.invoke(cli, ["profile", "install-backend", "demo", "llamacpp", "does-not-exist"])
    assert result.exit_code != 0
    assert "Install failed" in result.output


def test_cli_search_models_shows_results(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    runner = CliRunner()
    _add_profile(runner, fake_lemonade)

    result = runner.invoke(cli, ["profile", "search-models", "demo", "llama-3.2-1b"])
    assert result.exit_code == 0, result.output
    assert "bartowski/Llama-3.2-1B-Instruct-GGUF" in result.output


def test_cli_pull_variants_shows_quantizations_and_sizes(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    runner = CliRunner()
    _add_profile(runner, fake_lemonade)

    result = runner.invoke(
        cli, ["profile", "pull-variants", "demo", "bartowski/Llama-3.2-1B-Instruct-GGUF"]
    )
    assert result.exit_code == 0, result.output
    assert "Q4_K_M" in result.output
    assert "0.81 GB" in result.output  # 807694464 bytes


def test_cli_pull_model_starts_background_job(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    runner = CliRunner()
    _add_profile(runner, fake_lemonade)

    result = runner.invoke(
        cli,
        [
            "profile", "pull-model", "demo", "user.Llama-3.2-1B-Instruct-GGUF-Q4_K_M",
            "--recipe", "llamacpp", "--checkpoint", "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Started downloading" in result.output

    downloads_result = runner.invoke(cli, ["profile", "downloads", "demo"])
    assert downloads_result.exit_code == 0, downloads_result.output
    assert "downloading" in downloads_result.output


def test_cli_downloads_empty_state(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    runner = CliRunner()
    _add_profile(runner, fake_lemonade)

    result = runner.invoke(cli, ["profile", "downloads", "demo"])
    assert result.exit_code == 0, result.output
    assert "No download jobs" in result.output


def test_cli_downloads_control_pause_then_cancel_then_remove(fake_lemonade, tmp_path, monkeypatch):
    monkeypatch.setattr("lemonmatrix.profile.DEFAULT_PROFILE_DIR", tmp_path)
    monkeypatch.setattr("lemonmatrix.cli.DEFAULT_PROFILE_DIR", tmp_path)
    runner = CliRunner()
    _add_profile(runner, fake_lemonade)
    runner.invoke(
        cli,
        ["profile", "pull-model", "demo", "user.A", "--recipe", "llamacpp", "--checkpoint", "x/y:Q4_K_M"],
    )
    job_id = "model:user.A"

    paused = runner.invoke(cli, ["profile", "downloads-control", "demo", job_id, "pause"])
    assert paused.exit_code == 0, paused.output
    assert "Paused" in paused.output

    cancelled = runner.invoke(cli, ["profile", "downloads-control", "demo", job_id, "cancel"])
    assert cancelled.exit_code == 0, cancelled.output
    assert "Cancelled" in cancelled.output

    removed = runner.invoke(cli, ["profile", "downloads-control", "demo", job_id, "remove"])
    assert removed.exit_code == 0, removed.output

    downloads_result = runner.invoke(cli, ["profile", "downloads", "demo"])
    assert "No download jobs" in downloads_result.output


def _write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _valid_model_result(run_id="run1"):
    return {
        "schema_version": "0.1.0",
        "run_id": run_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "model": {"name": "Llama-3.1-8B-Instruct-GGUF", "class": "dense", "quantization": "Q4_K_M", "context_length": 4096},
        "config": {"compute_engine": "igpu", "backend": "llamacpp-vulkan", "os": "windows", "power_state": "plugged"},
        "environment": {
            "device_model": "Test Device", "cpu": "Test CPU", "memory_gb": 32,
            "os_version": "Windows 11", "driver_version": "1.0",
        },
        "metrics": {"prefill": {"tokens_per_sec": 100}, "decode": {"tokens_per_sec": 50}, "ttft_ms": 100, "peak_memory_gb": 8},
        "validity": {"valid": True, "warmup_discarded": True, "thermal_ok": True, "exclusive_run": True},
    }


def test_validate_submission_passes_a_conformant_result(tmp_path):
    _write_json(tmp_path / "demo" / "run1.json", _valid_model_result())

    runner = CliRunner()
    result = runner.invoke(cli, ["validate-submission", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "1 passed, 0 failed" in result.output


def test_validate_submission_fails_a_broken_result(tmp_path):
    _write_json(tmp_path / "demo" / "run1.json", {"schema_version": "0.1.0", "run_id": "run1"})

    runner = CliRunner()
    result = runner.invoke(cli, ["validate-submission", "--results-dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "FAIL" in result.output
    assert "1 passed, 1 failed" in result.output or "0 passed, 1 failed" in result.output


def test_validate_submission_uses_the_right_schema_per_modality_directory(tmp_path):
    classify_result = {
        "schema_version": "0.1.0",
        "run_type": "classify",
        "run_id": "run1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "model": {"name": "Phishing-Email-Detection-ONNX"},
        "config": {"compute_engine": "cpu", "backend": "onnxruntime-cpu", "os": "linux", "power_state": "plugged"},
        "environment": {"device_model": "Test", "cpu": "Test", "memory_gb": 32, "os_version": "Linux", "driver_version": "1.0"},
        "metrics": {"latency_ms": 10.0, "classifications_per_sec": 100.0, "trial_count": 5, "input_chars": 20},
        "validity": {"valid": True, "warmup_discarded": True, "exclusive_run": True},
    }
    _write_json(tmp_path / "demo" / "classify" / "run1.json", classify_result)
    # A model-shaped result would fail if checked against classify's schema
    # (or vice versa) -- this confirms the modality directory actually
    # selects the matching schema, not just "some schema or other".
    _write_json(tmp_path / "demo" / "run2.json", _valid_model_result("run2"))

    runner = CliRunner()
    result = runner.invoke(cli, ["validate-submission", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "2 passed, 0 failed" in result.output


def test_validate_submission_skips_failures_and_trials_sidecars(tmp_path):
    _write_json(tmp_path / "demo" / "run1.json", _valid_model_result())
    _write_json(tmp_path / "demo" / "failures" / "f1.json", {"anything": "goes"})
    _write_json(tmp_path / "demo" / "trials" / "run1.json", {"trials": [{"decode_tokens_per_sec": 1.0}]})

    runner = CliRunner()
    result = runner.invoke(cli, ["validate-submission", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "1 passed, 0 failed, 2 skipped" in result.output


def test_validate_submission_accepts_explicit_paths(tmp_path):
    good = tmp_path / "demo" / "run1.json"
    _write_json(good, _valid_model_result())

    runner = CliRunner()
    result = runner.invoke(cli, ["validate-submission", str(good)])
    assert result.exit_code == 0, result.output
    assert "1 passed, 0 failed" in result.output


def test_validate_submission_reports_nothing_to_check(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["validate-submission", "--results-dir", str(tmp_path / "empty")])
    assert result.exit_code == 0, result.output
    assert "No result JSON files found" in result.output
