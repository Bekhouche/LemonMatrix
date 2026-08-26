"""`lemonmatrix` command-line entry point."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import click
import jsonschema

from .bench import (AudioGenConfig, ClassifyConfig, EmbeddingsConfig, ImageGenConfig, MeshGenConfig,
                     RerankConfig, STTConfig, SweepConfig, TTSConfig, run_audiogen, run_classify,
                     run_embeddings, run_imagegen, run_meshgen, run_rerank, run_stt, run_sweep,
                     run_sweep_via_job, run_tts)
from .capabilities import (available_backends, available_compute_engines, available_routers,
                           compatible_engines_for_backend, engine_backend_compatible, host_os,
                           is_router_model, validate_combo_against_profile)
from .client import BENCH_TIMEOUT, LemonadeClient, job_progress_percent
from .discover import QUICK_SELECT_PORTS, expand_subnet, scan, scan_localhost
from .profile import DEFAULT_PORT, DEFAULT_PROFILE_DIR, Profile, build_url, connect_and_save
from .results_store import (list_audiogen_results, list_classify_results, list_embeddings_results,
                             list_imagegen_results, list_meshgen_results, list_rerank_results,
                             list_results, list_stt_results, list_tts_results, load_trials,
                             results_to_csv, save_audiogen_result, save_classify_result,
                             save_embeddings_result, save_imagegen_result, save_meshgen_result,
                             save_rerank_result, save_stt_result, save_trials, save_tts_result)
from .validate import (validate_audiogen_result, validate_classify_result, validate_embeddings_result,
                        validate_imagegen_result, validate_meshgen_result, validate_rerank_result,
                        validate_result, validate_stt_result, validate_tts_result)


def _save_profile(name: str, base_url: str, api_key: str | None, driver_version, igpu, dgpu) -> Profile:
    try:
        prof, gaps = connect_and_save(
            name, base_url, api_key, driver_version=driver_version, igpu_override=igpu, dgpu_override=dgpu
        )
    except ConnectionError as exc:
        raise click.ClickException(str(exc))

    click.echo(f"Saved profile '{prof.name}' -> {prof.path()}")
    click.echo(json.dumps(prof.environment, indent=2))
    if gaps:
        click.secho(
            f"Warning: could not auto-discover {', '.join(gaps)}; filled with \"unknown\". "
            "Re-run with the matching --override flag, or edit the profile file directly.",
            fg="yellow",
        )
    return prof


@click.group()
def cli() -> None:
    """LemonMatrix: benchmark Lemonade instances and emit comparable results."""


@cli.group()
def profile() -> None:
    """Manage saved profiles (one per Lemonade instance)."""


@profile.command("add")
@click.argument("name")
@click.option("--host", default="localhost", show_default=True, help="IP or hostname of the Lemonade instance.")
@click.option("--port", type=int, default=DEFAULT_PORT, show_default=True, help="Lemonade's default changed to 13305 in v10.1 (was 8000).")
@click.option("--token", "api_key", default=None, help="Bearer token, if the instance requires LEMONADE_API_KEY.")
@click.option("--scheme", type=click.Choice(["http", "https"]), default="http", show_default=True)
@click.option("--url", "url_override", default=None, help="Full base URL, overriding --host/--port/--scheme (e.g. for a cloud endpoint with a path prefix).")
@click.option("--driver-version", default=None, help="Override if not auto-discoverable.")
@click.option("--igpu", default=None, help="Override the auto-detected integrated GPU name.")
@click.option("--dgpu", default=None, help="Override the auto-detected discrete GPU name.")
def profile_add(
    name: str,
    host: str,
    port: int,
    api_key: str | None,
    scheme: str,
    url_override: str | None,
    driver_version: str | None,
    igpu: str | None,
    dgpu: str | None,
) -> None:
    """Connect to a Lemonade instance and save its auto-discovered fingerprint as NAME."""
    base_url = url_override or build_url(host, port, scheme)
    _save_profile(name, base_url, api_key, driver_version, igpu, dgpu)


@profile.command("detect")
@click.option("--subnet", default=None, help="CIDR range to also scan, e.g. 192.168.1.0/24 (opt-in; localhost is always scanned).")
@click.option("--ports", default=None, help="Comma-separated ports to probe. Defaults to Lemonade's quick-select list.")
@click.option("--timeout", type=float, default=0.5, show_default=True, help="Per-probe timeout in seconds.")
@click.option("--save", "save_name", default=None, help="If exactly one instance is found, save it under this profile name.")
@click.option("--token", "api_key", default=None, help="Bearer token to use when --save connects, if required.")
def profile_detect(subnet: str | None, ports: str | None, timeout: float, save_name: str | None, api_key: str | None) -> None:
    """Scan localhost (and optionally a subnet) for live Lemonade instances."""
    port_list = [int(p) for p in ports.split(",")] if ports else QUICK_SELECT_PORTS

    click.echo(f"Scanning 127.0.0.1 on ports {port_list}...")
    found = set(scan_localhost(ports=port_list, timeout=timeout))

    if subnet:
        try:
            hosts = expand_subnet(subnet)
        except ValueError as exc:
            raise click.ClickException(str(exc))
        click.echo(f"Scanning {len(hosts)} host(s) in {subnet} on {len(port_list)} port(s) -- this may take a moment...")
        found |= set(scan(hosts, ports=port_list, timeout=timeout))

    if not found:
        click.echo("No live Lemonade instances found.")
        return

    click.echo(f"Found {len(found)} instance(s):")
    for host, port in sorted(found):
        base_url = build_url(host, port, "http")
        client = LemonadeClient(base_url, api_key=api_key)
        try:
            health = client.health()
            info = client.system_info()
            label = f"{info.get('OEM System', 'unknown device')} (lemonade {health.get('version', '?')})"
        except Exception:
            label = "(couldn't read details)"
        click.echo(f"  {base_url}\t{label}")

    if save_name:
        if len(found) != 1:
            raise click.ClickException(
                f"--save requires exactly one match, found {len(found)}. "
                "Use `lemonmatrix profile add` with an explicit --host/--port instead."
            )
        host, port = next(iter(found))
        _save_profile(save_name, build_url(host, port, "http"), api_key, None, None, None)


@profile.command("list")
def profile_list() -> None:
    """List saved profiles."""
    profiles = Profile.list_all()
    if not profiles:
        click.echo(f"No profiles saved yet in {DEFAULT_PROFILE_DIR}.")
        return
    for prof in profiles:
        click.echo(f"{prof.name}\t{prof.base_url}\t{prof.environment.get('device_model', 'unknown')}")


@profile.command("show")
@click.argument("name")
def profile_show(name: str) -> None:
    """Print a saved profile's full environment fingerprint."""
    try:
        prof = Profile.load(name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{name}'. See `lemonmatrix profile list`.")
    click.echo(json.dumps(prof.to_dict(), indent=2))


@profile.command("refresh")
@click.argument("name")
@click.option("--driver-version", default=None, help="Override if not auto-discoverable.")
@click.option("--igpu", default=None, help="Override the auto-detected integrated GPU name.")
@click.option("--dgpu", default=None, help="Override the auto-detected discrete GPU name.")
def profile_refresh(name: str, driver_version: str | None, igpu: str | None, dgpu: str | None) -> None:
    """Re-connect to a saved profile's instance and update its environment fingerprint."""
    try:
        prof = Profile.load(name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{name}'. See `lemonmatrix profile list`.")
    _save_profile(name, prof.base_url, prof.api_key, driver_version, igpu, dgpu)


@profile.command("delete")
@click.argument("name")
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def profile_delete(name: str, yes: bool) -> None:
    """Delete a saved profile by name."""
    try:
        prof = Profile.load(name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{name}'. See `lemonmatrix profile list`.")

    if not yes:
        click.confirm(f"Delete profile '{name}' at {prof.path()}?", abort=True)

    prof.path().unlink()
    click.secho(f"Deleted profile '{name}'.", fg="green")


@profile.command("debug")
@click.argument("name")
def profile_debug(name: str) -> None:
    """Print the raw /api/v1/system-info and /api/v1/health responses for a saved profile.

    Use this when discover_environment's field-name guesses don't match a
    given Lemonade instance -- the output here is what profile.py's mapping
    needs to be corrected against.
    """
    try:
        prof = Profile.load(name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{name}'. See `lemonmatrix profile list`.")

    client = LemonadeClient(prof.base_url, api_key=prof.api_key)
    click.echo("--- /api/v1/system-info ---")
    click.echo(json.dumps(client.system_info(), indent=2))
    click.echo("--- /api/v1/health ---")
    click.echo(json.dumps(client.health(), indent=2))


def _load_profile_or_fail(name: str) -> Profile:
    try:
        return Profile.load(name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{name}'. See `lemonmatrix profile list`.")


@profile.command("install-backend")
@click.argument("name")
@click.argument("recipe")
@click.argument("backend_key")
def profile_install_backend(name: str, recipe: str, backend_key: str) -> None:
    """Install a backend engine (e.g. llamacpp vulkan) on a profile's Lemonade instance.

    \b
      lemonmatrix profile install-backend strix llamacpp rocm

    Blocks for the whole download (up to 30 minutes). See `lemonmatrix profile
    show NAME`'s system-info for which recipe/backend pairs are "installable".
    """
    prof = _load_profile_or_fail(name)
    client = LemonadeClient(prof.base_url, api_key=prof.api_key)
    try:
        client.install_backend(recipe, backend_key)
    except Exception as exc:
        raise click.ClickException(f"Install failed for {recipe}:{backend_key} -- {exc}")
    click.secho(f"Installed {recipe}:{backend_key} on '{name}'.", fg="green")


@profile.command("search-models")
@click.argument("name")
@click.argument("query")
@click.option("--source", default="huggingface", show_default=True)
@click.option("--limit", type=int, default=10, show_default=True)
@click.option("--format", "fmt", default="gguf", show_default=True)
def profile_search_models(name: str, query: str, source: str, limit: int, fmt: str) -> None:
    """Search a model registry (read-only, no download) through a profile's Lemonade instance."""
    prof = _load_profile_or_fail(name)
    client = LemonadeClient(prof.base_url, api_key=prof.api_key)
    try:
        results = client.search_registry(query, source=source, limit=limit, fmt=fmt).get("results", [])
    except Exception as exc:
        raise click.ClickException(f"Search failed: {exc}")

    if not results:
        click.echo("No results.")
        return
    for r in results:
        click.echo(f"{r.get('repository_id', '?')}\t{r.get('downloads', 0)} downloads\t{r.get('likes', 0)} likes")
    click.echo("\nUse `lemonmatrix profile pull-variants NAME REPOSITORY_ID` to see quantization variants.")


@profile.command("pull-variants")
@click.argument("name")
@click.argument("checkpoint")
def profile_pull_variants(name: str, checkpoint: str) -> None:
    """List quantization variants + sizes for a registry checkpoint (read-only, no download)."""
    prof = _load_profile_or_fail(name)
    client = LemonadeClient(prof.base_url, api_key=prof.api_key)
    try:
        data = client.pull_variants(checkpoint)
    except Exception as exc:
        raise click.ClickException(f"Couldn't fetch variants for '{checkpoint}': {exc}")

    for v in data.get("variants", []):
        size_gb = (v.get("size_bytes") or 0) / 1e9
        click.echo(f"{v['name']}\t{size_gb:.2f} GB\t{v.get('primary_file', '')}")
    click.echo(
        f"\nUse `lemonmatrix profile pull-model {name} MODEL_NAME --recipe {data.get('recipe', '')} "
        f"--checkpoint {checkpoint}:VARIANT` to download one."
    )


@profile.command("pull-model")
@click.argument("name")
@click.argument("model_name")
@click.option("--recipe", default=None, help="Required when registering a new model (not already in the model list).")
@click.option("--checkpoint", default=None,
              help="\"owner/repo:VARIANT\" form, required when registering a new model. "
                   "See `lemonmatrix profile pull-variants` for exact variant names.")
def profile_pull_model(name: str, model_name: str, recipe: str | None, checkpoint: str | None) -> None:
    """Start a background model download on a profile's Lemonade instance.

    Returns immediately; the download continues on the Lemonade server even
    if this command exits. Poll progress with `lemonmatrix profile downloads NAME`.
    """
    prof = _load_profile_or_fail(name)
    client = LemonadeClient(prof.base_url, api_key=prof.api_key)
    try:
        job = client.start_model_download(model_name, recipe=recipe, checkpoint=checkpoint)
    except Exception as exc:
        raise click.ClickException(f"Couldn't start download of '{model_name}': {exc}")
    click.secho(
        f"Started downloading '{model_name}' in the background (job {job.get('id', '?')}). "
        f"Poll with `lemonmatrix profile downloads {name}`.",
        fg="green",
    )


@profile.command("downloads")
@click.argument("name")
def profile_downloads(name: str) -> None:
    """List a profile's server-owned download jobs (any state), with progress."""
    prof = _load_profile_or_fail(name)
    client = LemonadeClient(prof.base_url, api_key=prof.api_key)
    try:
        jobs = client.list_downloads()
    except Exception as exc:
        raise click.ClickException(f"Couldn't list downloads: {exc}")

    if not jobs:
        click.echo("No download jobs.")
        return
    for job in jobs:
        percent = job_progress_percent(job)
        click.echo(f"{job.get('id', '?')}\t{job.get('status', '?')}\t{percent:.1f}%")


@profile.command("downloads-control")
@click.argument("name")
@click.argument("download_id")
@click.argument("action", type=click.Choice(["pause", "cancel", "remove"]))
def profile_downloads_control(name: str, download_id: str, action: str) -> None:
    """Pause, cancel, or remove one of a profile's download jobs."""
    prof = _load_profile_or_fail(name)
    client = LemonadeClient(prof.base_url, api_key=prof.api_key)
    try:
        client.control_download(download_id, action)
    except Exception as exc:
        raise click.ClickException(f"Couldn't {action} '{download_id}': {exc}")
    past_tense = {"pause": "Paused", "cancel": "Cancelled", "remove": "Removed"}[action]
    click.secho(f"{past_tense} '{download_id}'.", fg="green")


@cli.command("run")
@click.option("--profile", "profile_name", required=True, help="Saved profile name (see `lemonmatrix profile list`).")
@click.option("--model", "model_name", required=True, help="Model or router name as registered in Lemonade.")
@click.option("--run-type", "run_type", type=click.Choice(["model", "router"]), default="model", show_default=True,
              help="'router' benchmarks a Lemonade collection.router; engine/backend/quant/class are auto-set.")
# Model-only options (required for model runs; ignored / auto-set for router runs)
@click.option("--model-class", type=click.Choice(["dense", "moe", "router"]), default=None,
              help="Model architecture class. Auto-set to 'router' when --run-type router.")
@click.option("--quant", "quantization", default=None,
              help="Exact quant string, e.g. Q4_K_M. Auto-set to 'none' for router runs.")
@click.option("--context-length", type=int, required=True)
@click.option("--engine", "compute_engine", type=click.Choice(["cpu", "igpu", "dgpu", "npu", "hybrid", "router"]),
              default=None, help="Compute engine. Auto-set to 'router' for --run-type router.")
@click.option("--backend", default=None,
              help="e.g. llamacpp-vulkan, llamacpp-rocm. Auto-set to 'collection.router' for router runs.")
@click.option("--power-state", type=click.Choice(["plugged", "battery", "power_capped"]), required=True)
@click.option("--power-cap-w", type=float, default=None)
@click.option("--warmup", "warmup_trials", type=int, default=2, show_default=True)
@click.option("--trials", "measured_trials", type=int, default=5, show_default=True)
@click.option("--max-tokens", type=int, default=256, show_default=True)
@click.option("--prompt-file", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--exclusive/--not-exclusive", "exclusive_run", default=True,
              help="Whether a competing GPU/NPU workload ran during measurement.")
@click.option("--parameters-b", type=float, default=None)
@click.option("--active-parameters-b", type=float, default=None)
@click.option("--price-per-kwh", "energy_price_usd_per_kwh", type=float, default=None)
@click.option("--hardware-cost-usd", type=float, default=None, help="Purchase price of the hardware, for the cost model's amortized-hardware component.")
@click.option("--hardware-lifetime-hours", type=float, default=None, help="Expected service life in hours, for the cost model's amortized-hardware component.")
@click.option("--via-job-engine", is_flag=True, default=False,
              help="Execute as a single durable Lemonade job (POST /v1/jobs) instead of N direct HTTP calls from this process -- survives this process disconnecting mid-run, since Lemonade itself owns and persists the job. Model runs only (not --run-type router).")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--force", is_flag=True, default=False,
              help="Bypass engine/backend compatibility checks and run anyway (result will be marked invalid).")
def run(
    profile_name: str,
    model_name: str,
    run_type: str,
    model_class: str | None,
    quantization: str | None,
    context_length: int,
    compute_engine: str | None,
    backend: str | None,
    power_state: str,
    power_cap_w: float | None,
    warmup_trials: int,
    measured_trials: int,
    max_tokens: int,
    prompt_file: str | None,
    exclusive_run: bool,
    parameters_b: float | None,
    active_parameters_b: float | None,
    energy_price_usd_per_kwh: float | None,
    hardware_cost_usd: float | None,
    hardware_lifetime_hours: float | None,
    via_job_engine: bool,
    out_dir: str,
    force: bool,
) -> None:
    """Run one benchmark against a saved profile and write a result JSON.

    \b
    Model run (default):
      lemonmatrix run --profile strix \\
        --model Llama-3.1-8B-Instruct-GGUF --model-class dense --quant Q4_K_M \\
        --context-length 4096 --engine igpu --backend llamacpp-vulkan --power-state plugged

    \b
    Router run:
      lemonmatrix run --profile strix --run-type router \\
        --model my-collection-router \\
        --context-length 4096 --power-state plugged

    For router runs, --model-class, --quant, --engine, and --backend are
    auto-set (router / none / router / collection.router).  Route decisions
    are captured per trial in the sidecar; timing is recorded just like a
    model run.
    """
    try:
        prof = Profile.load(profile_name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{profile_name}'. See `lemonmatrix profile list`.")

    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)

    # OS is a fixed fact of the profile (IDEA.md: a different OS is a
    # different profile, not a per-run setting), so it is always derived
    # from the profile's own discovered environment -- there is no --os flag.
    run_os = host_os(prof.environment)
    if not run_os:
        raise click.ClickException(
            f"Could not determine profile '{profile_name}'s OS from its environment. "
            "Re-run `lemonmatrix profile add` to refresh discovery, or edit the profile file directly."
        )

    if run_type == "router":
        # Router runs: auto-fill everything that doesn't apply to a router.
        model_class = model_class or "router"
        quantization = quantization or "none"
        compute_engine = compute_engine or "router"
        backend = backend or "collection.router"
    else:
        # Model runs: validate required flags that Click can't make conditional.
        missing = [name for name, val in [("--model-class", model_class), ("--quant", quantization),
                                          ("--engine", compute_engine), ("--backend", backend)] if val is None]
        if missing:
            raise click.UsageError(f"Missing required option(s) for a model run: {', '.join(missing)}")

        # Live profile validation: engine/backend logic + hardware presence + backend installed.
        try:
            system_info = client.system_info()
        except Exception:
            system_info = {}

        issues = validate_combo_against_profile(compute_engine, backend, system_info, prof.environment)
        if issues:
            click.secho("Configuration problem(s) detected:", fg="red", bold=True)
            for issue in issues:
                for line in issue.splitlines():
                    click.secho(f"  {line}", fg="red")
            if force:
                click.secho("--force supplied: proceeding anyway. Result will be marked invalid.", fg="yellow")
            else:
                # Offer compatible engines as a hint.
                good_engines = compatible_engines_for_backend(backend)
                if good_engines:
                    click.secho(
                        f"\nCompatible engines for '{backend}': {', '.join(good_engines)}",
                        fg="cyan",
                    )
                raise click.ClickException(
                    "Aborting. Fix the combination above or pass --force to run anyway."
                )

    cfg = SweepConfig(
        model_name=model_name,
        model_class=model_class,
        quantization=quantization,
        context_length=context_length,
        compute_engine=compute_engine,
        backend=backend,
        os=run_os,
        power_state=power_state,
        power_cap_w=power_cap_w,
        warmup_trials=warmup_trials,
        measured_trials=measured_trials,
        max_tokens=max_tokens,
        exclusive_run=exclusive_run,
        parameters_b=parameters_b,
        active_parameters_b=active_parameters_b,
        energy_price_usd_per_kwh=energy_price_usd_per_kwh,
        hardware_cost_usd=hardware_cost_usd,
        hardware_lifetime_hours=hardware_lifetime_hours,
        run_type=run_type,
    )
    if prompt_file:
        cfg.prompt = Path(prompt_file).read_text()

    if via_job_engine and run_type == "router":
        raise click.UsageError("--via-job-engine does not support --run-type router (a router has no fixed backend/ctx_size for a job's load step).")

    click.echo(f"Running {measured_trials} trial(s) ({warmup_trials} warmup) of {model_name} on '{profile_name}'...")
    if via_job_engine:
        click.echo("(executing as a single durable Lemonade job -- POST /v1/jobs)")
        result, raw_trials = run_sweep_via_job(client, prof.environment, cfg)
    else:
        result, raw_trials = run_sweep(client, prof.environment, cfg)

    try:
        validate_result(result)
    except jsonschema.ValidationError as exc:
        click.secho("Result failed schema validation, not writing it out:", fg="red")
        raise click.ClickException(str(exc)) from exc

    out_path = Path(out_dir) / profile_name
    out_path.mkdir(parents=True, exist_ok=True)
    result_path = out_path / f"{result['run_id']}.json"
    result_path.write_text(json.dumps(result, indent=2))
    save_trials(out_dir, profile_name, result["run_id"], raw_trials)

    click.secho(f"Wrote {result_path}", fg="green")
    click.echo(
        f"decode: {result['metrics']['decode']['tokens_per_sec']:.1f} tok/s   "
        f"prefill: {result['metrics']['prefill']['tokens_per_sec'] or 0:.1f} tok/s   "
        f"ttft: {result['metrics']['ttft_ms']:.0f} ms"
    )
    stddev = result["metrics"]["decode"].get("stddev")
    if stddev is not None:
        click.echo(f"  decode stddev: {stddev:.2f} tok/s   p95: {result['metrics']['decode'].get('p95', 0):.1f} tok/s")
    if not result["validity"]["valid"]:
        click.secho(f"Marked invalid: {result['validity'].get('notes', 'see validity block')}", fg="yellow")


@cli.command("classify")
@click.option("--profile", "profile_name", required=True, help="Saved profile name (see `lemonmatrix profile list`).")
@click.option("--model", "model_name", required=True, help="Classification model name as registered in Lemonade (onnxruntime recipe).")
@click.option("--engine", "compute_engine", type=click.Choice(["cpu", "igpu", "dgpu", "npu"]), required=True)
@click.option("--backend", required=True, help="e.g. onnxruntime-cpu.")
@click.option("--power-state", type=click.Choice(["plugged", "battery", "power_capped"]), required=True)
@click.option("--power-cap-w", type=float, default=None)
@click.option("--input-text", default=None, help="Text to classify on every trial. Mutually exclusive with --input-file.")
@click.option("--input-file", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--top-k", type=int, default=None)
@click.option("--warmup", "warmup_trials", type=int, default=2, show_default=True)
@click.option("--trials", "measured_trials", type=int, default=5, show_default=True)
@click.option("--exclusive/--not-exclusive", "exclusive_run", default=True,
              help="Whether a competing workload ran during measurement.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="results", show_default=True)
def classify(
    profile_name: str,
    model_name: str,
    compute_engine: str,
    backend: str,
    power_state: str,
    power_cap_w: float | None,
    input_text: str | None,
    input_file: str | None,
    top_k: int | None,
    warmup_trials: int,
    measured_trials: int,
    exclusive_run: bool,
    out_dir: str,
) -> None:
    """Benchmark an ONNX text-classifier's latency and throughput.

    Deliberately separate from `run`/`sweep`: classification latency isn't
    comparable to LLM token throughput, so results are written to their own
    results/<profile>/classify/ tree and never appear on the model/router
    leaderboard. See `lemonmatrix classify-results` to list them.

    \b
      lemonmatrix classify --profile strix --model Phishing-Email-Detection-ONNX \\
        --engine cpu --backend onnxruntime-cpu --power-state plugged \\
        --input-text "Please verify your account now."
    """
    try:
        prof = Profile.load(profile_name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{profile_name}'. See `lemonmatrix profile list`.")

    if input_file and input_text:
        raise click.UsageError("Pass either --input-text or --input-file, not both.")
    if input_file:
        input_text = Path(input_file).read_text()
    if not input_text:
        raise click.UsageError("Missing required option: --input-text or --input-file.")

    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)

    # OS is a fixed fact of the profile, same rule as `run` -- there is no --os flag.
    run_os = host_os(prof.environment)
    if not run_os:
        raise click.ClickException(
            f"Could not determine profile '{profile_name}'s OS from its environment. "
            "Re-run `lemonmatrix profile add` to refresh discovery, or edit the profile file directly."
        )

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}
    issues = validate_combo_against_profile(compute_engine, backend, system_info, prof.environment, modality="Text classification")
    if issues:
        click.secho("Configuration problem(s) detected:", fg="red", bold=True)
        for issue in issues:
            for line in issue.splitlines():
                click.secho(f"  {line}", fg="red")
        raise click.ClickException("Aborting. Fix the engine/backend combination before running.")

    cfg = ClassifyConfig(
        model_name=model_name,
        compute_engine=compute_engine,
        backend=backend,
        os=run_os,
        power_state=power_state,
        input_text=input_text,
        power_cap_w=power_cap_w,
        warmup_trials=warmup_trials,
        measured_trials=measured_trials,
        top_k=top_k,
        exclusive_run=exclusive_run,
    )

    click.echo(f"Running {measured_trials} classify trial(s) ({warmup_trials} warmup) of {model_name} on '{profile_name}'...")
    result, _raw_trials = run_classify(client, prof.environment, cfg)

    try:
        validate_classify_result(result)
    except jsonschema.ValidationError as exc:
        click.secho("Result failed schema validation, not writing it out:", fg="red")
        raise click.ClickException(str(exc)) from exc

    result_path = save_classify_result(out_dir, profile_name, result)
    click.secho(f"Wrote {result_path}", fg="green")
    click.echo(
        f"latency: {result['metrics']['latency_ms']:.1f} ms   "
        f"throughput: {result['metrics']['classifications_per_sec']:.2f} classifications/sec"
    )
    if not result["validity"]["valid"]:
        click.secho(f"Marked invalid: {result['validity'].get('notes', 'see validity block')}", fg="yellow")


@cli.command("classify-results")
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--profile", "profile_name", default=None, help="Filter to a single profile.")
def classify_results(results_dir: str, profile_name: str | None) -> None:
    """List saved classification benchmark results."""
    results = list_classify_results(results_dir, profile=profile_name)
    if not results:
        click.echo("No classify results found.", err=True)
        return
    for r in results:
        flag = "" if r["validity"]["valid"] else "  [INVALID]"
        click.echo(
            f"{r['timestamp']}  {r['_profile']:<20} {r['model']['name']:<30} "
            f"{r['metrics']['latency_ms']:.1f} ms  "
            f"{r['metrics']['classifications_per_sec']:.2f}/sec{flag}"
        )


@cli.command("tts")
@click.option("--profile", "profile_name", required=True, help="Saved profile name (see `lemonmatrix profile list`).")
@click.option("--model", "model_name", required=True, help="TTS model name as registered in Lemonade (kokoro/openmoss recipe).")
@click.option("--engine", "compute_engine", type=click.Choice(["cpu", "igpu", "dgpu", "npu"]), required=True)
@click.option("--backend", required=True, help="e.g. kokoro-cpu.")
@click.option("--power-state", type=click.Choice(["plugged", "battery", "power_capped"]), required=True)
@click.option("--power-cap-w", type=float, default=None)
@click.option("--input-text", default=None, help="Text to speak on every trial. Mutually exclusive with --input-file.")
@click.option("--input-file", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--voice", default=None)
@click.option("--speed", type=float, default=None)
@click.option("--warmup", "warmup_trials", type=int, default=2, show_default=True)
@click.option("--trials", "measured_trials", type=int, default=5, show_default=True)
@click.option("--exclusive/--not-exclusive", "exclusive_run", default=True,
              help="Whether a competing workload ran during measurement.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="results", show_default=True)
def tts(
    profile_name: str,
    model_name: str,
    compute_engine: str,
    backend: str,
    power_state: str,
    power_cap_w: float | None,
    input_text: str | None,
    input_file: str | None,
    voice: str | None,
    speed: float | None,
    warmup_trials: int,
    measured_trials: int,
    exclusive_run: bool,
    out_dir: str,
) -> None:
    """Benchmark a text-to-speech model's real-time-factor.

    Deliberately separate from `run`/`sweep`: real-time-factor isn't
    comparable to LLM token throughput, so results are written to their own
    results/<profile>/tts/ tree and never appear on the model/router
    leaderboard. See `lemonmatrix tts-results` to list them.

    \b
      lemonmatrix tts --profile strix --model kokoro-v1 \\
        --engine cpu --backend kokoro-cpu --power-state plugged \\
        --input-text "Lemonade can speak"
    """
    try:
        prof = Profile.load(profile_name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{profile_name}'. See `lemonmatrix profile list`.")

    if input_file and input_text:
        raise click.UsageError("Pass either --input-text or --input-file, not both.")
    if input_file:
        input_text = Path(input_file).read_text()
    if not input_text:
        raise click.UsageError("Missing required option: --input-text or --input-file.")

    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)

    # OS is a fixed fact of the profile, same rule as `run` -- there is no --os flag.
    run_os = host_os(prof.environment)
    if not run_os:
        raise click.ClickException(
            f"Could not determine profile '{profile_name}'s OS from its environment. "
            "Re-run `lemonmatrix profile add` to refresh discovery, or edit the profile file directly."
        )

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}
    issues = validate_combo_against_profile(compute_engine, backend, system_info, prof.environment, modality="Text-to-speech")
    if issues:
        click.secho("Configuration problem(s) detected:", fg="red", bold=True)
        for issue in issues:
            for line in issue.splitlines():
                click.secho(f"  {line}", fg="red")
        raise click.ClickException("Aborting. Fix the engine/backend combination before running.")

    cfg = TTSConfig(
        model_name=model_name,
        compute_engine=compute_engine,
        backend=backend,
        os=run_os,
        power_state=power_state,
        input_text=input_text,
        power_cap_w=power_cap_w,
        warmup_trials=warmup_trials,
        measured_trials=measured_trials,
        voice=voice,
        speed=speed,
        exclusive_run=exclusive_run,
    )

    click.echo(f"Running {measured_trials} TTS trial(s) ({warmup_trials} warmup) of {model_name} on '{profile_name}'...")
    result, _raw_trials = run_tts(client, prof.environment, cfg)

    try:
        validate_tts_result(result)
    except jsonschema.ValidationError as exc:
        click.secho("Result failed schema validation, not writing it out:", fg="red")
        raise click.ClickException(str(exc)) from exc

    result_path = save_tts_result(out_dir, profile_name, result)
    click.secho(f"Wrote {result_path}", fg="green")
    click.echo(
        f"generation time: {result['metrics']['generation_time_ms']:.1f} ms   "
        f"audio: {result['metrics']['audio_duration_s']:.2f}s   "
        f"RTF: {result['metrics']['real_time_factor']:.2f}x"
    )
    if not result["validity"]["valid"]:
        click.secho(f"Marked invalid: {result['validity'].get('notes', 'see validity block')}", fg="yellow")


@cli.command("tts-results")
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--profile", "profile_name", default=None, help="Filter to a single profile.")
def tts_results(results_dir: str, profile_name: str | None) -> None:
    """List saved text-to-speech benchmark results."""
    results = list_tts_results(results_dir, profile=profile_name)
    if not results:
        click.echo("No TTS results found.", err=True)
        return
    for r in results:
        flag = "" if r["validity"]["valid"] else "  [INVALID]"
        click.echo(
            f"{r['timestamp']}  {r['_profile']:<20} {r['model']['name']:<30} "
            f"RTF {r['metrics']['real_time_factor']:.2f}x{flag}"
        )


@cli.command("stt")
@click.option("--profile", "profile_name", required=True, help="Saved profile name (see `lemonmatrix profile list`).")
@click.option("--model", "model_name", required=True, help="STT model name as registered in Lemonade (whispercpp/moonshine recipe).")
@click.option("--engine", "compute_engine", type=click.Choice(["cpu", "igpu", "dgpu", "npu"]), required=True)
@click.option("--backend", required=True, help="e.g. whispercpp-cpu.")
@click.option("--power-state", type=click.Choice(["plugged", "battery", "power_capped"]), required=True)
@click.option("--power-cap-w", type=float, default=None)
@click.option("--audio-file", type=click.Path(exists=True, dir_okay=False), required=True,
              help="A WAV file -- its exact duration (read from its header) is what real_time_factor is computed against.")
@click.option("--language", default=None)
@click.option("--warmup", "warmup_trials", type=int, default=2, show_default=True)
@click.option("--trials", "measured_trials", type=int, default=5, show_default=True)
@click.option("--exclusive/--not-exclusive", "exclusive_run", default=True,
              help="Whether a competing workload ran during measurement.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="results", show_default=True)
def stt(
    profile_name: str,
    model_name: str,
    compute_engine: str,
    backend: str,
    power_state: str,
    power_cap_w: float | None,
    audio_file: str,
    language: str | None,
    warmup_trials: int,
    measured_trials: int,
    exclusive_run: bool,
    out_dir: str,
) -> None:
    """Benchmark a speech-to-text model's real-time-factor.

    Deliberately separate from `run`/`sweep`: real-time-factor isn't
    comparable to LLM token throughput, so results are written to their own
    results/<profile>/stt/ tree and never appear on the model/router
    leaderboard. See `lemonmatrix stt-results` to list them.

    \b
      lemonmatrix stt --profile strix --model Whisper-Tiny \\
        --engine cpu --backend whispercpp-cpu --power-state plugged \\
        --audio-file sample.wav
    """
    try:
        prof = Profile.load(profile_name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{profile_name}'. See `lemonmatrix profile list`.")

    audio_bytes = Path(audio_file).read_bytes()

    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)

    run_os = host_os(prof.environment)
    if not run_os:
        raise click.ClickException(
            f"Could not determine profile '{profile_name}'s OS from its environment. "
            "Re-run `lemonmatrix profile add` to refresh discovery, or edit the profile file directly."
        )

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}
    issues = validate_combo_against_profile(compute_engine, backend, system_info, prof.environment, modality="Speech-to-text")
    if issues:
        click.secho("Configuration problem(s) detected:", fg="red", bold=True)
        for issue in issues:
            for line in issue.splitlines():
                click.secho(f"  {line}", fg="red")
        raise click.ClickException("Aborting. Fix the engine/backend combination before running.")

    cfg = STTConfig(
        model_name=model_name,
        compute_engine=compute_engine,
        backend=backend,
        os=run_os,
        power_state=power_state,
        audio_bytes=audio_bytes,
        audio_filename=Path(audio_file).name,
        language=language,
        power_cap_w=power_cap_w,
        warmup_trials=warmup_trials,
        measured_trials=measured_trials,
        exclusive_run=exclusive_run,
    )

    click.echo(f"Running {measured_trials} STT trial(s) ({warmup_trials} warmup) of {model_name} on '{profile_name}'...")
    result, _raw_trials = run_stt(client, prof.environment, cfg)

    try:
        validate_stt_result(result)
    except jsonschema.ValidationError as exc:
        click.secho("Result failed schema validation, not writing it out:", fg="red")
        raise click.ClickException(str(exc)) from exc

    result_path = save_stt_result(out_dir, profile_name, result)
    click.secho(f"Wrote {result_path}", fg="green")
    click.echo(
        f"transcription time: {result['metrics']['transcription_time_ms']:.1f} ms   "
        f"audio: {result['metrics']['audio_duration_s']:.2f}s   "
        f"RTF: {result['metrics']['real_time_factor']:.2f}x"
    )
    if not result["validity"]["valid"]:
        click.secho(f"Marked invalid: {result['validity'].get('notes', 'see validity block')}", fg="yellow")


@cli.command("stt-results")
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--profile", "profile_name", default=None, help="Filter to a single profile.")
def stt_results(results_dir: str, profile_name: str | None) -> None:
    """List saved speech-to-text benchmark results."""
    results = list_stt_results(results_dir, profile=profile_name)
    if not results:
        click.echo("No STT results found.", err=True)
        return
    for r in results:
        flag = "" if r["validity"]["valid"] else "  [INVALID]"
        click.echo(
            f"{r['timestamp']}  {r['_profile']:<20} {r['model']['name']:<30} "
            f"RTF {r['metrics']['real_time_factor']:.2f}x{flag}"
        )


@cli.command("imagegen")
@click.option("--profile", "profile_name", required=True, help="Saved profile name (see `lemonmatrix profile list`).")
@click.option("--model", "model_name", required=True, help="Image-generation model name as registered in Lemonade (sd-cpp/thenoise recipe).")
@click.option("--engine", "compute_engine", type=click.Choice(["cpu", "igpu", "dgpu", "npu"]), required=True)
@click.option("--backend", required=True, help="e.g. sd-cpp-cuda.")
@click.option("--power-state", type=click.Choice(["plugged", "battery", "power_capped"]), required=True)
@click.option("--power-cap-w", type=float, default=None)
@click.option("--operation", type=click.Choice(["generate", "edit", "variation"]), default="generate", show_default=True,
              help="'generate' is text-to-image; 'edit' is a prompt-guided edit of --input-image; 'variation' is an unguided variation of --input-image (no prompt/steps/cfg-scale/seed -- Lemonade's own endpoint doesn't accept them).")
@click.option("--prompt", default=None, help="Required for generate/edit, unused for variation.")
@click.option("--input-image", type=click.Path(exists=True, dir_okay=False), default=None, help="Required for edit/variation.")
@click.option("--mask-image", type=click.Path(exists=True, dir_okay=False), default=None, help="Optional, edit only.")
@click.option("--size", "image_size", default="512x512", show_default=True)
@click.option("--steps", type=int, default=4, show_default=True, help="Diffusion step count -- materially affects generation time. Not used for variation.")
@click.option("--cfg-scale", type=float, default=None)
@click.option("--seed", type=int, default=None)
@click.option("--warmup", "warmup_trials", type=int, default=2, show_default=True)
@click.option("--trials", "measured_trials", type=int, default=5, show_default=True)
@click.option("--exclusive/--not-exclusive", "exclusive_run", default=True,
              help="Whether a competing workload ran during measurement.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="results", show_default=True)
def imagegen(
    profile_name: str,
    model_name: str,
    compute_engine: str,
    backend: str,
    power_state: str,
    power_cap_w: float | None,
    operation: str,
    prompt: str | None,
    input_image: str | None,
    mask_image: str | None,
    image_size: str,
    steps: int,
    cfg_scale: float | None,
    seed: int | None,
    warmup_trials: int,
    measured_trials: int,
    exclusive_run: bool,
    out_dir: str,
) -> None:
    """Benchmark an image model's throughput: generation (images/sec),
    prompt-guided editing, or unguided variation of an input image.

    Deliberately separate from `run`/`sweep`: images/sec isn't comparable to
    LLM token throughput, so results are written to their own
    results/<profile>/imagegen/ tree and never appear on the model/router
    leaderboard. See `lemonmatrix imagegen-results` to list them. Image
    upscale is not supported -- Lemonade shells it out to a CLI subprocess
    per request rather than a persistent loaded model, so there's nothing
    for this tool's exclusivity/reload-freedom checks to verify against.

    \b
      lemonmatrix imagegen --profile strix --model SD-Turbo \\
        --engine dgpu --backend sd-cpp-cuda --power-state plugged \\
        --prompt "A red circle" --size 256x256 --steps 2

    \b
      lemonmatrix imagegen --profile strix --model SD-Turbo --operation edit \\
        --engine dgpu --backend sd-cpp-cuda --power-state plugged \\
        --prompt "add a hat" --input-image cat.png --size 256x256
    """
    try:
        prof = Profile.load(profile_name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{profile_name}'. See `lemonmatrix profile list`.")

    if operation in ("generate", "edit") and not prompt:
        raise click.UsageError(f"--prompt is required for --operation {operation}.")
    if operation in ("edit", "variation") and not input_image:
        raise click.UsageError(f"--input-image is required for --operation {operation}.")
    if mask_image and operation != "edit":
        raise click.UsageError("--mask-image is only valid with --operation edit.")

    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)

    run_os = host_os(prof.environment)
    if not run_os:
        raise click.ClickException(
            f"Could not determine profile '{profile_name}'s OS from its environment. "
            "Re-run `lemonmatrix profile add` to refresh discovery, or edit the profile file directly."
        )

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}
    issues = validate_combo_against_profile(compute_engine, backend, system_info, prof.environment, modality="Image generation")
    if issues:
        click.secho("Configuration problem(s) detected:", fg="red", bold=True)
        for issue in issues:
            for line in issue.splitlines():
                click.secho(f"  {line}", fg="red")
        raise click.ClickException("Aborting. Fix the engine/backend combination before running.")

    cfg = ImageGenConfig(
        model_name=model_name,
        compute_engine=compute_engine,
        backend=backend,
        os=run_os,
        power_state=power_state,
        operation=operation,
        prompt=prompt or "",
        input_image_bytes=Path(input_image).read_bytes() if input_image else None,
        input_image_filename=Path(input_image).name if input_image else "input.png",
        mask_bytes=Path(mask_image).read_bytes() if mask_image else None,
        mask_filename=Path(mask_image).name if mask_image else "mask.png",
        image_size=image_size,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        power_cap_w=power_cap_w,
        warmup_trials=warmup_trials,
        measured_trials=measured_trials,
        exclusive_run=exclusive_run,
    )

    click.echo(f"Running {measured_trials} image-gen trial(s) ({warmup_trials} warmup) of {model_name} on '{profile_name}'...")
    result, _raw_trials = run_imagegen(client, prof.environment, cfg)

    try:
        validate_imagegen_result(result)
    except jsonschema.ValidationError as exc:
        click.secho("Result failed schema validation, not writing it out:", fg="red")
        raise click.ClickException(str(exc)) from exc

    result_path = save_imagegen_result(out_dir, profile_name, result)
    click.secho(f"Wrote {result_path}", fg="green")
    click.echo(
        f"generation time: {result['metrics']['generation_time_ms']:.1f} ms   "
        f"throughput: {result['metrics']['images_per_sec']:.2f} images/sec"
    )
    if not result["validity"]["valid"]:
        click.secho(f"Marked invalid: {result['validity'].get('notes', 'see validity block')}", fg="yellow")


@cli.command("imagegen-results")
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--profile", "profile_name", default=None, help="Filter to a single profile.")
def imagegen_results(results_dir: str, profile_name: str | None) -> None:
    """List saved image-generation benchmark results."""
    results = list_imagegen_results(results_dir, profile=profile_name)
    if not results:
        click.echo("No image-generation results found.", err=True)
        return
    for r in results:
        flag = "" if r["validity"]["valid"] else "  [INVALID]"
        click.echo(
            f"{r['timestamp']}  {r['_profile']:<20} {r['model']['name']:<30} "
            f"{r['metrics']['images_per_sec']:.2f} img/sec{flag}"
        )


@cli.command("audiogen")
@click.option("--profile", "profile_name", required=True, help="Saved profile name (see `lemonmatrix profile list`).")
@click.option("--model", "model_name", required=True, help="Audio-generation model name as registered in Lemonade (acestep/thinksound recipe).")
@click.option("--engine", "compute_engine", type=click.Choice(["cpu", "igpu", "dgpu", "npu"]), required=True)
@click.option("--backend", required=True, help="e.g. acestep-cuda.")
@click.option("--power-state", type=click.Choice(["plugged", "battery", "power_capped"]), required=True)
@click.option("--power-cap-w", type=float, default=None)
@click.option("--prompt", required=True, help="Description of the audio to generate (music, sound effects) -- not speech.")
@click.option("--lyrics", default=None)
@click.option("--vocal-language", default=None)
@click.option("--warmup", "warmup_trials", type=int, default=2, show_default=True)
@click.option("--trials", "measured_trials", type=int, default=5, show_default=True)
@click.option("--exclusive/--not-exclusive", "exclusive_run", default=True,
              help="Whether a competing workload ran during measurement.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="results", show_default=True)
def audiogen(
    profile_name: str,
    model_name: str,
    compute_engine: str,
    backend: str,
    power_state: str,
    power_cap_w: float | None,
    prompt: str,
    lyrics: str | None,
    vocal_language: str | None,
    warmup_trials: int,
    measured_trials: int,
    exclusive_run: bool,
    out_dir: str,
) -> None:
    """Benchmark an audio-generation model's real-time-factor.

    Text/music/sound-effect generation (acestep/thinksound) -- not speech;
    see `lemonmatrix tts` for text-to-speech. Deliberately separate from
    `run`/`sweep`/`tts`: real-time-factor isn't comparable to LLM token
    throughput, and generating music is a different task with a different
    compute profile than speech, so results are written to their own
    results/<profile>/audiogen/ tree and never appear on the model/router
    leaderboard or mixed with TTS results. See `lemonmatrix audiogen-results`
    to list them.

    \b
      lemonmatrix audiogen --profile strix --model ACE-Step-v1 \\
        --engine dgpu --backend acestep-cuda --power-state plugged \\
        --prompt "An upbeat acoustic guitar riff"
    """
    try:
        prof = Profile.load(profile_name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{profile_name}'. See `lemonmatrix profile list`.")

    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)

    run_os = host_os(prof.environment)
    if not run_os:
        raise click.ClickException(
            f"Could not determine profile '{profile_name}'s OS from its environment. "
            "Re-run `lemonmatrix profile add` to refresh discovery, or edit the profile file directly."
        )

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}
    issues = validate_combo_against_profile(compute_engine, backend, system_info, prof.environment, modality="Audio generation")
    if issues:
        click.secho("Configuration problem(s) detected:", fg="red", bold=True)
        for issue in issues:
            for line in issue.splitlines():
                click.secho(f"  {line}", fg="red")
        raise click.ClickException("Aborting. Fix the engine/backend combination before running.")

    cfg = AudioGenConfig(
        model_name=model_name,
        compute_engine=compute_engine,
        backend=backend,
        os=run_os,
        power_state=power_state,
        prompt=prompt,
        lyrics=lyrics,
        vocal_language=vocal_language,
        power_cap_w=power_cap_w,
        warmup_trials=warmup_trials,
        measured_trials=measured_trials,
        exclusive_run=exclusive_run,
    )

    click.echo(f"Running {measured_trials} audio-gen trial(s) ({warmup_trials} warmup) of {model_name} on '{profile_name}'...")
    result, _raw_trials = run_audiogen(client, prof.environment, cfg)

    try:
        validate_audiogen_result(result)
    except jsonschema.ValidationError as exc:
        click.secho("Result failed schema validation, not writing it out:", fg="red")
        raise click.ClickException(str(exc)) from exc

    result_path = save_audiogen_result(out_dir, profile_name, result)
    click.secho(f"Wrote {result_path}", fg="green")
    click.echo(
        f"generation time: {result['metrics']['generation_time_ms']:.1f} ms   "
        f"audio: {result['metrics']['audio_duration_s']:.2f}s   "
        f"RTF: {result['metrics']['real_time_factor']:.2f}x"
    )
    if not result["validity"]["valid"]:
        click.secho(f"Marked invalid: {result['validity'].get('notes', 'see validity block')}", fg="yellow")


@cli.command("audiogen-results")
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--profile", "profile_name", default=None, help="Filter to a single profile.")
def audiogen_results(results_dir: str, profile_name: str | None) -> None:
    """List saved audio-generation benchmark results."""
    results = list_audiogen_results(results_dir, profile=profile_name)
    if not results:
        click.echo("No audio-generation results found.", err=True)
        return
    for r in results:
        flag = "" if r["validity"]["valid"] else "  [INVALID]"
        click.echo(
            f"{r['timestamp']}  {r['_profile']:<20} {r['model']['name']:<30} "
            f"RTF {r['metrics']['real_time_factor']:.2f}x{flag}"
        )


@cli.command("embeddings")
@click.option("--profile", "profile_name", required=True, help="Saved profile name (see `lemonmatrix profile list`).")
@click.option("--model", "model_name", required=True, help="Embeddings model name as registered in Lemonade (llamacpp GGUF or FastFlowLM recipe).")
@click.option("--engine", "compute_engine", type=click.Choice(["cpu", "igpu", "dgpu", "npu"]), required=True)
@click.option("--backend", required=True, help="e.g. llamacpp-vulkan, flm-npu.")
@click.option("--power-state", type=click.Choice(["plugged", "battery", "power_capped"]), required=True)
@click.option("--power-cap-w", type=float, default=None)
@click.option("--input", "input_texts", multiple=True, required=True, help="Text to embed (repeat for a larger batch).")
@click.option("--warmup", "warmup_trials", type=int, default=2, show_default=True)
@click.option("--trials", "measured_trials", type=int, default=5, show_default=True)
@click.option("--exclusive/--not-exclusive", "exclusive_run", default=True,
              help="Whether a competing workload ran during measurement.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="results", show_default=True)
def embeddings(
    profile_name: str,
    model_name: str,
    compute_engine: str,
    backend: str,
    power_state: str,
    power_cap_w: float | None,
    input_texts: tuple[str, ...],
    warmup_trials: int,
    measured_trials: int,
    exclusive_run: bool,
    out_dir: str,
) -> None:
    """Benchmark a batched-embedding request's latency and throughput.

    Deliberately separate from `run`/`sweep`: embeddings throughput isn't
    comparable to LLM token throughput, so results are written to their own
    results/<profile>/embeddings/ tree and never appear on the model/router
    leaderboard. See `lemonmatrix embeddings-results` to list them.

    \b
      lemonmatrix embeddings --profile strix --model BGE-Small-EN-GGUF \\
        --engine igpu --backend llamacpp-vulkan --power-state plugged \\
        --input "hello world" --input "a second sentence"
    """
    try:
        prof = Profile.load(profile_name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{profile_name}'. See `lemonmatrix profile list`.")

    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)

    run_os = host_os(prof.environment)
    if not run_os:
        raise click.ClickException(
            f"Could not determine profile '{profile_name}'s OS from its environment. "
            "Re-run `lemonmatrix profile add` to refresh discovery, or edit the profile file directly."
        )

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}
    issues = validate_combo_against_profile(compute_engine, backend, system_info, prof.environment, modality="Text generation")
    if issues:
        click.secho("Configuration problem(s) detected:", fg="red", bold=True)
        for issue in issues:
            for line in issue.splitlines():
                click.secho(f"  {line}", fg="red")
        raise click.ClickException("Aborting. Fix the engine/backend combination before running.")

    cfg = EmbeddingsConfig(
        model_name=model_name,
        compute_engine=compute_engine,
        backend=backend,
        os=run_os,
        power_state=power_state,
        input_texts=list(input_texts),
        power_cap_w=power_cap_w,
        warmup_trials=warmup_trials,
        measured_trials=measured_trials,
        exclusive_run=exclusive_run,
    )

    click.echo(f"Running {measured_trials} embeddings trial(s) ({warmup_trials} warmup) of {model_name} on '{profile_name}'...")
    result, _raw_trials = run_embeddings(client, prof.environment, cfg)

    try:
        validate_embeddings_result(result)
    except jsonschema.ValidationError as exc:
        click.secho("Result failed schema validation, not writing it out:", fg="red")
        raise click.ClickException(str(exc)) from exc

    result_path = save_embeddings_result(out_dir, profile_name, result)
    click.secho(f"Wrote {result_path}", fg="green")
    click.echo(
        f"latency: {result['metrics']['latency_ms']:.1f} ms   "
        f"throughput: {result['metrics']['embeddings_per_sec']:.2f} embeddings/sec"
    )
    if not result["validity"]["valid"]:
        click.secho(f"Marked invalid: {result['validity'].get('notes', 'see validity block')}", fg="yellow")


@cli.command("embeddings-results")
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--profile", "profile_name", default=None, help="Filter to a single profile.")
def embeddings_results(results_dir: str, profile_name: str | None) -> None:
    """List saved embeddings benchmark results."""
    results = list_embeddings_results(results_dir, profile=profile_name)
    if not results:
        click.echo("No embeddings results found.", err=True)
        return
    for r in results:
        flag = "" if r["validity"]["valid"] else "  [INVALID]"
        click.echo(
            f"{r['timestamp']}  {r['_profile']:<20} {r['model']['name']:<30} "
            f"{r['metrics']['embeddings_per_sec']:.2f} emb/sec{flag}"
        )


@cli.command("rerank")
@click.option("--profile", "profile_name", required=True, help="Saved profile name (see `lemonmatrix profile list`).")
@click.option("--model", "model_name", required=True, help="Reranking model name as registered in Lemonade (llamacpp GGUF recipe).")
@click.option("--engine", "compute_engine", type=click.Choice(["cpu", "igpu", "dgpu", "npu"]), required=True)
@click.option("--backend", required=True, help="e.g. llamacpp-vulkan.")
@click.option("--power-state", type=click.Choice(["plugged", "battery", "power_capped"]), required=True)
@click.option("--power-cap-w", type=float, default=None)
@click.option("--query", required=True)
@click.option("--document", "documents", multiple=True, required=True, help="Document to rerank (repeat for more).")
@click.option("--top-n", type=int, default=None)
@click.option("--warmup", "warmup_trials", type=int, default=2, show_default=True)
@click.option("--trials", "measured_trials", type=int, default=5, show_default=True)
@click.option("--exclusive/--not-exclusive", "exclusive_run", default=True,
              help="Whether a competing workload ran during measurement.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="results", show_default=True)
def rerank(
    profile_name: str,
    model_name: str,
    compute_engine: str,
    backend: str,
    power_state: str,
    power_cap_w: float | None,
    query: str,
    documents: tuple[str, ...],
    top_n: int | None,
    warmup_trials: int,
    measured_trials: int,
    exclusive_run: bool,
    out_dir: str,
) -> None:
    """Benchmark a reranking request's latency and throughput.

    Deliberately separate from `run`/`sweep`: reranking throughput isn't
    comparable to LLM token throughput, so results are written to their own
    results/<profile>/rerank/ tree and never appear on the model/router
    leaderboard. See `lemonmatrix rerank-results` to list them.

    \b
      lemonmatrix rerank --profile strix --model BGE-Reranker-Base-GGUF \\
        --engine igpu --backend llamacpp-vulkan --power-state plugged \\
        --query "capital of France" --document "Paris is the capital of France." \\
        --document "Berlin is the capital of Germany."
    """
    try:
        prof = Profile.load(profile_name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{profile_name}'. See `lemonmatrix profile list`.")

    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)

    run_os = host_os(prof.environment)
    if not run_os:
        raise click.ClickException(
            f"Could not determine profile '{profile_name}'s OS from its environment. "
            "Re-run `lemonmatrix profile add` to refresh discovery, or edit the profile file directly."
        )

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}
    issues = validate_combo_against_profile(compute_engine, backend, system_info, prof.environment, modality="Text generation")
    if issues:
        click.secho("Configuration problem(s) detected:", fg="red", bold=True)
        for issue in issues:
            for line in issue.splitlines():
                click.secho(f"  {line}", fg="red")
        raise click.ClickException("Aborting. Fix the engine/backend combination before running.")

    cfg = RerankConfig(
        model_name=model_name,
        compute_engine=compute_engine,
        backend=backend,
        os=run_os,
        power_state=power_state,
        query=query,
        documents=list(documents),
        top_n=top_n,
        power_cap_w=power_cap_w,
        warmup_trials=warmup_trials,
        measured_trials=measured_trials,
        exclusive_run=exclusive_run,
    )

    click.echo(f"Running {measured_trials} rerank trial(s) ({warmup_trials} warmup) of {model_name} on '{profile_name}'...")
    result, _raw_trials = run_rerank(client, prof.environment, cfg)

    try:
        validate_rerank_result(result)
    except jsonschema.ValidationError as exc:
        click.secho("Result failed schema validation, not writing it out:", fg="red")
        raise click.ClickException(str(exc)) from exc

    result_path = save_rerank_result(out_dir, profile_name, result)
    click.secho(f"Wrote {result_path}", fg="green")
    click.echo(
        f"latency: {result['metrics']['latency_ms']:.1f} ms   "
        f"throughput: {result['metrics']['documents_per_sec']:.2f} documents/sec"
    )
    if not result["validity"]["valid"]:
        click.secho(f"Marked invalid: {result['validity'].get('notes', 'see validity block')}", fg="yellow")


@cli.command("rerank-results")
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--profile", "profile_name", default=None, help="Filter to a single profile.")
def rerank_results(results_dir: str, profile_name: str | None) -> None:
    """List saved reranking benchmark results."""
    results = list_rerank_results(results_dir, profile=profile_name)
    if not results:
        click.echo("No rerank results found.", err=True)
        return
    for r in results:
        flag = "" if r["validity"]["valid"] else "  [INVALID]"
        click.echo(
            f"{r['timestamp']}  {r['_profile']:<20} {r['model']['name']:<30} "
            f"{r['metrics']['documents_per_sec']:.2f} docs/sec{flag}"
        )


@cli.command("meshgen")
@click.option("--profile", "profile_name", required=True, help="Saved profile name (see `lemonmatrix profile list`).")
@click.option("--model", "model_name", required=True, help="3D-generation model name as registered in Lemonade (trellis recipe).")
@click.option("--engine", "compute_engine", type=click.Choice(["cpu", "igpu", "dgpu", "npu"]), required=True)
@click.option("--backend", required=True, help="e.g. trellis-cuda.")
@click.option("--power-state", type=click.Choice(["plugged", "battery", "power_capped"]), required=True)
@click.option("--power-cap-w", type=float, default=None)
@click.option("--input-image", type=click.Path(exists=True, dir_okay=False), required=True, help="Input image (PNG/JPEG/BMP/GIF) to convert to a 3D mesh.")
@click.option("--resolution", type=click.Choice(["512", "1024", "1536"]), default=None)
@click.option("--bg-removal", type=click.Choice(["threshold", "birefnet"]), default=None)
@click.option("--uv", type=click.Choice(["box", "xatlas"]), default=None)
@click.option("--seed", type=int, default=None)
@click.option("--warmup", "warmup_trials", type=int, default=2, show_default=True)
@click.option("--trials", "measured_trials", type=int, default=5, show_default=True)
@click.option("--exclusive/--not-exclusive", "exclusive_run", default=True,
              help="Whether a competing workload ran during measurement.")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="results", show_default=True)
def meshgen(
    profile_name: str,
    model_name: str,
    compute_engine: str,
    backend: str,
    power_state: str,
    power_cap_w: float | None,
    input_image: str,
    resolution: str | None,
    bg_removal: str | None,
    uv: str | None,
    seed: int | None,
    warmup_trials: int,
    measured_trials: int,
    exclusive_run: bool,
    out_dir: str,
) -> None:
    """Benchmark an image-to-3D-mesh model's throughput (meshes/sec).

    Deliberately separate from `run`/`sweep`: mesh-generation throughput
    isn't comparable to LLM token throughput, so results are written to
    their own results/<profile>/meshgen/ tree and never appear on the
    model/router leaderboard. See `lemonmatrix meshgen-results` to list them.

    \b
      lemonmatrix meshgen --profile strix --model TRELLIS-image-large \\
        --engine dgpu --backend trellis-cuda --power-state plugged \\
        --input-image cat.png --resolution 512
    """
    try:
        prof = Profile.load(profile_name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{profile_name}'. See `lemonmatrix profile list`.")

    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)

    run_os = host_os(prof.environment)
    if not run_os:
        raise click.ClickException(
            f"Could not determine profile '{profile_name}'s OS from its environment. "
            "Re-run `lemonmatrix profile add` to refresh discovery, or edit the profile file directly."
        )

    try:
        system_info = client.system_info()
    except Exception:
        system_info = {}
    issues = validate_combo_against_profile(compute_engine, backend, system_info, prof.environment, modality="3D generation")
    if issues:
        click.secho("Configuration problem(s) detected:", fg="red", bold=True)
        for issue in issues:
            for line in issue.splitlines():
                click.secho(f"  {line}", fg="red")
        raise click.ClickException("Aborting. Fix the engine/backend combination before running.")

    cfg = MeshGenConfig(
        model_name=model_name,
        compute_engine=compute_engine,
        backend=backend,
        os=run_os,
        power_state=power_state,
        input_image_bytes=Path(input_image).read_bytes(),
        resolution=resolution,
        bg_removal=bg_removal,
        uv=uv,
        seed=seed,
        power_cap_w=power_cap_w,
        warmup_trials=warmup_trials,
        measured_trials=measured_trials,
        exclusive_run=exclusive_run,
    )

    click.echo(f"Running {measured_trials} mesh-gen trial(s) ({warmup_trials} warmup) of {model_name} on '{profile_name}'...")
    result, _raw_trials = run_meshgen(client, prof.environment, cfg)

    try:
        validate_meshgen_result(result)
    except jsonschema.ValidationError as exc:
        click.secho("Result failed schema validation, not writing it out:", fg="red")
        raise click.ClickException(str(exc)) from exc

    result_path = save_meshgen_result(out_dir, profile_name, result)
    click.secho(f"Wrote {result_path}", fg="green")
    click.echo(
        f"generation time: {result['metrics']['generation_time_ms']:.1f} ms   "
        f"throughput: {result['metrics']['meshes_per_sec']:.2f} meshes/sec"
    )
    if not result["validity"]["valid"]:
        click.secho(f"Marked invalid: {result['validity'].get('notes', 'see validity block')}", fg="yellow")


@cli.command("meshgen-results")
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--profile", "profile_name", default=None, help="Filter to a single profile.")
def meshgen_results(results_dir: str, profile_name: str | None) -> None:
    """List saved 3D mesh-generation benchmark results."""
    results = list_meshgen_results(results_dir, profile=profile_name)
    if not results:
        click.echo("No mesh-generation results found.", err=True)
        return
    for r in results:
        flag = "" if r["validity"]["valid"] else "  [INVALID]"
        click.echo(
            f"{r['timestamp']}  {r['_profile']:<20} {r['model']['name']:<30} "
            f"{r['metrics']['meshes_per_sec']:.2f} meshes/sec{flag}"
        )


@cli.command("sweep")
@click.option("--profile", "profile_name", required=True, help="Saved profile name.")
@click.option("--run-type", "run_type", type=click.Choice(["model", "router"]), default="model", show_default=True,
              help="'router' to sweep a collection.router; --backend and --engine default to router values.")
@click.option("--model", "models", multiple=True, help="Model/router name(s) to include (repeat for multiple). Omit to list available.")
@click.option("--model-class", type=click.Choice(["dense", "moe", "router"]), default=None,
              help="Auto-set to 'router' when --run-type router.")
@click.option("--quant", "quantizations", multiple=True, help="Quantization(s) (repeat for multiple, omit for all). Ignored for router runs.")
@click.option("--context-length", "context_length", type=int, default=None, help="Context length override; default: min(model max, 4096).")
@click.option("--engine", "engines", multiple=True, help="Engine(s) to sweep (omit to auto-detect; auto-set to 'router' for router runs).")
@click.option("--backend", "backends", multiple=True, help="Backend(s) to sweep, e.g. llamacpp-vulkan. Auto-set to 'collection.router' for router runs.")
@click.option("--power-state", "power_states", multiple=True, default=["plugged"], show_default=True, help="Power state(s) (repeat for multiple).")
@click.option("--power-cap-w", type=float, default=None)
@click.option("--warmup", "warmup_trials", type=int, default=2, show_default=True)
@click.option("--trials", "measured_trials", type=int, default=5, show_default=True)
@click.option("--max-tokens", type=int, default=256, show_default=True)
@click.option("--exclusive/--not-exclusive", "exclusive_run", default=True)
@click.option("--price-per-kwh", "energy_price_usd_per_kwh", type=float, default=None)
@click.option("--hardware-cost-usd", type=float, default=None, help="Purchase price of the hardware, for the cost model's amortized-hardware component.")
@click.option("--hardware-lifetime-hours", type=float, default=None, help="Expected service life in hours, for the cost model's amortized-hardware component.")
@click.option("--via-job-engine", is_flag=True, default=False,
              help="Execute each combination as a single durable Lemonade job (POST /v1/jobs) instead of N direct HTTP calls from this process. Model runs only (not --run-type router).")
@click.option("--out", "out_dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--skip-incompatible/--no-skip-incompatible", default=True, show_default=True,
              help="Skip impossible engine/backend combos (default). Use --no-skip-incompatible to run them anyway (will be marked invalid).")
def sweep(
    profile_name: str,
    run_type: str,
    models: tuple[str, ...],
    model_class: str | None,
    quantizations: tuple[str, ...],
    context_length: int | None,
    engines: tuple[str, ...],
    backends: tuple[str, ...],
    power_states: tuple[str, ...],
    power_cap_w: float | None,
    warmup_trials: int,
    measured_trials: int,
    max_tokens: int,
    exclusive_run: bool,
    energy_price_usd_per_kwh: float | None,
    hardware_cost_usd: float | None,
    hardware_lifetime_hours: float | None,
    via_job_engine: bool,
    out_dir: str,
    skip_incompatible: bool,
) -> None:
    """Run a Cartesian matrix sweep against a saved profile.

    \b
    Model sweep:
      lemonmatrix sweep --profile strix \\
        --model Llama-3.1-8B-Instruct-GGUF \\
        --backend llamacpp-vulkan --backend llamacpp-cpu \\
        --engine igpu --engine cpu --power-state plugged

    \b
    Router sweep (--backend and --engine are auto-set):
      lemonmatrix sweep --profile strix --run-type router \\
        --model my-collection-router --power-state plugged

    For router runs, engine/backend/class are auto-set; engine+backend
    validation is bypassed since the router controls backend selection.
    """
    try:
        prof = Profile.load(profile_name)
    except FileNotFoundError:
        raise click.ClickException(f"No profile named '{profile_name}'. See `lemonmatrix profile list`.")

    # Router runs: auto-fill flags that don't apply to a collection.router.
    if run_type == "router":
        model_class = model_class or "router"
        engines = engines or ("router",)
        backends = backends or ("collection.router",)

    # Model runs: --backend is required (nothing sensible to default to).
    if run_type != "router" and not backends:
        raise click.UsageError("--backend is required for model sweeps (e.g. --backend llamacpp-vulkan).")

    if via_job_engine and run_type == "router":
        raise click.UsageError("--via-job-engine does not support --run-type router (a router has no fixed backend/ctx_size for a job's load step).")

    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)

    # Query what is available on this instance.
    try:
        system_info = client.system_info()
        available_models = client.models()
    except Exception as exc:
        raise click.ClickException(f"Could not connect to profile '{profile_name}': {exc}")

    if not models:
        routers = available_routers(available_models)
        regular = [m for m in available_models if not is_router_model(m)]
        click.echo("Available models (use --model to select):")
        for m in regular:
            click.echo(f"  {m.get('id', '?')}")
        if routers:
            click.echo("Available routers (add --run-type router):")
            for m in routers:
                click.echo(f"  {m.get('id', '?')}  [router]")
        return

    # Build the list of (model_id, quantization, context_length) variants.
    DEFAULT_CTX = 4096
    model_set = set(models)
    variants: list[dict] = []
    import re as _re
    for m in available_models:
        mid = m.get("id", "")
        if run_type == "router":
            # For router sweeps, match only routers by exact id.
            if not is_router_model(m):
                continue
            if model_set and mid not in model_set:
                continue
            ctx = context_length or DEFAULT_CTX
            variants.append({"id": mid, "quantization": "none", "context_length": ctx})
        else:
            # Match by full id or by stripped base name (remove trailing -QUANTIZATION).
            match = _re.search(r"-((?:Q\d|IQ\d|F16|F32|BF16)[A-Za-z0-9_]*)$", mid)
            quant = match.group(1) if match else ""
            base = mid[: -(len(quant) + 1)] if quant and mid.endswith(f"-{quant}") else mid
            if mid not in model_set and base not in model_set:
                continue
            if quantizations and quant not in quantizations:
                continue
            ctx = context_length or min(m.get("max_context_window") or DEFAULT_CTX, DEFAULT_CTX)
            variants.append({"id": mid, "quantization": quant or "unknown", "context_length": ctx})

    if not variants:
        raise click.ClickException(
            f"No available models matched --model {list(models)} "
            + (f"--quant {list(quantizations)}" if quantizations else "")
            + ". Use --model without arguments to list what is pulled."
        )

    # Default engines: whatever hardware Lemonade reports.
    engine_list: list[str] = list(engines) or available_compute_engines(system_info) or ["cpu"]

    # Build the full Cartesian product with upfront compatibility check.
    combos = []
    skipped_report: list[tuple[str, str, list[str]]] = []  # (engine, backend, issues)
    for variant, engine, backend_str, power_state in itertools.product(variants, engine_list, backends, power_states):
        if run_type != "router":
            issues = validate_combo_against_profile(engine, backend_str, system_info, prof.environment)
            if issues:
                skipped_report.append((engine, backend_str, issues))
                if skip_incompatible:
                    continue
        combos.append({**variant, "compute_engine": engine, "backend": backend_str, "power_state": power_state})

    if skipped_report:
        click.secho(f"\n{'Skipping' if skip_incompatible else 'Warning:'} {len(skipped_report)} invalid combo(s):",
                    fg="yellow", bold=True)
        for eng, bck, issues in skipped_report:
            click.secho(f"  engine={eng} backend={bck}", fg="yellow")
            for issue in issues:
                for line in issue.splitlines():
                    click.secho(f"    {line}", fg="yellow")
        if not skip_incompatible:
            click.secho("  (proceeding anyway — results will be marked invalid; use --skip-incompatible to drop them)\n",
                        fg="yellow")
        click.echo()

    if not combos:
        raise click.ClickException("No valid combinations to run after filtering. Check your --engine/--backend flags.")

    run_os = host_os(prof.environment)
    if not run_os:
        raise click.ClickException(
            f"Could not determine profile '{profile_name}'s OS. Run `lemonmatrix profile refresh {profile_name}` first."
        )

    out_path = Path(out_dir) / profile_name
    out_path.mkdir(parents=True, exist_ok=True)

    completed = failed = 0
    for i, combo in enumerate(combos, 1):
        label = f"{combo['id']} {combo['backend']} [{combo['compute_engine']}] {combo['power_state']}"
        click.echo(f"[{i}/{len(combos)}] {label}")
        effective_class = model_class or ("router" if run_type == "router" else "dense")
        cfg = SweepConfig(
            model_name=combo["id"],
            model_class=effective_class,
            quantization=combo["quantization"],
            context_length=combo["context_length"],
            compute_engine=combo["compute_engine"],
            backend=combo["backend"],
            os=run_os,
            power_state=combo["power_state"],
            power_cap_w=power_cap_w,
            warmup_trials=warmup_trials,
            measured_trials=measured_trials,
            max_tokens=max_tokens,
            exclusive_run=exclusive_run,
            energy_price_usd_per_kwh=energy_price_usd_per_kwh,
            hardware_cost_usd=hardware_cost_usd,
            hardware_lifetime_hours=hardware_lifetime_hours,
            run_type=run_type,
        )
        try:
            if via_job_engine:
                result, raw_trials = run_sweep_via_job(client, prof.environment, cfg)
            else:
                result, raw_trials = run_sweep(client, prof.environment, cfg)
            validate_result(result)
            (out_path / f"{result['run_id']}.json").write_text(json.dumps(result, indent=2))
            save_trials(out_dir, profile_name, result["run_id"], raw_trials)
            tps = result["metrics"]["decode"]["tokens_per_sec"]
            click.secho(f"  OK {tps:.1f} tok/s", fg="green")
            if not result["validity"]["valid"]:
                click.secho(f"  (invalid: {result['validity'].get('notes', '')})", fg="yellow")
            completed += 1
        except Exception as exc:
            click.secho(f"  FAILED: {exc}", fg="red")
            failed += 1

    click.echo(f"\nSweep done: {completed} completed, {failed} failed.")


_SUBMISSION_MODALITY_VALIDATORS = {
    "classify": validate_classify_result,
    "tts": validate_tts_result,
    "stt": validate_stt_result,
    "imagegen": validate_imagegen_result,
    "audiogen": validate_audiogen_result,
    "embeddings": validate_embeddings_result,
    "rerank": validate_rerank_result,
    "meshgen": validate_meshgen_result,
}
# Sidecars, not part of the public leaderboard's data contract -- failures/
# aren't schema-conformant results at all, and trials/ are raw per-trial
# measurements rather than an aggregated result.
_SUBMISSION_SKIP_DIRS = {"failures", "trials"}


@cli.command("validate-submission")
@click.argument("paths", nargs=-1, type=click.Path(exists=True, dir_okay=False))
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True,
              help="Only used when PATHS is empty, to discover every result JSON under it.")
def validate_submission(paths: tuple[str, ...], results_dir: str) -> None:
    """Validate result JSON file(s) against the schema for their location --
    the same check a community leaderboard PR's CI run must pass before merge.

    Which schema applies is read from the file's own parent directory name:
    a bare "<profile>/<run_id>.json" validates against the model/router
    schema; "<profile>/<modality>/<run_id>.json" (classify, tts, stt,
    imagegen, audiogen, embeddings, rerank, meshgen) validates against that
    modality's own schema, since none of those are comparable to LLM token
    throughput and must never be mistaken for a model/router result. Files
    under a failures/ or trials/ directory are skipped -- neither is part of
    the public leaderboard's data contract.

    Exits non-zero (failing the CI check) if any file fails to parse or
    fails schema validation.

    \b
      lemonmatrix validate-submission results/my-machine/*.json
      lemonmatrix validate-submission                      # validates every result under --results-dir
    """
    if paths:
        targets = [Path(p) for p in paths]
    else:
        targets = sorted(Path(results_dir).glob("**/*.json"))

    if not targets:
        click.echo("No result JSON files found to validate.")
        return

    passed = 0
    skipped = 0
    failures: list[tuple[Path, str]] = []
    for path in targets:
        if path.parent.name in _SUBMISSION_SKIP_DIRS:
            skipped += 1
            continue

        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            failures.append((path, f"couldn't read/parse: {exc}"))
            continue

        validator = _SUBMISSION_MODALITY_VALIDATORS.get(path.parent.name, validate_result)
        try:
            validator(data)
        except jsonschema.ValidationError as exc:
            failures.append((path, str(exc)))
            continue

        click.secho(f"OK    {path}", fg="green")
        passed += 1

    for path, reason in failures:
        click.secho(f"FAIL  {path}", fg="red")
        click.secho(f"      {reason}", fg="red")

    click.echo(f"\n{passed} passed, {len(failures)} failed, {skipped} skipped.")
    if failures:
        raise click.ClickException(f"{len(failures)} result file(s) failed validation.")


@cli.command("export")
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--profile", "profile_name", default=None, help="Filter to a single profile.")
@click.option("--valid-only", is_flag=True, default=False, help="Exclude runs marked invalid.")
@click.option("--out", "out_file", type=click.Path(dir_okay=False), default=None, help="Write CSV to this path (default: stdout).")
def export(results_dir: str, profile_name: str | None, valid_only: bool, out_file: str | None) -> None:
    """Export benchmark results as CSV."""
    results = list_results(results_dir)
    if profile_name:
        results = [r for r in results if r.get("_profile") == profile_name]
    if valid_only:
        results = [r for r in results if r.get("validity", {}).get("valid")]

    if not results:
        click.echo("No results found.", err=True)
        return

    csv_text = results_to_csv(results)
    if out_file:
        Path(out_file).write_text(csv_text)
        click.secho(f"Wrote {len(results)} row(s) to {out_file}", fg="green")
    else:
        click.echo(csv_text, nl=False)


@cli.command("trials")
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.argument("profile_name")
@click.argument("run_id")
def trials(results_dir: str, profile_name: str, run_id: str) -> None:
    """Print raw per-trial measurements for a completed run."""
    data = load_trials(results_dir, profile_name, run_id)
    if data is None:
        raise click.ClickException(f"No trials sidecar found for {profile_name}/{run_id}.")
    click.echo(json.dumps(data, indent=2))


@cli.command("dashboard")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address. Use 0.0.0.0 to expose beyond localhost.")
@click.option("--port", type=int, default=5050, show_default=True)
@click.option("--results-dir", type=click.Path(file_okay=False), default="results", show_default=True)
@click.option("--debug", is_flag=True, default=False, help="Enable Flask's auto-reload and debugger.")
def dashboard(host: str, port: int, results_dir: str, debug: bool) -> None:
    """Run the local web dashboard: profiles, leaderboard, and a form to launch new sweeps."""
    from .webapp import create_app

    app = create_app(results_dir=results_dir)
    click.echo(f"Dashboard running at http://{host}:{port}  (results dir: {results_dir})")
    app.run(host=host, port=port, debug=debug)


def main() -> None:
    try:
        cli()
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)


if __name__ == "__main__":
    main()
