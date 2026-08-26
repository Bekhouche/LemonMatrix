"""A local Flask dashboard over the CLI's own data: profiles + result JSON files.

Runs single-user, synchronous, local-only by default -- there is no job queue,
so submitting a benchmark run blocks the request for as long as the sweep
takes (the same duration `lemonmatrix run` would take on the command line).
That's an acceptable trade for a local control-plane admin tool; it would not
be for a multi-user hosted deployment.
"""

from __future__ import annotations

import io
import itertools
import json
import secrets
import wave
from pathlib import Path

from flask import Flask, abort, flash, redirect, render_template, request, url_for

import jsonschema
from flask import Response

from .bench import (AudioGenConfig, ClassifyConfig, EmbeddingsConfig, ImageGenConfig, MeshGenConfig,
                     RerankConfig, STTConfig, SweepConfig, TTSConfig, run_audiogen, run_classify,
                     run_embeddings, run_imagegen, run_meshgen, run_rerank, run_stt, run_sweep,
                     run_sweep_via_job, run_tts)
from .capabilities import (available_backends, available_compute_engines, available_routers,
                           compatible_engines_for_backend, host_os, parse_quantization,
                           validate_combo_against_profile)
from .capabilities import _BACKEND_KEY_ENGINES as _COMPAT_MAP_RAW
from .client import BENCH_TIMEOUT, LemonadeClient, job_progress_percent
from .discover import QUICK_SELECT_PORTS, expand_subnet, scan, scan_localhost
from .profile import DEFAULT_PORT, Profile, build_url, connect_and_save
from .results_store import (get_path, list_audiogen_results, list_classify_results,
                             list_embeddings_results, list_failures, list_imagegen_results,
                             list_meshgen_results, list_rerank_results, list_results, list_stt_results,
                             list_tts_results, load_result, load_trials, results_to_csv,
                             save_audiogen_result, save_classify_result, save_embeddings_result,
                             save_failure, save_imagegen_result, save_meshgen_result, save_rerank_result,
                             save_stt_result, save_trials, save_tts_result)
from .sweep_batch import MAX_SWEEP_COMBINATIONS, SweepBatch, expand_combinations, start_batch
from .sweep_store import SweepStore
from .validate import (validate_audiogen_result, validate_classify_result, validate_embeddings_result,
                        validate_imagegen_result, validate_meshgen_result, validate_rerank_result,
                        validate_result, validate_stt_result, validate_tts_result)


DEFAULT_BENCH_CONTEXT_LENGTH = 4096


def _model_options(models: list[dict]) -> list[dict]:
    """Enrich each already-pulled model with a best-effort parsed quantization
    and context length, for the run form's auto-fill.

    Quantization is parsed from the checkpoint's ":VARIANT" suffix (the same
    convention /v1/pull uses, confirmed live) when present, else from a
    trailing GGUF-quant-looking segment of the model's own id (Q4_K_M, Q8_0,
    IQ4_XS, F16, BF16, ...) -- deliberately narrow, since a looser pattern
    would also match a non-quant suffix like "...-GGUF" itself and silently
    mislabel the file format as if it were the quantization. Neither path is
    guaranteed by the API, so this is presentation-only pre-fill -- the
    schema's actual quantization always comes from what the form submits,
    never silently substituted here.

    context_length defaults to min(model's max, DEFAULT_BENCH_CONTEXT_LENGTH),
    not the model's max outright -- confirmed live that defaulting to a
    model's full max_context_window (262144 for one real model) makes even
    loading painfully slow (huge KV-cache allocation) before any inference
    runs. The field stays editable up to the real max for anyone who
    actually wants to benchmark at long context.
    """
    options = []
    for m in models:
        model_id = m.get("id") or ""
        checkpoint = m.get("checkpoint") or ""
        quant = parse_quantization(model_id, checkpoint)

        # Strip the quant suffix to get the base model name two pulled
        # quants of the same model (e.g. "...-Q4_K_M" and "...-Q8_0") share --
        # lets the run form group them as "one model, pick a quantization"
        # instead of two unrelated-looking full ids. Falls back to the id
        # itself when there's no quant to strip, so it's still its own group.
        base_name = model_id[: -(len(quant) + 1)] if quant and model_id.endswith(f"-{quant}") else model_id

        options.append(
            {
                "id": model_id,
                "base_name": base_name,
                "quantization": quant,
                "context_length": min(m.get("max_context_window") or DEFAULT_BENCH_CONTEXT_LENGTH, DEFAULT_BENCH_CONTEXT_LENGTH),
                "recipe": m.get("recipe", ""),
                "size_gb": m.get("size"),
            }
        )
    return options


def _group_by_base_model(options: list[dict]) -> dict[str, list[dict]]:
    """Group _model_options() output by base_name, preserving first-seen
    order -- powers the run form's two dependent dropdowns (pick a model,
    then pick which of its pulled quantizations to run)."""
    groups: dict[str, list[dict]] = {}
    for opt in options:
        groups.setdefault(opt["base_name"], []).append(opt)
    return groups


def _running_batch_for_profile(app: Flask, profile_name: str) -> SweepBatch | None:
    """Whether a sweep batch is already in flight for this profile -- since
    Lemonade holds only one model loaded at a time, a second concurrent
    batch against the same profile would just race the first one's load()
    calls, so starting one is blocked while another is running."""
    for batch in app.config["SWEEP_BATCHES"].values():
        if batch.profile_name == profile_name and batch.status == "running":
            return batch
    return None


def _live_capabilities(prof: Profile) -> dict:
    """Best-effort: engines/backends/models this profile's live instance can
    run right now. Never raises -- a profile page must still render even when
    the instance is offline, just with an explanatory error instead."""
    client = LemonadeClient(prof.base_url, api_key=prof.api_key)
    try:
        system_info = client.system_info()
        models = client.models()
    except Exception as exc:
        return {
            "engines": [], "backends": [],
            "backends_installed": [], "backends_installable": [],
            "models": [], "error": str(exc), "system_info": {},
        }
    all_backends = available_backends(system_info, prof.environment)
    return {
        "engines": available_compute_engines(system_info),
        "backends": all_backends,
        "backends_installed": [b for b in all_backends if b["state"] == "installed"],
        "backends_installable": [b for b in all_backends if b["state"] == "installable"],
        "models": models,
        "error": None,
        "system_info": system_info,
    }


PROFILE_PING_TIMEOUT = 2.0


def _profile_reachable(prof: Profile) -> bool:
    """Whether this profile's Lemonade instance answers /api/v1/health right
    now. Deliberately a short, dedicated timeout (not DEFAULT_TIMEOUT) and
    just the one cheap call -- the profiles list can hold many entries and
    must still load promptly even when some point at machines that are now
    off, unlike _live_capabilities's fuller system-info+models probe."""
    client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=PROFILE_PING_TIMEOUT)
    try:
        client.health()
        return True
    except Exception:
        return False


def _models_for_recipes(models: list[dict], recipes: set[str]) -> list[dict]:
    """Already-pulled models whose recipe matches one of `recipes` -- narrows
    the run form's model dropdown to models that can actually serve this
    modality (e.g. onnxruntime for classify, kokoro/openmoss for tts)."""
    return [m for m in models if m.get("recipe") in recipes]


def _compat_map_json(backends: list[dict]) -> str:
    """Build a JSON map of {backend_str: [compatible_engines]} for client-side validation."""
    result = {}
    for b in backends:
        backend_key = b.get("backend_key", "")
        allowed = _COMPAT_MAP_RAW.get(backend_key)
        result[b["backend"]] = sorted(allowed) if allowed else []
    return json.dumps(result)

LEADERBOARD_COLUMNS = [
    ("timestamp", "Timestamp", "text"),
    ("_profile", "Profile", "text"),
    ("environment.device_model", "Device", "text"),
    ("model.name", "Model", "text"),
    ("model.quantization", "Quant", "text"),
    ("config.compute_engine", "Engine", "chip"),
    ("config.backend", "Backend", "text"),
    ("config.power_state", "Power", "text"),
    ("metrics.decode.tokens_per_sec", "Decode tok/s", "num1"),
    ("metrics.prefill.tokens_per_sec", "Prefill tok/s", "num1"),
    ("metrics.ttft_ms", "TTFT ms", "num0"),
    ("metrics.decode.joules_per_token", "J/tok (decode)", "num3"),
    ("validity.valid", "Valid", "status"),
]


class _PersistedBatch:
    """Read-only view of a batch loaded from SweepStore on startup.

    Has the same interface as SweepBatch so the existing templates and the
    _running_batch_for_profile helper work without modification. A persisted
    batch is never 'running' (interrupted_running_batches marked it done or
    interrupted), so starting new batches is never blocked by ghost entries.
    """

    def __init__(self, raw: dict) -> None:
        self.id = raw["id"]
        self.profile_name = raw["profile_name"]
        self.created_at = raw["created_at"]
        self.status = raw["status"]
        self.items = raw["items"]

    @property
    def completed_count(self) -> int:
        return sum(1 for i in self.items if i.get("status") in ("completed", "failed"))

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def failed_count(self) -> int:
        return sum(1 for i in self.items if i.get("status") == "failed")


def create_app(results_dir: str | Path = "results") -> Flask:
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(16)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    app.config["RESULTS_DIR"] = results_path
    app.config["SWEEP_BATCHES"] = {}  # batch_id -> SweepBatch, in-process only

    # Durable sweep store — survives restarts.
    store = SweepStore(results_path / ".sweeps.db")
    store.interrupt_running_batches()
    app.config["SWEEP_STORE"] = store

    # Rehydrate persisted batch summaries into the in-memory dict so previous
    # runs are visible in the UI.  These are not live SweepBatch objects (the
    # background thread is gone), so we wrap them in a lightweight proxy that
    # makes the templates happy without restarting any threads.
    for raw in store.load_all_batches():
        proxy = _PersistedBatch(raw)
        app.config["SWEEP_BATCHES"][proxy.id] = proxy

    @app.route("/")
    def index():
        results_dir = app.config["RESULTS_DIR"]
        results = list_results(results_dir)

        # "valid_only" defaults to on: invalid runs are meant to be visible on
        # request, not by default (IDEA.md's own ranking-integrity gap list).
        # A plain absent-checkbox can't distinguish "never filtered" from "the
        # user explicitly unchecked it", so the filter form carries a hidden
        # "filtered" marker -- once present, the checkbox's actual state is
        # honored; sort-link clicks preserve whatever was already in the query
        # string (including "filtered"), so this persists across sorting.
        explicit_filters = "filtered" in request.args
        filters = {
            "profile": request.args.get("profile", ""),
            "engine": request.args.get("engine", ""),
            "backend": request.args.get("backend", ""),
            "model": request.args.get("model", ""),
            "valid_only": request.args.get("valid_only", "") if explicit_filters else "1",
        }
        if filters["profile"]:
            results = [r for r in results if r.get("_profile") == filters["profile"]]
        if filters["engine"]:
            results = [r for r in results if get_path(r, "config.compute_engine") == filters["engine"]]
        if filters["backend"]:
            results = [r for r in results if get_path(r, "config.backend") == filters["backend"]]
        if filters["model"]:
            needle = filters["model"].lower()
            results = [r for r in results if needle in (get_path(r, "model.name") or "").lower()]
        if filters["valid_only"]:
            results = [r for r in results if get_path(r, "validity.valid")]

        sort_key = request.args.get("sort", "timestamp")
        sort_dir = request.args.get("dir", "desc")
        known_keys = {key for key, _, _ in LEADERBOARD_COLUMNS}
        if sort_key not in known_keys:
            sort_key = "timestamp"
        results.sort(key=lambda r: (get_path(r, sort_key) is None, get_path(r, sort_key)), reverse=(sort_dir == "desc"))

        all_results = list_results(results_dir)
        stats = {
            "total": len(all_results),
            "valid": sum(1 for r in all_results if get_path(r, "validity.valid")),
            "profiles": len({r.get("_profile") for r in all_results}),
        }

        facets = {
            "profiles": sorted({r.get("_profile") for r in all_results if r.get("_profile")}),
            "engines": sorted({get_path(r, "config.compute_engine") for r in all_results if get_path(r, "config.compute_engine")}),
            "backends": sorted({get_path(r, "config.backend") for r in all_results if get_path(r, "config.backend")}),
        }

        # Load failures, optionally filtered by the same profile filter.
        failures = list_failures(results_dir, filters["profile"] or None)
        failures.sort(key=lambda f: f.get("timestamp", ""), reverse=True)

        def next_dir(key: str) -> str:
            return "asc" if (key == sort_key and sort_dir == "desc") else "desc"

        return render_template(
            "index.html",
            columns=LEADERBOARD_COLUMNS,
            results=results,
            get=get_path,
            stats=stats,
            facets=facets,
            filters=filters,
            sort_key=sort_key,
            sort_dir=sort_dir,
            next_dir=next_dir,
            failures=failures,
        )

    @app.route("/profiles")
    def profiles():
        all_profiles = Profile.list_all()
        # A short, dedicated timeout -- this page can list many profiles and
        # must still load promptly even if one of them points at a machine
        # that's now off or unreachable; _live_capabilities's full
        # system-info+models probe is unnecessary just to know reachability.
        online_by_name = {p.name: _profile_reachable(p) for p in all_profiles}
        return render_template(
            "profiles.html", profiles=all_profiles, default_port=DEFAULT_PORT, detected=None,
            online_by_name=online_by_name,
        )

    @app.route("/profiles/detect", methods=["POST"])
    def profiles_detect():
        ports = request.form.get("ports", "").strip()
        port_list = [int(p) for p in ports.split(",") if p.strip()] if ports else QUICK_SELECT_PORTS
        subnet = request.form.get("subnet", "").strip()

        found = set(scan_localhost(ports=port_list))
        if subnet:
            try:
                hosts = expand_subnet(subnet)
            except ValueError as exc:
                flash(str(exc), "error")
                return redirect(url_for("profiles"))
            found |= set(scan(hosts, ports=port_list))

        detected = []
        for host, port in sorted(found):
            base_url = build_url(host, port)
            client = LemonadeClient(base_url)
            label = "unknown device"
            try:
                info = client.system_info()
                health = client.health()
                label = f"{info.get('OEM System', 'unknown device')} (lemonade {health.get('version', '?')})"
            except Exception:
                pass
            detected.append({"host": host, "port": port, "base_url": base_url, "label": label})

        if not detected:
            flash("No live Lemonade instances found.", "error")
        return render_template("profiles.html", profiles=Profile.list_all(), default_port=DEFAULT_PORT, detected=detected)

    @app.route("/profiles/add", methods=["POST"])
    def profiles_add():
        name = request.form["name"].strip()
        base_url = request.form.get("url", "").strip() or build_url(
            request.form.get("host", "localhost").strip(),
            int(request.form.get("port") or DEFAULT_PORT),
            request.form.get("scheme", "http"),
        )
        api_key = request.form.get("token", "").strip() or None

        try:
            prof, gaps = connect_and_save(name, base_url, api_key)
        except ConnectionError as exc:
            flash(str(exc), "error")
            return redirect(url_for("profiles"))

        message = f"Saved profile '{prof.name}'."
        if gaps:
            message += f" Could not auto-discover: {', '.join(gaps)} (defaulted to \"unknown\")."
        flash(message, "ok")
        return redirect(url_for("profile_detail", name=prof.name))

    @app.route("/profiles/<name>")
    def profile_detail(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)
        results = [r for r in list_results(app.config["RESULTS_DIR"]) if r.get("_profile") == name]
        capabilities = _live_capabilities(prof)
        failure_count = len(list_failures(app.config["RESULTS_DIR"], name))
        return render_template(
            "profile_detail.html",
            profile=prof,
            results=results,
            columns=LEADERBOARD_COLUMNS,
            get=get_path,
            capabilities=capabilities,
            failure_count=failure_count,
        )

    @app.route("/profiles/<name>/failures")
    def profile_failures(name: str):
        try:
            Profile.load(name)
        except FileNotFoundError:
            abort(404)
        failures = list_failures(app.config["RESULTS_DIR"], name)
        return render_template("failures.html", profile_name=name, failures=failures)

    @app.route("/profiles/<name>/backends/install", methods=["POST"])
    def profile_install_backend(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        recipe = request.form.get("recipe", "").strip()
        backend_key = request.form.get("backend_key", "").strip()
        if not recipe or not backend_key:
            flash("Missing recipe/backend to install.", "error")
            return redirect(url_for("profile_detail", name=name))

        client = LemonadeClient(prof.base_url, api_key=prof.api_key)
        try:
            client.install_backend(recipe, backend_key)
        except Exception as exc:
            flash(f"Install failed for {recipe}:{backend_key} -- {exc}", "error")
            return redirect(url_for("profile_detail", name=name))

        flash(f"Installed {recipe}:{backend_key}.", "ok")
        return redirect(url_for("profile_detail", name=name))

    @app.route("/profiles/<name>/models/add")
    def profile_models_add(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        query = request.args.get("q", "").strip()
        checkpoint = request.args.get("checkpoint", "").strip()
        client = LemonadeClient(prof.base_url, api_key=prof.api_key)

        search_results, variants, error = None, None, None
        try:
            if query:
                search_results = client.search_registry(query).get("results", [])
            if checkpoint:
                variants = client.pull_variants(checkpoint)
        except Exception as exc:
            error = str(exc)

        return render_template(
            "models_add.html",
            profile=prof,
            query=query,
            checkpoint=checkpoint,
            search_results=search_results,
            variants=variants,
            error=error,
        )

    @app.route("/profiles/<name>/models/pull", methods=["POST"])
    def profile_pull_model(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        model_name = request.form.get("model_name", "").strip()
        recipe = request.form.get("recipe", "").strip()
        checkpoint = request.form.get("checkpoint", "").strip()
        if not model_name or not checkpoint:
            flash("Missing model name or checkpoint to pull.", "error")
            return redirect(url_for("profile_detail", name=name))

        client = LemonadeClient(prof.base_url, api_key=prof.api_key)
        try:
            client.start_model_download(model_name, recipe=recipe or None, checkpoint=checkpoint)
        except Exception as exc:
            flash(f"Couldn't start download of {model_name} -- {exc}", "error")
            return redirect(url_for("profile_detail", name=name))

        flash(f"Started downloading {model_name} in the background.", "ok")
        return redirect(url_for("profile_downloads", name=name))

    @app.route("/profiles/<name>/downloads")
    def profile_downloads(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        client = LemonadeClient(prof.base_url, api_key=prof.api_key)
        try:
            jobs = client.list_downloads()
            error = None
        except Exception as exc:
            jobs, error = [], str(exc)

        for job in jobs:
            job["_progress_percent"] = job_progress_percent(job)

        return render_template("downloads.html", profile=prof, jobs=jobs, error=error)

    @app.route("/profiles/<name>/downloads/control", methods=["POST"])
    def profile_downloads_control(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        download_id = request.form.get("id", "").strip()
        action = request.form.get("action", "").strip()
        if not download_id or action not in ("pause", "cancel", "remove"):
            flash("Missing or invalid download id/action.", "error")
            return redirect(url_for("profile_downloads", name=name))

        client = LemonadeClient(prof.base_url, api_key=prof.api_key)
        try:
            client.control_download(download_id, action)
        except Exception as exc:
            flash(f"Couldn't {action} {download_id} -- {exc}", "error")
        return redirect(url_for("profile_downloads", name=name))

    @app.route("/profiles/<name>/run", methods=["GET", "POST"])
    def profile_run(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            models_by_base = _group_by_base_model(_model_options(capabilities.get("models") or []))
            return render_template(
                "run_form.html",
                profile=prof,
                capabilities=capabilities,
                models_by_base=models_by_base,
                models_by_base_json=json.dumps(models_by_base),
                compat_map_json=_compat_map_json(capabilities.get("backends") or []),
                default_os=host_os(prof.environment),
                routers=available_routers(capabilities.get("models") or []),
            )

        form = request.form
        run_type = form.get("run_type", "model").strip() or "model"

        # OS is a fixed fact of the profile (IDEA.md: a different OS is a
        # different profile, not a per-run setting) -- always derived from
        # the profile's own discovered environment, never taken from the
        # form, so there is nothing for a user to pick inconsistently here.
        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_run", name=name))

        if run_type == "router":
            # Router runs: Lemonade handles routing transparently over the
            # same chat/completions endpoint, so engine/backend/quant/class
            # are all fixed facts of "this is a router", not user choices --
            # same auto-fill the CLI's `run --run-type router` applies.
            router_model = form.get("router_model", "").strip()
            if not router_model:
                flash("Pick a router to benchmark.", "error")
                return redirect(url_for("profile_run", name=name))
            if "via_job_engine" in form:
                flash("Job-engine execution doesn't support router runs (a router has no fixed backend/ctx_size for a job's load step).", "error")
                return redirect(url_for("profile_run", name=name))
            model_name = router_model
            model_class = "router"
            quantization = "none"
            compute_engine = "router"
            backend_val = "collection.router"
        else:
            missing = [
                field
                for field in ("model_name", "quantization", "backend")
                if not form.get(field, "").strip()
            ]
            if missing:
                flash(f"Missing required field(s): {', '.join(missing)}.", "error")
                return redirect(url_for("profile_run", name=name))

            model_name = form["model_name"].strip()
            model_class = form["model_class"]
            quantization = form["quantization"].strip()
            compute_engine = form.get("compute_engine", "").strip()
            backend_val = form.get("backend", "").strip()

            # Validate engine/backend against live profile before spending any time running.
            caps = _live_capabilities(prof)
            issues = validate_combo_against_profile(
                compute_engine, backend_val, caps.get("system_info") or {}, prof.environment
            )
            if issues:
                for issue in issues:
                    flash(issue.replace("\n", " "), "error")
                flash("Fix the engine/backend combination before running.", "error")
                return redirect(url_for("profile_run", name=name))

        cfg = SweepConfig(
            model_name=model_name,
            model_class=model_class,
            quantization=quantization,
            context_length=int(form["context_length"]),
            compute_engine=compute_engine,
            backend=backend_val,
            os=run_os,
            power_state=form["power_state"],
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            max_tokens=int(form.get("max_tokens") or 256),
            exclusive_run="exclusive_run" in form,
            energy_price_usd_per_kwh=float(form["energy_price_usd_per_kwh"]) if form.get("energy_price_usd_per_kwh") else None,
            hardware_cost_usd=float(form["hardware_cost_usd"]) if form.get("hardware_cost_usd") else None,
            hardware_lifetime_hours=float(form["hardware_lifetime_hours"]) if form.get("hardware_lifetime_hours") else None,
            run_type=run_type,
        )

        client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)
        attempted = {
            "model_name": cfg.model_name,
            "quantization": cfg.quantization,
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "power_state": cfg.power_state,
        }
        try:
            if "via_job_engine" in form and run_type != "router":
                result, raw_trials = run_sweep_via_job(client, prof.environment, cfg)
            else:
                result, raw_trials = run_sweep(client, prof.environment, cfg)
            validate_result(result)
        except jsonschema.ValidationError as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "schema_validation", str(exc))
            flash(f"Result failed schema validation: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_run", name=name))
        except Exception as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "run", str(exc))
            flash(f"Run failed: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_run", name=name))

        out_dir = app.config["RESULTS_DIR"] / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{result['run_id']}.json").write_text(json.dumps(result, indent=2))
        save_trials(app.config["RESULTS_DIR"], name, result["run_id"], raw_trials)

        flash(f"Run complete: {result['metrics']['decode']['tokens_per_sec']:.1f} decode tok/s.", "ok")
        return redirect(url_for("result_detail", profile=name, run_id=result["run_id"]))

    @app.route("/profiles/<name>/classify", methods=["GET", "POST"])
    def profile_classify(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            classify_models = _models_for_recipes(capabilities.get("models") or [], {"onnxruntime"})
            classify_backends = available_backends(
                capabilities.get("system_info") or {}, prof.environment, modality="Text classification"
            )
            results = list_classify_results(app.config["RESULTS_DIR"], profile=name)
            return render_template(
                "classify.html",
                profile=prof,
                capabilities=capabilities,
                classify_models=classify_models,
                classify_backends=classify_backends,
                results=results,
                default_os=host_os(prof.environment),
            )

        form = request.form
        model_name = form.get("model_name", "").strip()
        input_text = form.get("input_text", "").strip()
        backend_val = form.get("backend", "").strip()
        compute_engine = form.get("compute_engine", "").strip()
        missing = [
            label for label, val in
            [("model", model_name), ("input text", input_text), ("backend", backend_val), ("engine", compute_engine)]
            if not val
        ]
        if missing:
            flash(f"Missing required field(s): {', '.join(missing)}.", "error")
            return redirect(url_for("profile_classify", name=name))

        # OS is a fixed fact of the profile, same rule as the model run form.
        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_classify", name=name))

        # Pre-flight check, same as profile_run/profile_sweep -- without this,
        # an impossible engine/backend pair would run the full warmup+measured
        # trial set (wasting real time and, for slow backends, real minutes)
        # before validity.valid=False ever surfaced the mistake.
        caps = _live_capabilities(prof)
        issues = validate_combo_against_profile(
            compute_engine, backend_val, caps.get("system_info") or {}, prof.environment,
            modality="Text classification",
        )
        if issues:
            for issue in issues:
                flash(issue.replace("\n", " "), "error")
            flash("Fix the engine/backend combination before running.", "error")
            return redirect(url_for("profile_classify", name=name))

        cfg = ClassifyConfig(
            model_name=model_name,
            compute_engine=compute_engine,
            backend=backend_val,
            os=run_os,
            power_state=form.get("power_state", "plugged"),
            input_text=input_text,
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            top_k=int(form["top_k"]) if form.get("top_k") else None,
            exclusive_run="exclusive_run" in form,
        )

        client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)
        attempted = {
            "model_name": cfg.model_name,
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "power_state": cfg.power_state,
        }
        try:
            result, _raw_trials = run_classify(client, prof.environment, cfg)
            validate_classify_result(result)
        except jsonschema.ValidationError as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "schema_validation", str(exc))
            flash(f"Result failed schema validation: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_classify", name=name))
        except Exception as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "classify_run", str(exc))
            flash(f"Run failed: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_classify", name=name))

        save_classify_result(app.config["RESULTS_DIR"], name, result)
        flash(
            f"Classify run complete: {result['metrics']['latency_ms']:.1f} ms, "
            f"{result['metrics']['classifications_per_sec']:.2f} classifications/sec.",
            "ok",
        )
        return redirect(url_for("profile_classify", name=name))

    @app.route("/profiles/<name>/tts", methods=["GET", "POST"])
    def profile_tts(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            tts_models = _models_for_recipes(capabilities.get("models") or [], {"kokoro", "openmoss"})
            tts_backends = available_backends(
                capabilities.get("system_info") or {}, prof.environment, modality="Text-to-speech"
            )
            results = list_tts_results(app.config["RESULTS_DIR"], profile=name)
            return render_template(
                "tts.html",
                profile=prof,
                capabilities=capabilities,
                tts_models=tts_models,
                tts_backends=tts_backends,
                results=results,
                default_os=host_os(prof.environment),
            )

        form = request.form
        model_name = form.get("model_name", "").strip()
        input_text = form.get("input_text", "").strip()
        backend_val = form.get("backend", "").strip()
        compute_engine = form.get("compute_engine", "").strip()
        missing = [
            label for label, val in
            [("model", model_name), ("input text", input_text), ("backend", backend_val), ("engine", compute_engine)]
            if not val
        ]
        if missing:
            flash(f"Missing required field(s): {', '.join(missing)}.", "error")
            return redirect(url_for("profile_tts", name=name))

        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_tts", name=name))

        caps = _live_capabilities(prof)
        issues = validate_combo_against_profile(
            compute_engine, backend_val, caps.get("system_info") or {}, prof.environment,
            modality="Text-to-speech",
        )
        if issues:
            for issue in issues:
                flash(issue.replace("\n", " "), "error")
            flash("Fix the engine/backend combination before running.", "error")
            return redirect(url_for("profile_tts", name=name))

        cfg = TTSConfig(
            model_name=model_name,
            compute_engine=compute_engine,
            backend=backend_val,
            os=run_os,
            power_state=form.get("power_state", "plugged"),
            input_text=input_text,
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            voice=form.get("voice", "").strip() or None,
            speed=float(form["speed"]) if form.get("speed") else None,
            exclusive_run="exclusive_run" in form,
        )

        client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)
        attempted = {
            "model_name": cfg.model_name,
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "power_state": cfg.power_state,
        }
        try:
            result, _raw_trials = run_tts(client, prof.environment, cfg)
            validate_tts_result(result)
        except jsonschema.ValidationError as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "schema_validation", str(exc))
            flash(f"Result failed schema validation: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_tts", name=name))
        except Exception as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "tts_run", str(exc))
            flash(f"Run failed: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_tts", name=name))

        save_tts_result(app.config["RESULTS_DIR"], name, result)
        flash(
            f"TTS run complete: {result['metrics']['real_time_factor']:.2f}x real-time "
            f"({result['metrics']['generation_time_ms']:.1f} ms for {result['metrics']['audio_duration_s']:.2f}s of audio).",
            "ok",
        )
        return redirect(url_for("profile_tts", name=name))

    @app.route("/profiles/<name>/stt", methods=["GET", "POST"])
    def profile_stt(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            stt_models = _models_for_recipes(capabilities.get("models") or [], {"whispercpp", "moonshine"})
            stt_backends = available_backends(
                capabilities.get("system_info") or {}, prof.environment, modality="Speech-to-text"
            )
            results = list_stt_results(app.config["RESULTS_DIR"], profile=name)
            return render_template(
                "stt.html",
                profile=prof,
                capabilities=capabilities,
                stt_models=stt_models,
                stt_backends=stt_backends,
                results=results,
                default_os=host_os(prof.environment),
            )

        form = request.form
        model_name = form.get("model_name", "").strip()
        backend_val = form.get("backend", "").strip()
        compute_engine = form.get("compute_engine", "").strip()
        audio_file = request.files.get("audio_file")
        missing = [
            label for label, val in
            [("model", model_name), ("backend", backend_val), ("engine", compute_engine)]
            if not val
        ]
        if not audio_file or not audio_file.filename:
            missing.append("audio file")
        if missing:
            flash(f"Missing required field(s): {', '.join(missing)}.", "error")
            return redirect(url_for("profile_stt", name=name))

        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_stt", name=name))

        audio_bytes = audio_file.read()
        try:
            wave.open(io.BytesIO(audio_bytes), "rb").close()
        except wave.Error as exc:
            flash(f"Couldn't read '{audio_file.filename}' as a WAV file: {exc}", "error")
            return redirect(url_for("profile_stt", name=name))

        caps = _live_capabilities(prof)
        issues = validate_combo_against_profile(
            compute_engine, backend_val, caps.get("system_info") or {}, prof.environment,
            modality="Speech-to-text",
        )
        if issues:
            for issue in issues:
                flash(issue.replace("\n", " "), "error")
            flash("Fix the engine/backend combination before running.", "error")
            return redirect(url_for("profile_stt", name=name))

        cfg = STTConfig(
            model_name=model_name,
            compute_engine=compute_engine,
            backend=backend_val,
            os=run_os,
            power_state=form.get("power_state", "plugged"),
            audio_bytes=audio_bytes,
            audio_filename=audio_file.filename,
            language=form.get("language", "").strip() or None,
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            exclusive_run="exclusive_run" in form,
        )

        client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)
        attempted = {
            "model_name": cfg.model_name,
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "power_state": cfg.power_state,
        }
        try:
            result, _raw_trials = run_stt(client, prof.environment, cfg)
            validate_stt_result(result)
        except jsonschema.ValidationError as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "schema_validation", str(exc))
            flash(f"Result failed schema validation: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_stt", name=name))
        except Exception as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "stt_run", str(exc))
            flash(f"Run failed: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_stt", name=name))

        save_stt_result(app.config["RESULTS_DIR"], name, result)
        flash(
            f"STT run complete: {result['metrics']['real_time_factor']:.2f}x real-time "
            f"({result['metrics']['transcription_time_ms']:.1f} ms for {result['metrics']['audio_duration_s']:.2f}s of audio).",
            "ok",
        )
        return redirect(url_for("profile_stt", name=name))

    @app.route("/profiles/<name>/imagegen", methods=["GET", "POST"])
    def profile_imagegen(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            imagegen_models = _models_for_recipes(capabilities.get("models") or [], {"sd-cpp", "thenoise"})
            imagegen_backends = available_backends(
                capabilities.get("system_info") or {}, prof.environment, modality="Image generation"
            )
            results = list_imagegen_results(app.config["RESULTS_DIR"], profile=name)
            return render_template(
                "imagegen.html",
                profile=prof,
                capabilities=capabilities,
                imagegen_models=imagegen_models,
                imagegen_backends=imagegen_backends,
                results=results,
                default_os=host_os(prof.environment),
            )

        form = request.form
        model_name = form.get("model_name", "").strip()
        prompt = form.get("prompt", "").strip()
        backend_val = form.get("backend", "").strip()
        compute_engine = form.get("compute_engine", "").strip()
        missing = [
            label for label, val in
            [("model", model_name), ("prompt", prompt), ("backend", backend_val), ("engine", compute_engine)]
            if not val
        ]
        if missing:
            flash(f"Missing required field(s): {', '.join(missing)}.", "error")
            return redirect(url_for("profile_imagegen", name=name))

        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_imagegen", name=name))

        caps = _live_capabilities(prof)
        issues = validate_combo_against_profile(
            compute_engine, backend_val, caps.get("system_info") or {}, prof.environment,
            modality="Image generation",
        )
        if issues:
            for issue in issues:
                flash(issue.replace("\n", " "), "error")
            flash("Fix the engine/backend combination before running.", "error")
            return redirect(url_for("profile_imagegen", name=name))

        cfg = ImageGenConfig(
            model_name=model_name,
            compute_engine=compute_engine,
            backend=backend_val,
            os=run_os,
            power_state=form.get("power_state", "plugged"),
            prompt=prompt,
            image_size=form.get("image_size", "").strip() or "512x512",
            steps=int(form["steps"]) if form.get("steps") else 4,
            cfg_scale=float(form["cfg_scale"]) if form.get("cfg_scale") else None,
            seed=int(form["seed"]) if form.get("seed") else None,
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            exclusive_run="exclusive_run" in form,
        )

        client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)
        attempted = {
            "model_name": cfg.model_name,
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "power_state": cfg.power_state,
        }
        try:
            result, _raw_trials = run_imagegen(client, prof.environment, cfg)
            validate_imagegen_result(result)
        except jsonschema.ValidationError as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "schema_validation", str(exc))
            flash(f"Result failed schema validation: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_imagegen", name=name))
        except Exception as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "imagegen_run", str(exc))
            flash(f"Run failed: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_imagegen", name=name))

        save_imagegen_result(app.config["RESULTS_DIR"], name, result)
        flash(
            f"Image-gen run complete: {result['metrics']['generation_time_ms']:.1f} ms, "
            f"{result['metrics']['images_per_sec']:.2f} images/sec.",
            "ok",
        )
        return redirect(url_for("profile_imagegen", name=name))

    @app.route("/profiles/<name>/audiogen", methods=["GET", "POST"])
    def profile_audiogen(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            audiogen_models = _models_for_recipes(capabilities.get("models") or [], {"acestep", "thinksound"})
            audiogen_backends = available_backends(
                capabilities.get("system_info") or {}, prof.environment, modality="Audio generation"
            )
            results = list_audiogen_results(app.config["RESULTS_DIR"], profile=name)
            return render_template(
                "audiogen.html",
                profile=prof,
                capabilities=capabilities,
                audiogen_models=audiogen_models,
                audiogen_backends=audiogen_backends,
                results=results,
                default_os=host_os(prof.environment),
            )

        form = request.form
        model_name = form.get("model_name", "").strip()
        prompt = form.get("prompt", "").strip()
        backend_val = form.get("backend", "").strip()
        compute_engine = form.get("compute_engine", "").strip()
        missing = [
            label for label, val in
            [("model", model_name), ("prompt", prompt), ("backend", backend_val), ("engine", compute_engine)]
            if not val
        ]
        if missing:
            flash(f"Missing required field(s): {', '.join(missing)}.", "error")
            return redirect(url_for("profile_audiogen", name=name))

        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_audiogen", name=name))

        caps = _live_capabilities(prof)
        issues = validate_combo_against_profile(
            compute_engine, backend_val, caps.get("system_info") or {}, prof.environment,
            modality="Audio generation",
        )
        if issues:
            for issue in issues:
                flash(issue.replace("\n", " "), "error")
            flash("Fix the engine/backend combination before running.", "error")
            return redirect(url_for("profile_audiogen", name=name))

        cfg = AudioGenConfig(
            model_name=model_name,
            compute_engine=compute_engine,
            backend=backend_val,
            os=run_os,
            power_state=form.get("power_state", "plugged"),
            prompt=prompt,
            lyrics=form.get("lyrics", "").strip() or None,
            vocal_language=form.get("vocal_language", "").strip() or None,
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            exclusive_run="exclusive_run" in form,
        )

        client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)
        attempted = {
            "model_name": cfg.model_name,
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "power_state": cfg.power_state,
        }
        try:
            result, _raw_trials = run_audiogen(client, prof.environment, cfg)
            validate_audiogen_result(result)
        except jsonschema.ValidationError as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "schema_validation", str(exc))
            flash(f"Result failed schema validation: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_audiogen", name=name))
        except Exception as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "audiogen_run", str(exc))
            flash(f"Run failed: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_audiogen", name=name))

        save_audiogen_result(app.config["RESULTS_DIR"], name, result)
        flash(
            f"Audio-gen run complete: {result['metrics']['real_time_factor']:.2f}x real-time "
            f"({result['metrics']['generation_time_ms']:.1f} ms for {result['metrics']['audio_duration_s']:.2f}s of audio).",
            "ok",
        )
        return redirect(url_for("profile_audiogen", name=name))

    @app.route("/profiles/<name>/embeddings", methods=["GET", "POST"])
    def profile_embeddings(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            # Embeddings-capable models ride the same llamacpp/FastFlowLM
            # serving path as ordinary chat models -- Lemonade's /api/v1/models
            # doesn't expose a separate "embeddings" recipe, so this list is
            # necessarily every llamacpp/flm model, not just embedding ones.
            embeddings_models = _models_for_recipes(capabilities.get("models") or [], {"llamacpp", "flm"})
            embeddings_backends = available_backends(
                capabilities.get("system_info") or {}, prof.environment, modality="Text generation"
            )
            results = list_embeddings_results(app.config["RESULTS_DIR"], profile=name)
            return render_template(
                "embeddings.html",
                profile=prof,
                capabilities=capabilities,
                embeddings_models=embeddings_models,
                embeddings_backends=embeddings_backends,
                results=results,
                default_os=host_os(prof.environment),
            )

        form = request.form
        model_name = form.get("model_name", "").strip()
        backend_val = form.get("backend", "").strip()
        compute_engine = form.get("compute_engine", "").strip()
        input_texts = [line.strip() for line in form.get("input_texts", "").splitlines() if line.strip()]
        missing = [
            label for label, val in
            [("model", model_name), ("backend", backend_val), ("engine", compute_engine)]
            if not val
        ]
        if not input_texts:
            missing.append("input text")
        if missing:
            flash(f"Missing required field(s): {', '.join(missing)}.", "error")
            return redirect(url_for("profile_embeddings", name=name))

        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_embeddings", name=name))

        caps = _live_capabilities(prof)
        issues = validate_combo_against_profile(
            compute_engine, backend_val, caps.get("system_info") or {}, prof.environment,
            modality="Text generation",
        )
        if issues:
            for issue in issues:
                flash(issue.replace("\n", " "), "error")
            flash("Fix the engine/backend combination before running.", "error")
            return redirect(url_for("profile_embeddings", name=name))

        cfg = EmbeddingsConfig(
            model_name=model_name,
            compute_engine=compute_engine,
            backend=backend_val,
            os=run_os,
            power_state=form.get("power_state", "plugged"),
            input_texts=input_texts,
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            exclusive_run="exclusive_run" in form,
        )

        client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)
        attempted = {
            "model_name": cfg.model_name,
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "power_state": cfg.power_state,
        }
        try:
            result, _raw_trials = run_embeddings(client, prof.environment, cfg)
            validate_embeddings_result(result)
        except jsonschema.ValidationError as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "schema_validation", str(exc))
            flash(f"Result failed schema validation: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_embeddings", name=name))
        except Exception as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "embeddings_run", str(exc))
            flash(f"Run failed: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_embeddings", name=name))

        save_embeddings_result(app.config["RESULTS_DIR"], name, result)
        flash(
            f"Embeddings run complete: {result['metrics']['latency_ms']:.1f} ms, "
            f"{result['metrics']['embeddings_per_sec']:.2f} embeddings/sec.",
            "ok",
        )
        return redirect(url_for("profile_embeddings", name=name))

    @app.route("/profiles/<name>/rerank", methods=["GET", "POST"])
    def profile_rerank(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            # Reranking is llamacpp-GGUF only (FastFlowLM doesn't support it) --
            # confirmed against Lemonade's own source -- but the models list
            # can't distinguish a reranker-labeled GGUF from a chat one either.
            rerank_models = _models_for_recipes(capabilities.get("models") or [], {"llamacpp"})
            rerank_backends = available_backends(
                capabilities.get("system_info") or {}, prof.environment, modality="Text generation"
            )
            results = list_rerank_results(app.config["RESULTS_DIR"], profile=name)
            return render_template(
                "rerank.html",
                profile=prof,
                capabilities=capabilities,
                rerank_models=rerank_models,
                rerank_backends=rerank_backends,
                results=results,
                default_os=host_os(prof.environment),
            )

        form = request.form
        model_name = form.get("model_name", "").strip()
        query = form.get("query", "").strip()
        backend_val = form.get("backend", "").strip()
        compute_engine = form.get("compute_engine", "").strip()
        documents = [line.strip() for line in form.get("documents", "").splitlines() if line.strip()]
        missing = [
            label for label, val in
            [("model", model_name), ("query", query), ("backend", backend_val), ("engine", compute_engine)]
            if not val
        ]
        if not documents:
            missing.append("documents")
        if missing:
            flash(f"Missing required field(s): {', '.join(missing)}.", "error")
            return redirect(url_for("profile_rerank", name=name))

        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_rerank", name=name))

        caps = _live_capabilities(prof)
        issues = validate_combo_against_profile(
            compute_engine, backend_val, caps.get("system_info") or {}, prof.environment,
            modality="Text generation",
        )
        if issues:
            for issue in issues:
                flash(issue.replace("\n", " "), "error")
            flash("Fix the engine/backend combination before running.", "error")
            return redirect(url_for("profile_rerank", name=name))

        cfg = RerankConfig(
            model_name=model_name,
            compute_engine=compute_engine,
            backend=backend_val,
            os=run_os,
            power_state=form.get("power_state", "plugged"),
            query=query,
            documents=documents,
            top_n=int(form["top_n"]) if form.get("top_n") else None,
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            exclusive_run="exclusive_run" in form,
        )

        client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)
        attempted = {
            "model_name": cfg.model_name,
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "power_state": cfg.power_state,
        }
        try:
            result, _raw_trials = run_rerank(client, prof.environment, cfg)
            validate_rerank_result(result)
        except jsonschema.ValidationError as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "schema_validation", str(exc))
            flash(f"Result failed schema validation: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_rerank", name=name))
        except Exception as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "rerank_run", str(exc))
            flash(f"Run failed: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_rerank", name=name))

        save_rerank_result(app.config["RESULTS_DIR"], name, result)
        flash(
            f"Rerank run complete: {result['metrics']['latency_ms']:.1f} ms, "
            f"{result['metrics']['documents_per_sec']:.2f} documents/sec.",
            "ok",
        )
        return redirect(url_for("profile_rerank", name=name))

    @app.route("/profiles/<name>/meshgen", methods=["GET", "POST"])
    def profile_meshgen(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            meshgen_models = _models_for_recipes(capabilities.get("models") or [], {"trellis"})
            meshgen_backends = available_backends(
                capabilities.get("system_info") or {}, prof.environment, modality="3D generation"
            )
            results = list_meshgen_results(app.config["RESULTS_DIR"], profile=name)
            return render_template(
                "meshgen.html",
                profile=prof,
                capabilities=capabilities,
                meshgen_models=meshgen_models,
                meshgen_backends=meshgen_backends,
                results=results,
                default_os=host_os(prof.environment),
            )

        form = request.form
        model_name = form.get("model_name", "").strip()
        backend_val = form.get("backend", "").strip()
        compute_engine = form.get("compute_engine", "").strip()
        image_file = request.files.get("input_image")
        missing = [
            label for label, val in
            [("model", model_name), ("backend", backend_val), ("engine", compute_engine)]
            if not val
        ]
        if not image_file or not image_file.filename:
            missing.append("input image")
        if missing:
            flash(f"Missing required field(s): {', '.join(missing)}.", "error")
            return redirect(url_for("profile_meshgen", name=name))

        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_meshgen", name=name))

        caps = _live_capabilities(prof)
        issues = validate_combo_against_profile(
            compute_engine, backend_val, caps.get("system_info") or {}, prof.environment,
            modality="3D generation",
        )
        if issues:
            for issue in issues:
                flash(issue.replace("\n", " "), "error")
            flash("Fix the engine/backend combination before running.", "error")
            return redirect(url_for("profile_meshgen", name=name))

        cfg = MeshGenConfig(
            model_name=model_name,
            compute_engine=compute_engine,
            backend=backend_val,
            os=run_os,
            power_state=form.get("power_state", "plugged"),
            input_image_bytes=image_file.read(),
            resolution=form.get("resolution", "").strip() or None,
            bg_removal=form.get("bg_removal", "").strip() or None,
            uv=form.get("uv", "").strip() or None,
            seed=int(form["seed"]) if form.get("seed") else None,
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            exclusive_run="exclusive_run" in form,
        )

        client = LemonadeClient(prof.base_url, api_key=prof.api_key, timeout=BENCH_TIMEOUT)
        attempted = {
            "model_name": cfg.model_name,
            "compute_engine": cfg.compute_engine,
            "backend": cfg.backend,
            "power_state": cfg.power_state,
        }
        try:
            result, _raw_trials = run_meshgen(client, prof.environment, cfg)
            validate_meshgen_result(result)
        except jsonschema.ValidationError as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "schema_validation", str(exc))
            flash(f"Result failed schema validation: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_meshgen", name=name))
        except Exception as exc:
            save_failure(app.config["RESULTS_DIR"], name, attempted, "meshgen_run", str(exc))
            flash(f"Run failed: {exc} (saved to failure log)", "error")
            return redirect(url_for("profile_meshgen", name=name))

        save_meshgen_result(app.config["RESULTS_DIR"], name, result)
        flash(
            f"Mesh-gen run complete: {result['metrics']['generation_time_ms']:.1f} ms, "
            f"{result['metrics']['meshes_per_sec']:.2f} meshes/sec.",
            "ok",
        )
        return redirect(url_for("profile_meshgen", name=name))

    @app.route("/profiles/<name>/debug")
    def profile_debug(name: str):
        """Raw /api/v1/system-info and /api/v1/health for this profile --
        dashboard equivalent of `lemonmatrix profile debug`, for diagnosing
        discovery/field-mapping mismatches without a terminal."""
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        client = LemonadeClient(prof.base_url, api_key=prof.api_key)
        try:
            system_info = client.system_info()
            health = client.health()
            error = None
        except Exception as exc:
            system_info, health, error = {}, {}, str(exc)

        return render_template(
            "profile_debug.html",
            profile=prof,
            error=error,
            system_info_json=json.dumps(system_info, indent=2),
            health_json=json.dumps(health, indent=2),
        )

    @app.route("/profiles/<name>/queue", methods=["GET", "POST"])
    def profile_queue(name: str):
        """A hand-built run queue: unlike /sweep (one model's pulled
        quantizations x a Cartesian product of engine/backend/power axes),
        this lets you add arbitrary specific combinations one at a time --
        different models, different backends, even a mix of model and
        router runs -- then executes the queue sequentially in the
        background, reusing the exact same SweepBatch/SweepStore machinery
        (and the same one-active-batch-per-profile guard) as /sweep.
        """
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            models_by_base = _group_by_base_model(_model_options(capabilities.get("models") or []))
            return render_template(
                "queue.html",
                profile=prof,
                capabilities=capabilities,
                models_by_base=models_by_base,
                models_by_base_json=json.dumps(models_by_base),
                compat_map_json=_compat_map_json(capabilities.get("backends") or []),
                default_os=host_os(prof.environment),
                running_batch=_running_batch_for_profile(app, name),
                max_combinations=MAX_SWEEP_COMBINATIONS,
                routers=available_routers(capabilities.get("models") or []),
            )

        if _running_batch_for_profile(app, name):
            flash("A sweep/queue is already running against this profile -- wait for it to finish first.", "error")
            return redirect(url_for("profile_queue", name=name))

        form = request.form
        try:
            queue_items = json.loads(form.get("queue_json", "[]"))
        except json.JSONDecodeError:
            queue_items = None
        if not isinstance(queue_items, list) or not queue_items:
            flash("Add at least one run to the queue before starting it.", "error")
            return redirect(url_for("profile_queue", name=name))

        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_queue", name=name))

        caps = _live_capabilities(prof)
        system_info = caps.get("system_info") or {}

        # Re-validate every item server-side -- the queue-builder's own JS
        # already keeps a user from adding an incompatible pair (see
        # lmRestrictOptions() in queue.html), but the live instance can
        # change between building the queue and submitting it (a backend
        # finishing an install, hardware capabilities changing), so this is
        # not just defense-in-depth against a hand-crafted POST.
        combos = []
        skipped = []
        has_router_item = False
        for raw in queue_items:
            if not isinstance(raw, dict):
                continue
            model_name = str(raw.get("model_name") or "").strip()
            if not model_name:
                continue
            item_run_type = raw.get("run_type") if raw.get("run_type") in ("model", "router") else "model"

            if item_run_type == "router":
                has_router_item = True
                combos.append(
                    {
                        "model_name": model_name,
                        "model_class": "router",
                        "quantization": "none",
                        "context_length": int(raw.get("context_length") or 4096),
                        "compute_engine": "router",
                        "backend": "collection.router",
                        "power_state": raw.get("power_state") or "plugged",
                        "run_type": "router",
                    }
                )
                continue

            compute_engine = str(raw.get("compute_engine") or "").strip()
            backend_val = str(raw.get("backend") or "").strip()
            issues = validate_combo_against_profile(compute_engine, backend_val, system_info, prof.environment)
            if issues:
                skipped.append(f"{model_name} ({compute_engine}/{backend_val})")
                continue
            combos.append(
                {
                    "model_name": model_name,
                    "model_class": raw.get("model_class") or "dense",
                    "quantization": str(raw.get("quantization") or "").strip() or "unknown",
                    "context_length": int(raw.get("context_length") or 4096),
                    "compute_engine": compute_engine,
                    "backend": backend_val,
                    "power_state": raw.get("power_state") or "plugged",
                }
            )

        if skipped:
            flash(
                f"Dropped {len(skipped)} queued run(s) that are no longer valid against this profile: "
                f"{', '.join(skipped[:5])}{'…' if len(skipped) > 5 else ''}.",
                "error",
            )
        if not combos:
            flash("No valid queued runs remain after re-checking them against this profile.", "error")
            return redirect(url_for("profile_queue", name=name))
        if len(combos) > MAX_SWEEP_COMBINATIONS:
            flash(f"{len(combos)} queued run(s) exceeds the {MAX_SWEEP_COMBINATIONS} limit -- remove some.", "error")
            return redirect(url_for("profile_queue", name=name))

        via_job_engine = "via_job_engine" in form
        if via_job_engine and has_router_item:
            flash("Job-engine execution doesn't support router runs -- uncheck it or remove the router item(s) from the queue.", "error")
            return redirect(url_for("profile_queue", name=name))

        batch = SweepBatch(name, combos)
        app.config["SWEEP_BATCHES"][batch.id] = batch
        start_batch(
            batch,
            base_url=prof.base_url,
            api_key=prof.api_key,
            environment=prof.environment,
            results_dir=app.config["RESULTS_DIR"],
            model_class="dense",  # unused per-combo default -- every item carries its own model_class
            os_name=run_os,
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            max_tokens=int(form.get("max_tokens") or 256),
            exclusive_run="exclusive_run" in form,
            energy_price_usd_per_kwh=float(form["energy_price_usd_per_kwh"]) if form.get("energy_price_usd_per_kwh") else None,
            hardware_cost_usd=float(form["hardware_cost_usd"]) if form.get("hardware_cost_usd") else None,
            hardware_lifetime_hours=float(form["hardware_lifetime_hours"]) if form.get("hardware_lifetime_hours") else None,
            via_job_engine=via_job_engine,
            run_type="model",  # unused per-combo default -- router items carry their own run_type
            store=app.config["SWEEP_STORE"],
        )
        flash(f"Started a queue of {len(combos)} run(s) in the background.", "ok")
        return redirect(url_for("sweep_status", name=name, batch_id=batch.id))

    @app.route("/profiles/<name>/sweep", methods=["GET", "POST"])
    def profile_sweep(name: str):
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        if request.method == "GET":
            capabilities = _live_capabilities(prof)
            models_by_base = _group_by_base_model(_model_options(capabilities.get("models") or []))
            return render_template(
                "sweep_form.html",
                profile=prof,
                capabilities=capabilities,
                models_by_base=models_by_base,
                models_by_base_json=json.dumps(models_by_base),
                compat_map_json=_compat_map_json(capabilities.get("backends") or []),
                default_os=host_os(prof.environment),
                running_batch=_running_batch_for_profile(app, name),
                max_combinations=MAX_SWEEP_COMBINATIONS,
                routers=available_routers(capabilities.get("models") or []),
            )

        if _running_batch_for_profile(app, name):
            flash("A sweep is already running against this profile -- wait for it to finish first.", "error")
            return redirect(url_for("profile_sweep", name=name))

        form = request.form
        run_type = form.get("run_type", "model").strip() or "model"
        power_states = request.form.getlist("power_states")

        run_os = host_os(prof.environment)
        if not run_os:
            flash("Couldn't determine this profile's OS from its environment -- fix discovery on the profile page first.", "error")
            return redirect(url_for("profile_sweep", name=name))

        if run_type == "router":
            # Router sweeps: the only real axis is "which router" x power
            # state -- engine/backend/quant are fixed facts of a router, same
            # auto-fill as the CLI's `sweep --run-type router`, and there's
            # nothing to validate_combo_against_profile since a router
            # controls its own backend selection internally.
            selected_routers = request.form.getlist("router_models")
            if not (selected_routers and power_states):
                flash("Pick at least one router and power state.", "error")
                return redirect(url_for("profile_sweep", name=name))
            if "via_job_engine" in form:
                flash("Job-engine execution doesn't support router runs (a router has no fixed backend/ctx_size for a job's load step).", "error")
                return redirect(url_for("profile_sweep", name=name))

            context_length = int(form["router_context_length"]) if form.get("router_context_length") else 4096
            combos = [
                {
                    "model_name": router_id,
                    "quantization": "none",
                    "context_length": context_length,
                    "compute_engine": "router",
                    "backend": "collection.router",
                    "power_state": power_state,
                }
                for router_id, power_state in itertools.product(selected_routers, power_states)
            ]
            model_class = "router"
            via_job_engine = False
        else:
            model_base = form.get("model_base", "").strip()
            selected_quants = request.form.getlist("quantizations")
            engines = request.form.getlist("engines")
            backends = request.form.getlist("backends")
            if not (model_base and selected_quants and engines and backends and power_states):
                flash("Pick at least one quantization, engine, backend, and power state.", "error")
                return redirect(url_for("profile_sweep", name=name))

            capabilities = _live_capabilities(prof)
            models_by_base = _group_by_base_model(_model_options(capabilities.get("models") or []))
            variants = [v for v in models_by_base.get(model_base, []) if v["quantization"] in selected_quants]
            if not variants:
                flash(f"No pulled quantizations matched the selection for {model_base}.", "error")
                return redirect(url_for("profile_sweep", name=name))

            raw_combos = expand_combinations(variants, engines, backends, power_states)
            caps_post = _live_capabilities(prof)
            si_post = caps_post.get("system_info") or {}
            skipped_labels = []
            combos = []
            for c in raw_combos:
                issues = validate_combo_against_profile(c["compute_engine"], c["backend"], si_post, prof.environment)
                if issues:
                    skipped_labels.append(f"{c['compute_engine']}/{c['backend']}")
                else:
                    combos.append(c)
            if skipped_labels:
                unique_pairs = sorted(set(skipped_labels))
                flash(
                    f"Dropped {len(skipped_labels)} impossible combination(s) "
                    f"({', '.join(unique_pairs[:5])}{'…' if len(unique_pairs) > 5 else ''}) — "
                    "engine/backend not compatible or backend not installed.",
                    "error",
                )
            if not combos:
                flash("No valid combinations remain after filtering impossible engine/backend pairs.", "error")
                return redirect(url_for("profile_sweep", name=name))
            model_class = form.get("model_class", "dense")
            via_job_engine = "via_job_engine" in form

        if len(combos) > MAX_SWEEP_COMBINATIONS:
            flash(f"{len(combos)} combinations exceeds the {MAX_SWEEP_COMBINATIONS} limit -- narrow your selection.", "error")
            return redirect(url_for("profile_sweep", name=name))

        batch = SweepBatch(name, combos)
        app.config["SWEEP_BATCHES"][batch.id] = batch
        start_batch(
            batch,
            base_url=prof.base_url,
            api_key=prof.api_key,
            environment=prof.environment,
            results_dir=app.config["RESULTS_DIR"],
            model_class=model_class,
            os_name=run_os,
            power_cap_w=float(form["power_cap_w"]) if form.get("power_cap_w") else None,
            warmup_trials=int(form.get("warmup_trials") or 2),
            measured_trials=int(form.get("measured_trials") or 5),
            max_tokens=int(form.get("max_tokens") or 256),
            exclusive_run="exclusive_run" in form,
            energy_price_usd_per_kwh=float(form["energy_price_usd_per_kwh"]) if form.get("energy_price_usd_per_kwh") else None,
            hardware_cost_usd=float(form["hardware_cost_usd"]) if form.get("hardware_cost_usd") else None,
            hardware_lifetime_hours=float(form["hardware_lifetime_hours"]) if form.get("hardware_lifetime_hours") else None,
            via_job_engine=via_job_engine,
            run_type=run_type,
            store=app.config["SWEEP_STORE"],
        )
        flash(f"Started a sweep of {len(combos)} combination(s) in the background.", "ok")
        return redirect(url_for("sweep_status", name=name, batch_id=batch.id))

    @app.route("/profiles/<name>/sweeps/<batch_id>")
    def sweep_status(name: str, batch_id: str):
        try:
            Profile.load(name)
        except FileNotFoundError:
            abort(404)
        batch = app.config["SWEEP_BATCHES"].get(batch_id)
        if batch is None or batch.profile_name != name:
            abort(404)
        return render_template("sweep_status.html", profile_name=name, batch=batch)

    @app.route("/results/<profile>/<run_id>")
    def result_detail(profile: str, run_id: str):
        result = load_result(app.config["RESULTS_DIR"], profile, run_id)
        if result is None:
            abort(404)
        raw = {k: v for k, v in result.items() if not k.startswith("_")}
        has_trials = load_trials(app.config["RESULTS_DIR"], profile, run_id) is not None
        return render_template(
            "result_detail.html", result=result, result_json=json.dumps(raw, indent=2), has_trials=has_trials
        )

    @app.route("/results/<profile>/<run_id>/trials")
    def result_trials(profile: str, run_id: str):
        """Raw per-trial measurements and (for router runs) per-trial route
        traces -- dashboard equivalent of `lemonmatrix trials`."""
        result = load_result(app.config["RESULTS_DIR"], profile, run_id)
        if result is None:
            abort(404)
        trials_data = load_trials(app.config["RESULTS_DIR"], profile, run_id)
        if trials_data is None:
            abort(404)
        return render_template(
            "trials_detail.html",
            result=result,
            trials_json=json.dumps(trials_data, indent=2),
        )

    # ------------------------------------------------------------------
    # Profile management — refresh and delete
    # ------------------------------------------------------------------

    @app.route("/profiles/<name>/refresh", methods=["POST"])
    def profile_refresh(name: str):
        """Re-run discovery against the existing base_url and overwrite the profile."""
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        try:
            updated, gaps = connect_and_save(name, prof.base_url, prof.api_key)
        except ConnectionError as exc:
            flash(f"Refresh failed: {exc}", "error")
            return redirect(url_for("profile_detail", name=name))

        msg = f"Profile '{name}' refreshed."
        if gaps:
            msg += f" Still missing: {', '.join(gaps)}."
        flash(msg, "ok")
        return redirect(url_for("profile_detail", name=name))

    @app.route("/profiles/<name>/delete", methods=["POST"])
    def profile_delete(name: str):
        """Delete a saved profile file."""
        try:
            prof = Profile.load(name)
        except FileNotFoundError:
            abort(404)

        prof.path().unlink(missing_ok=True)
        flash(f"Profile '{name}' deleted.", "ok")
        return redirect(url_for("profiles"))

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    @app.route("/export/results.csv")
    def export_results_csv():
        results = list_results(app.config["RESULTS_DIR"])

        profile_filter = request.args.get("profile", "")
        valid_only = request.args.get("valid_only", "")
        if profile_filter:
            results = [r for r in results if r.get("_profile") == profile_filter]
        if valid_only:
            results = [r for r in results if get_path(r, "validity.valid")]

        csv_text = results_to_csv(results)
        return Response(
            csv_text,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=lemonmatrix-results.csv"},
        )

    return app
