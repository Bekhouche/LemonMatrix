"""Open a pull request that adds local results to the public leaderboard repo.

The flow is the one a contributor would follow by hand: fork the site repo
(or reuse an existing fork), branch from the upstream main, copy each result
to the same results/<profile>/... path it has locally, together with the
run's per-trial file (<profile>/trials/<run_id>.json) when there is one, so
the run page can show its trials; commit, push to the fork and open a PR. The site repo's CI then validates the files again before
anyone merges them. Everything goes through ``git`` and the GitHub CLI
(``gh``), using whatever account ``gh`` is logged into.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SITE_REPO = "LemonMatrix/lemonmatrix.github.io"


class SubmitError(RuntimeError):
    pass


def run(args: list[str], cwd: Path | None = None) -> str:
    """Run a command and return stdout; raise SubmitError with stderr on failure."""
    try:
        done = subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError as exc:
        raise SubmitError(f"{args[0]} is not installed or not on PATH") from exc
    if done.returncode != 0:
        raise SubmitError(f"{' '.join(args[:3])} failed: {(done.stderr or done.stdout).strip()}")
    return done.stdout.strip()


@dataclass
class Submission:
    source: Path
    target: str  # path inside the site repo, e.g. results/my-machine/<run_id>.json
    summary: str
    sidecar: bool = False  # a run's per-trial file, sent along with its result


def _summary(data: dict) -> str:
    if data.get("run_type") == "imported":
        rows = len(data.get("rows") or [])
        return f"imported from {data.get('source', {}).get('tool')}: {rows} row(s)"
    model = (data.get("model") or {}).get("name", "?")
    config = data.get("config") or {}
    valid = (data.get("validity") or {}).get("valid")
    decode = ((data.get("metrics") or {}).get("decode") or {}).get("tokens_per_sec")
    metric = f", {decode:.1f} tok/s" if isinstance(decode, (int, float)) else ""
    state = "" if valid is None else (", valid" if valid else ", invalid")
    return f"{model} on {config.get('compute_engine', '?')} / {config.get('backend', '?')}{metric}{state}"


def plan(paths: list[Path], results_dir: Path, validate) -> list[Submission]:
    """Validate each file and work out where it goes in the site repo.

    `validate(path, data)` raises on a schema failure. Files must live under
    results_dir so their profile/modality layout can be kept.
    """
    results_dir = results_dir.resolve()
    submissions = []
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(results_dir)
        except ValueError as exc:
            raise SubmitError(f"{path} is not under {results_dir}; pass --results-dir") from exc
        if len(relative.parts) < 2 or relative.parts[-2] in ("failures", "trials"):
            raise SubmitError(f"{path} is not a result file (expected <profile>/.../<id>.json)")
        data = json.loads(resolved.read_text())
        validate(resolved, data)
        submissions.append(Submission(resolved, (Path("results") / relative).as_posix(), _summary(data)))
        trials = _trials_file(results_dir, relative)
        if trials is not None:
            submissions.append(Submission(
                trials, (Path("results") / trials.relative_to(results_dir)).as_posix(),
                "per-trial measurements", sidecar=True,
            ))
    if not submissions:
        raise SubmitError("nothing to submit")
    return submissions


def _trials_file(results_dir: Path, relative: Path) -> Path | None:
    """The per-trial file saved next to a model or router result, if any.

    Only <profile>/<run_id>.json results have one (results_store.save_trials).
    A file whose run_id doesn't match, or that has no trials list, is refused
    rather than silently sent along.
    """
    if len(relative.parts) != 2:
        return None
    path = results_dir / relative.parts[0] / "trials" / relative.name
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SubmitError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict) or data.get("run_id") != relative.stem or not isinstance(data.get("trials"), list):
        raise SubmitError(f"{path} is not the per-trial file of {relative.stem}")
    return path


def pr_text(submissions: list[Submission]) -> tuple[str, str]:
    results = [s for s in submissions if not s.sidecar]
    sidecars = len(submissions) - len(results)
    profiles = sorted({Path(s.target).parts[1] for s in results})
    title = f"Add {len(results)} result(s) from {', '.join(profiles)}"
    lines = ["Submitted with `lemonmatrix submit`.", ""]
    lines += [f"- `{s.target}`: {s.summary}" for s in results]
    if sidecars:
        lines += ["", f"With the per-trial file of {sidecars} run(s) under `trials/`."]
    return title, "\n".join(lines) + "\n"


def _fork_of(repo: str, login: str, runner) -> str:
    """Full name of `login`'s fork of `repo`, creating it if needed."""
    owner_repo = runner(["gh", "api", f"repos/{repo}/forks", "--paginate", "--jq",
                         f'.[] | select(.owner.login == "{login}") | .full_name'])
    if owner_repo:
        return owner_repo.splitlines()[0]
    runner(["gh", "repo", "fork", repo, "--clone=false"])
    owner_repo = runner(["gh", "api", f"repos/{repo}/forks", "--paginate", "--jq",
                         f'.[] | select(.owner.login == "{login}") | .full_name'])
    if not owner_repo:
        raise SubmitError(f"could not find {login}'s fork of {repo} after creating it")
    return owner_repo.splitlines()[0]


def submit(submissions: list[Submission], repo: str = DEFAULT_SITE_REPO, runner=run) -> str:
    """Fork, branch, copy, commit, push and open the PR. Returns the PR URL."""
    login = runner(["gh", "api", "user", "--jq", ".login"])
    fork = _fork_of(repo, login, runner)
    branch = "results-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    title, body = pr_text(submissions)

    workdir = Path(tempfile.mkdtemp(prefix="lemonmatrix-submit-"))
    try:
        clone = workdir / "site"
        runner(["gh", "repo", "clone", fork, str(clone), "--", "--depth", "1"])
        # gh adds an "upstream" remote when it clones a fork, so fetch by URL instead.
        runner(["git", "fetch", "--depth", "1", f"https://github.com/{repo}.git", "main"], cwd=clone)
        runner(["git", "switch", "-c", branch, "FETCH_HEAD"], cwd=clone)
        for s in submissions:
            target = clone / s.target
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(s.source, target)
            runner(["git", "add", s.target], cwd=clone)
        runner(["git", "commit", "-m", title], cwd=clone)
        runner(["git", "push", "origin", branch], cwd=clone)
        return runner(["gh", "pr", "create", "--repo", repo, "--base", "main",
                       "--head", f"{fork.split('/')[0]}:{branch}", "--title", title, "--body", body])
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
