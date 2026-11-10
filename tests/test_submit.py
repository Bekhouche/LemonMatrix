import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from lemonmatrix.cli import cli
from lemonmatrix.submit import SubmitError, plan, pr_text, submit

RESULT = {
    "run_type": "router",
    "model": {"name": "user.Router"},
    "config": {"compute_engine": "router", "backend": "collection.router"},
    "metrics": {"decode": {"tokens_per_sec": 99.5}},
    "validity": {"valid": True},
}


def _write(results_dir: Path, rel: str, data: dict) -> Path:
    path = results_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _accept(path, data):
    return None


def test_plan_keeps_the_local_layout(tmp_path):
    results = tmp_path / "results"
    a = _write(results, "box/run-1.json", RESULT)
    b = _write(results, "box/tts/run-2.json", {"model": {"name": "kokoro"}, "config": {}})

    subs = plan([a, b], results, _accept)

    assert [s.target for s in subs] == ["results/box/run-1.json", "results/box/tts/run-2.json"]
    assert subs[0].summary == "user.Router on router / collection.router, 99.5 tok/s, valid"


def test_plan_rejects_files_outside_the_results_dir(tmp_path):
    outside = _write(tmp_path, "elsewhere/run.json", RESULT)
    with pytest.raises(SubmitError, match="not under"):
        plan([outside], tmp_path / "results", _accept)


@pytest.mark.parametrize("rel", ["run.json", "box/trials/run.json", "box/failures/run.json"])
def test_plan_rejects_non_result_files(tmp_path, rel):
    results = tmp_path / "results"
    path = _write(results, rel, RESULT)
    with pytest.raises(SubmitError):
        plan([path], results, _accept)


def test_plan_stops_on_a_validation_failure(tmp_path):
    results = tmp_path / "results"
    path = _write(results, "box/run.json", RESULT)

    def reject(p, data):
        raise ValueError("bad")

    with pytest.raises(ValueError):
        plan([path], results, reject)


def test_pr_text_names_profiles_and_files(tmp_path):
    results = tmp_path / "results"
    subs = plan([_write(results, "box/run.json", RESULT)], results, _accept)
    title, body = pr_text(subs)
    assert title == "Add 1 result(s) from box"
    assert "`results/box/run.json`" in body


class FakeGitHub:
    def __init__(self, fork_exists: bool):
        self.fork_exists = fork_exists
        self.calls: list[list[str]] = []
        self.added: list[str] = []

    def __call__(self, args, cwd=None):
        self.calls.append(args)
        if args[:3] == ["gh", "api", "user"]:
            return "contributor"
        if args[:2] == ["gh", "api"] and args[2].endswith("/forks"):
            return "contributor/my-fork" if self.fork_exists else ""
        if args[:3] == ["gh", "repo", "fork"]:
            self.fork_exists = True
            return ""
        if args[:3] == ["gh", "repo", "clone"]:
            Path(args[4]).mkdir(parents=True)
            return ""
        if args[:2] == ["git", "add"]:
            assert (Path(cwd) / args[2]).exists()
            self.added.append(args[2])
        if args[:3] == ["gh", "pr", "create"]:
            return "https://github.com/owner/site/pull/7"
        return ""


def _subs(tmp_path):
    results = tmp_path / "results"
    return plan([_write(results, "box/run.json", RESULT)], results, _accept)


def test_submit_reuses_an_existing_fork_and_opens_the_pr(tmp_path):
    gh = FakeGitHub(fork_exists=True)

    url = submit(_subs(tmp_path), "owner/site", runner=gh)

    assert url == "https://github.com/owner/site/pull/7"
    assert not any(c[:3] == ["gh", "repo", "fork"] for c in gh.calls)
    assert gh.added == ["results/box/run.json"]
    pr = next(c for c in gh.calls if c[:3] == ["gh", "pr", "create"])
    assert pr[pr.index("--repo") + 1] == "owner/site"
    assert pr[pr.index("--head") + 1].startswith("contributor:results-")


def test_submit_creates_a_fork_when_there_is_none(tmp_path):
    gh = FakeGitHub(fork_exists=False)

    submit(_subs(tmp_path), "owner/site", runner=gh)

    assert any(c[:3] == ["gh", "repo", "fork"] for c in gh.calls)


def test_submit_branches_from_upstream_main(tmp_path):
    gh = FakeGitHub(fork_exists=True)

    submit(_subs(tmp_path), "owner/site", runner=gh)

    fetch = next(c for c in gh.calls if c[:2] == ["git", "fetch"])
    assert fetch[-2:] == ["https://github.com/owner/site.git", "main"]
    switch = next(c for c in gh.calls if c[:2] == ["git", "switch"])
    assert switch[-1] == "FETCH_HEAD"
    assert not any(c[:3] == ["git", "remote", "add"] for c in gh.calls)


def test_cli_dry_run_validates_without_touching_github(tmp_path, monkeypatch):
    import lemonmatrix.cli as cli_mod

    monkeypatch.setattr(cli_mod, "submit", lambda *a, **k: pytest.fail("dry run must not submit"))
    imported = {
        "schema_version": "0.1.0", "run_type": "imported", "import_id": "0123456789abcdef",
        "imported_at": "2026-11-07T00:00:00+00:00",
        "source": {"tool": "lemonade_lab", "file_sha256": "0" * 64},
        "environment": {}, "rows": [{"workload": "llm", "model": "m", "backend": "npu", "value": 1.0, "unit": "tok/s"}],
        "validity": {"validated": False, "notes": "n"},
    }
    results = tmp_path / "results"
    path = _write(results, "imported/lemonade_lab/0123456789abcdef.json", imported)

    result = CliRunner().invoke(cli, ["submit", str(path), "--results-dir", str(results), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output


def test_cli_refuses_a_file_that_fails_its_schema(tmp_path):
    results = tmp_path / "results"
    path = _write(results, "box/run.json", {"not": "a result"})

    result = CliRunner().invoke(cli, ["submit", str(path), "--results-dir", str(results), "--dry-run"])

    assert result.exit_code != 0
    assert "fails schema validation" in result.output
