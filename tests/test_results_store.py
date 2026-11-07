from lemonmatrix.results_store import list_failures, list_results, save_failure


def test_save_failure_writes_under_profile_failures_subdir(tmp_path):
    path = save_failure(
        tmp_path,
        "demo",
        attempted={"model_name": "X", "backend": "llamacpp-cuda"},
        stage="load",
        error="Failed to load model 'X': llama-server failed to start",
    )
    assert path.parent.name == "failures"
    assert path.parent.parent.name == "demo"


def test_list_failures_scoped_to_one_profile(tmp_path):
    save_failure(tmp_path, "demo", {"model_name": "A"}, "load", "boom A")
    save_failure(tmp_path, "other", {"model_name": "B"}, "load", "boom B")

    demo_failures = list_failures(tmp_path, "demo")
    assert len(demo_failures) == 1
    assert demo_failures[0]["attempted"]["model_name"] == "A"
    assert demo_failures[0]["_profile"] == "demo"

    all_failures = list_failures(tmp_path)
    assert len(all_failures) == 2


def test_list_failures_newest_first(tmp_path):
    save_failure(tmp_path, "demo", {}, "load", "first")
    import time

    time.sleep(0.01)
    save_failure(tmp_path, "demo", {}, "load", "second")

    failures = list_failures(tmp_path, "demo")
    assert failures[0]["error"] == "second"
    assert failures[1]["error"] == "first"


def test_list_failures_empty_when_no_results_dir(tmp_path):
    assert list_failures(tmp_path / "does-not-exist") == []


def test_save_failure_includes_batch_id_when_given(tmp_path):
    path = save_failure(tmp_path, "demo", {}, "sweep_item", "boom", batch_id="abc123")
    import json

    record = json.loads(path.read_text())
    assert record["batch_id"] == "abc123"


def test_failures_do_not_appear_in_list_results(tmp_path):
    save_failure(tmp_path, "demo", {"model_name": "A"}, "load", "boom")
    # A failures/ subdirectory is one level deeper than list_results()'s
    # <profile>/<run_id>.json glob, so it must never show up there.
    assert list_results(tmp_path) == []
