import time

from lemonmatrix.client import LemonadeClient
from lemonmatrix.profile import discover_environment
from lemonmatrix.sweep_batch import SweepBatch, _run_batch, expand_combinations, start_batch


def test_expand_combinations_computes_cartesian_product():
    variants = [
        {"id": "model-Q4_K_M", "quantization": "Q4_K_M", "context_length": 4096},
        {"id": "model-Q8_0", "quantization": "Q8_0", "context_length": 4096},
    ]
    combos = expand_combinations(variants, engines=["cpu", "dgpu"], backends=["llamacpp-vulkan"], power_states=["plugged"])
    assert len(combos) == 4  # 2 variants x 2 engines x 1 backend x 1 power_state
    assert {c["model_name"] for c in combos} == {"model-Q4_K_M", "model-Q8_0"}
    assert {c["compute_engine"] for c in combos} == {"cpu", "dgpu"}
    assert all(c["backend"] == "llamacpp-vulkan" and c["power_state"] == "plugged" for c in combos)


def test_sweep_batch_tracks_counts():
    batch = SweepBatch("demo", [{"model_name": "a"}, {"model_name": "b"}, {"model_name": "c"}])
    assert batch.total_count == 3
    assert batch.completed_count == 0
    assert all(i["status"] == "pending" for i in batch.items)

    batch.items[0]["status"] = "completed"
    batch.items[1]["status"] = "failed"
    assert batch.completed_count == 2  # completed + failed both count as "done"
    assert batch.failed_count == 1


def _run_batch_kwargs(environment, **overrides):
    defaults = dict(
        base_url=None,
        api_key=None,
        environment=environment,
        results_dir=None,
        model_class="dense",
        os_name="linux",
        power_cap_w=None,
        warmup_trials=1,
        measured_trials=1,
        max_tokens=32,
        exclusive_run=True,
        energy_price_usd_per_kwh=None,
    )
    defaults.update(overrides)
    return defaults


def test_run_batch_executes_all_items_and_writes_results(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    combos = expand_combinations(
        [{"id": "Llama-3.1-8B-Instruct-GGUF", "quantization": "Q4_K_M", "context_length": 4096}],
        engines=["igpu"],
        backends=["llama.cpp-vulkan", "llama.cpp-cuda"],
        power_states=["plugged"],
    )
    batch = SweepBatch("demo", combos)

    _run_batch(batch, **_run_batch_kwargs(environment, base_url=fake_lemonade, results_dir=tmp_path))

    assert batch.status == "done"
    assert batch.completed_count == 2
    assert batch.failed_count == 0
    assert all(i["run_id"] for i in batch.items)
    written = list((tmp_path / "demo").glob("*.json"))
    assert len(written) == 2


def test_run_batch_via_job_engine_executes_all_items_and_writes_results(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    combos = expand_combinations(
        [{"id": "Llama-3.1-8B-Instruct-GGUF", "quantization": "Q4_K_M", "context_length": 4096}],
        engines=["igpu"],
        backends=["llama.cpp-vulkan"],
        power_states=["plugged"],
    )
    batch = SweepBatch("demo", combos)

    _run_batch(batch, **_run_batch_kwargs(environment, base_url=fake_lemonade, results_dir=tmp_path, via_job_engine=True))

    assert batch.status == "done"
    assert batch.completed_count == 1
    assert batch.failed_count == 0
    written = list((tmp_path / "demo").glob("*.json"))
    assert len(written) == 1


def test_run_batch_marks_unresolvable_backend_invalid_not_failed(fake_lemonade, tmp_path):
    # An unresolvable backend doesn't raise inside run_sweep (see bench.py):
    # it completes with validity.valid=False and a note, since load() still
    # succeeds (just without a backend selector). Confirm that distinction
    # holds inside a batch too -- this is "completed", not "failed".
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    combos = expand_combinations(
        [{"id": "Llama-3.1-8B-Instruct-GGUF", "quantization": "Q4_K_M", "context_length": 4096}],
        engines=["igpu"],
        backends=["not-a-real-backend"],
        power_states=["plugged"],
    )
    batch = SweepBatch("demo", combos)

    _run_batch(batch, **_run_batch_kwargs(environment, base_url=fake_lemonade, results_dir=tmp_path))

    assert batch.items[0]["status"] == "completed"
    assert batch.items[0]["run_id"] is not None


def test_run_batch_continues_after_one_item_genuinely_fails(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    combos = expand_combinations(
        [{"id": "Llama-3.1-8B-Instruct-GGUF", "quantization": "Q4_K_M", "context_length": 4096}],
        engines=["igpu"],
        backends=["llama.cpp-vulkan"],
        # "plugged" produces a schema-valid result; the bogus value fails
        # jsonschema validation (power_state isn't in the schema's enum),
        # which run_sweep/validate_result surfaces as a real exception.
        power_states=["plugged", "not-a-real-power-state"],
    )
    batch = SweepBatch("demo", combos)

    _run_batch(batch, **_run_batch_kwargs(environment, base_url=fake_lemonade, results_dir=tmp_path))

    assert batch.status == "done"
    assert batch.completed_count == 2
    assert batch.failed_count == 1
    statuses = {i["power_state"]: i["status"] for i in batch.items}
    assert statuses["plugged"] == "completed"
    assert statuses["not-a-real-power-state"] == "failed"
    failed_item = [i for i in batch.items if i["power_state"] == "not-a-real-power-state"][0]
    assert failed_item["error"]
    assert failed_item["run_id"] is None


def test_start_batch_runs_in_background_thread(fake_lemonade, tmp_path):
    client = LemonadeClient(fake_lemonade)
    environment, _ = discover_environment(client)

    combos = expand_combinations(
        [{"id": "Llama-3.1-8B-Instruct-GGUF", "quantization": "Q4_K_M", "context_length": 4096}],
        engines=["igpu"],
        backends=["llama.cpp-vulkan"],
        power_states=["plugged"],
    )
    batch = SweepBatch("demo", combos)

    thread = start_batch(batch, **_run_batch_kwargs(environment, base_url=fake_lemonade, results_dir=tmp_path))

    assert thread.is_alive() or batch.status == "done"  # racy but harmless either way
    thread.join(timeout=10)
    assert batch.status == "done"
    assert batch.completed_count == 1
