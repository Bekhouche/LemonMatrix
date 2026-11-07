# AGENTS.md

This file provides guidance to AI agents working with this repository.

## Project Overview

LemonMatrix is a benchmarking control plane for [Lemonade](https://lemonade-server.ai/) instances. It manages multiple Lemonade profiles (local, LAN, cloud), runs benchmark sweeps across model/backend/engine/power-state combinations, validates results against a shared schema, and exposes results through a CLI and a local Flask dashboard.

## Core design principle — Lemonade is the only source of truth

**LemonMatrix must never reach outside Lemonade's own API to learn facts about hardware.**

Every measurement, device description, and resource reading must come from one of Lemonade's own endpoints:

| Data needed | Lemonade endpoint |
|---|---|
| Hardware description (CPU, GPU, NPU, memory) | `GET /api/v1/system-info` |
| Runtime resource usage (VRAM, RAM, watts) | `GET /api/v1/system-stats` |
| Server health and version | `GET /api/v1/health` |
| Inference timing and token counts | `GET /v1/stats` |
| Loaded models (including routers) | `GET /api/v1/models` |
| ONNX encoder classification | `POST /v1/classify` |

Do **not** call `nvidia-smi`, `rocm-smi`, `psutil`, or any OS-level sensor from LemonMatrix code. A profile can point at a remote machine — local hardware reads are meaningless and wrong.

When Lemonade does not yet expose a field (e.g. thermal temperature, competing-process count), the correct response is to forward-guess the likely future field name and leave the metric absent rather than approximating it from another source. See `_resource_samples` in `bench.py` for the established pattern.

## Architecture

```
Profile (one Lemonade instance)
  └── environment dict  ← fixed facts from /api/v1/system-info + /api/v1/health
  └── base_url + api_key

Run / Sweep
  └── SweepConfig  ← matrix axes: model, quant, engine, backend, power_state
  └── run_sweep()  ← warmup, measured trials, aggregation
        returns (result_dict, raw_trials_list)
  └── validate_result()  ← jsonschema against schema/result.schema.json

Results
  └── results/<profile>/<run_id>.json   ← schema-conformant result
  └── results/<profile>/trials/<run_id>.json  ← raw per-trial measurements (sidecar)
  └── results/<profile>/failures/<id>.json   ← runs that raised before producing a result

Sweep batch (dashboard or CLI)
  └── SweepBatch  ← Cartesian product of axes, run sequentially
  └── SweepStore  ← SQLite persistence so restarts don't lose batch history

CLI
  └── lemonmatrix profile add/detect/list/show/refresh/delete/debug
  └── lemonmatrix profile install-backend/search-models/pull-variants/pull-model/downloads/downloads-control
  └── lemonmatrix run        ← single configuration
  └── lemonmatrix sweep      ← Cartesian matrix sweep
  └── lemonmatrix export     ← CSV export
  └── lemonmatrix trials     ← print raw per-trial measurements
  └── lemonmatrix dashboard  ← start the Flask UI

Dashboard (Flask)
  └── /                        ← sortable/filterable leaderboard
  └── /profiles/...            ← profile management, run form, sweep form
  └── /export/results.csv      ← CSV download
  └── /results/<profile>/<id>  ← single result detail
```

## Run types: model vs router

LemonMatrix supports two first-class run types, both using the same `chat/completions` + `/v1/stats` pipeline:

### Model run (`run_type = "model"`, the default)
Benchmarks a single LLM loaded by Lemonade.  `model.name` is the model's registered id; `config.backend` and `config.compute_engine` describe the physical execution path.

### Router run (`run_type = "router"`)
Benchmarks a Lemonade **collection.router** — a routing policy that receives the request, classifies it, selects a downstream model, and forwards the request there.  From the client's perspective the call is identical (`POST /api/v1/chat/completions` with the router model as `model`); LemonMatrix passes `route_trace: true` so Lemonade appends `x_lemonade_route: { route_to, matched_rule, default_used, outputs, trace[] }` to the response body.

What changes for a router run:
- `model.class` = `"router"`, `config.compute_engine` = `"router"`, `config.backend` = `"collection.router"`.
- `run_type: "router"` is set at the top level of the result JSON.
- `config.router_default_model` is populated from the first trial where `default_used=True` (the policy's fallback candidate).
- Per-trial routing decisions (`route_to`, `matched_rule`, `default_used`) are stored in the **trials sidecar** (`results/<profile>/trials/<run_id>.json`) — not in the schema-conformant result.
- Engine/backend compatibility validation is skipped; the router manages backend selection internally.

Detecting routers:
- `capabilities.is_router_model(model_dict)` — returns True when `recipe` starts with `"collection"` or the model id starts with `"collection."`.
- `capabilities.available_routers(models_list)` — filters a `/api/v1/models` response to router entries only.

CLI usage:
```bash
lemonmatrix run \
  --profile my-strix \
  --model my-collection-router \
  --model-class router \
  --engine router \
  --backend collection.router \
  --power-state plugged \
  --run-type router \
  --quant none \
  --context-length 4096
```

## Validity model

A run's `validity.valid` flag is `False` if any hard gate fails:

- `exclusive_run` is `False` (user declared a competing workload was present)
- The backend could not be resolved to a Lemonade recipe (Lemonade loaded its default instead)
- `engine_backend_compatible()` returns `False` (e.g. cpu engine + rocm backend)

`thermal_ok` and power-state claims are currently user-asserted. They are recorded but not verified — LemonMatrix will verify them automatically once Lemonade exposes the corresponding fields in its API.

## Key invariants for agents

1. **Never add OS-level sensing.** If hardware data is needed and Lemonade doesn't expose it yet, add a forward-compatible key lookup to the existing field-guessing code; do not add a fallback to a local tool.
2. **`run_sweep` returns a tuple `(result, raw_trials)`.** All callers must unpack both. Raw trials are saved to the sidecar directory; they are never embedded in the schema-conformant result JSON.
3. **Schema is strict.** `result.schema.json` uses `additionalProperties: false` at every level. Adding a new field to a result requires a matching schema change first. Both copies of the schema (`schema/result.schema.json` and `src/lemonmatrix/schema/result.schema.json`) must be kept in sync.
4. **Tests use a fake Lemonade server** (`tests/conftest.py`). All new functionality that touches the Lemonade API must be exercised through the fake server, not mocked at the client level.
5. **There are two run types.** `run_type = "model"` (default) benchmarks a direct model. `run_type = "router"` benchmarks a `collection.router`; `SweepConfig.run_type` controls which path `_run_one` takes. Engine/backend validation is skipped for router runs.
6. **Route trace data belongs in the trials sidecar, not the result JSON.** Per-trial `route_to`, `matched_rule`, and `default_used` fields are stored in `results/<profile>/trials/<run_id>.json`. Only `router_default_model` (derived from those trials) appears in the schema-conformant result.
