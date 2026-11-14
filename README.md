# LemonMatrix

A benchmarking control plane for [Lemonade](https://lemonade-server.ai) instances.
One profile = one Lemonade instance, auto-discovered and reused as the environment
fingerprint for every run on that machine. See
[schema/result.schema.json](https://github.com/Bekhouche/LemonMatrix/blob/main/schema/result.schema.json) for the canonical result
schema.

**Live public leaderboard: [lemonmatrix.github.io](https://lemonmatrix.github.io)**
-- results and the site's own generator live in the sibling
[LemonMatrix/lemonmatrix.github.io](https://github.com/LemonMatrix/lemonmatrix.github.io)
repo; this repo is code (CLI, library, tests, schema) only. Community
submissions go through a PR adding a result JSON under `results/<profile>/`
there, gated by CI running `lemonmatrix validate-submission` against the
schemas in this repo.

## Install

```bash
pip install lemonmatrix
```

Python 3.10 or newer. The [step-by-step guide](https://lemonmatrix.github.io/guide.html)
walks from a running Lemonade to a result on the leaderboard.

To work on LemonMatrix itself:

```bash
git clone https://github.com/Bekhouche/LemonMatrix.git
cd LemonMatrix
pip install -e ".[dev]"
```

## Quickstart

```bash
# Find a Lemonade instance running on this machine and save it as a profile.
lemonmatrix profile detect --save my-machine

# Or add one you already know the address of.
lemonmatrix profile add my-machine --host 192.168.1.42 --port 13305 --token $LEMONADE_API_KEY

lemonmatrix profile list
lemonmatrix profile show my-machine

# Run a sweep and write a schema-conformant result JSON under results/.
# OS isn't a flag -- it's derived from the profile's own discovered
# environment, since a different OS is a different profile, not a per-run
# setting.
lemonmatrix run --profile my-machine \
  --model Llama-3.1-8B-Instruct-GGUF --model-class dense --quant Q4_K_M \
  --context-length 4096 --engine igpu --backend llama.cpp-vulkan \
  --power-state plugged

# Browse profiles and results, or launch new sweeps, from a local dashboard.
lemonmatrix dashboard
```

`lemonmatrix profile detect --subnet 192.168.1.0/24` extends the scan to a LAN
range for the multi-machine fleet case. Lemonade has no
announce/broadcast mechanism, so this is a bounded port scan, not magic
discovery -- capped at 1024 hosts per scan.

## What a model run measures

Each trial sends the same prompt with `cache_prompt: false`, so llama.cpp
processes the whole prompt every time instead of serving it from its prompt
cache, while the model stays loaded (warmups are real warmups). TTFT and
prefill therefore time the prompt, not a cache hit. Each trial records the
server's `cache_tokens`; a run where any measured trial still reused cached
prompt tokens is marked invalid. Results from LemonMatrix 0.1.0 and earlier
did not do this, so their TTFT and prefill are cache-assisted (decode is not
affected).

## Router cost runs

A `collection.router` that mixes local and cloud candidates answers one
question: how much of the cloud bill does local hardware take over? Give the
router a mixed prompt set and LemonMatrix records where each request went and
prices it:

```bash
lemonmatrix run --profile my-machine --run-type router \
  --model user.My-Router --context-length 4096 --power-state plugged \
  --prompt-set router-mix --trials 10
```

The result's `metrics.routing` block holds the share of requests per
candidate, the share that stayed local, the cost of the run, and the cost of
sending every request to the most expensive candidate instead. Prices are USD
per million tokens. They come from Lemonade's own `estimated_cost` when the
candidate carries cost metadata (cloud models get it from their provider), and
non-cloud candidates count as free. `--cost MODEL=IN,OUT` sets or overrides a
price, `--cost-baseline MODEL` picks the comparison model, and
`--prompts-file` takes your own prompts (one per line, or JSONL).

After the trials, each prompt the router sent elsewhere goes once to the
baseline model, and the baseline is priced on the tokens that model actually
wrote. This matters for reasoning models, which can write ten times the tokens
of a small local model for the same prompt. The extra requests run outside the
measured window; `--estimate-baseline` skips them and reuses the routed
model's token counts, and the result's `baseline_method` says which was used.

## Ryzen AI NPU and hybrid runs

Lemonade's `ryzenai-llm` recipe has one backend, `ryzenai-llm-npu`, and the
model's name says where it runs: `-Hybrid` (NPU and iGPU), `-NPU` or `-CPU`.
FastFlowLM (`flm-npu`) runs every model on the NPU. LemonMatrix takes the
engine from the model, so a run whose `--engine` contradicts it is refused,
and `lemonmatrix sweep` without `--engine` runs each of these models on its
own engine:

```bash
lemonmatrix sweep --profile my-laptop --backend ryzenai-llm-npu \
  --model Llama-3.2-1B-Instruct-Hybrid --model Llama-3.2-1B-Instruct-NPU \
  --model Llama-3.2-1B-Instruct-CPU --power-state plugged
```

Lemonade reports the device `npu` for every `ryzenai-llm` model, whatever
its deployment, so for `-Hybrid` and `-CPU` models the device check cannot
confirm the engine; the result is still ranked and its notes say so.

## Submitting results

`lemonmatrix submit results/my-machine/<run_id>.json` opens a pull request
that adds the file to the public leaderboard
([LemonMatrix/lemonmatrix.github.io](https://github.com/LemonMatrix/lemonmatrix.github.io)).
It validates the file against the same schemas the leaderboard's CI uses, forks
the repo (or reuses your fork), keeps the file's `results/<profile>/...` path,
pushes and opens the PR with the GitHub CLI. `--dry-run` shows the PR without
touching GitHub.

## Importing other tools' results

`lemonmatrix import FILE...` reads results from
[lemonade_lab](https://github.com/skywalk1411/lemonade_lab) (as published in
amdaibenchmarks) and [LemonMetrics](https://github.com/lemonmetrics/lemonmetrics)
`report.json` files, which carry Ryzen AI NPU/hybrid rows and energy
estimates. It also reads a saved copy of Lemonade's own bench site
(`curl -s https://bench.lemonade-server.ai/api/benchmarks > bench-site.json`),
which becomes one document per machine because the list mixes machines.
Each file is normalized into `results/imported/<tool>/` and listed
with `lemonmatrix imported-results`. These numbers did not go through
LemonMatrix's validity checks, so they are marked `validated: false` and never
appear on the leaderboard. Host names, commands, URLs and raw power samples are
not copied. When several backends report the same number for one model, the
import warns that the backend switch may not have taken effect.

## Dashboard

`lemonmatrix dashboard` starts a local Flask app (default
`http://127.0.0.1:5050`) with:

- A sortable, filterable leaderboard over every result under `results/`.
- Profile management: detect, add, and inspect environment fingerprints.
- A form to launch a new sweep against a saved profile (runs synchronously,
  same as the CLI's `run` command).

It's a single-user local tool with no job queue -- submitting a run blocks the
page until the sweep finishes.

## Development

```bash
pytest tests/
```

Tests run against a fake Lemonade server (`tests/conftest.py`) modeled on the
documented `/api/v1/*` response shapes, so they don't require real AMD hardware.
