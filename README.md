# LemonMatrix

A benchmarking control plane for [Lemonade](https://lemonade-server.ai) instances.
One profile = one Lemonade instance, auto-discovered and reused as the environment
fingerprint for every run on that machine. See [IDEA.md](IDEA.md) for the full
data model and [schema/result.schema.json](schema/result.schema.json) for the
canonical result schema.

## Install

```bash
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
range for the multi-machine fleet case described in IDEA.md. Lemonade has no
announce/broadcast mechanism, so this is a bounded port scan, not magic
discovery -- capped at 1024 hosts per scan.

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
