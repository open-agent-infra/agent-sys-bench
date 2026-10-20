# agsbench-request

The workload package for AgentSysBench — unified data preparation and load generation for all scenarios.

## Overview

This package provides two unified entry points that work across all 10 scenarios:

- **`prepare_data`** — Downloads datasets, generates manifests with task configurations
- **`loadgen`** — Dispatches benchmark requests to agent services via HTTP

The loadgen is a **pure HTTP client** — it imports no agent code. It constructs
scenario-defined `Request` models from a manifest, sends them via `ScenarioClient`
to the agent's fast-service endpoint, and deserializes responses into
scenario-defined `Result` models.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    agsbench-request                   │
│  ┌──────────────┐  ┌──────────────┐                │
│  │ prepare_data  │  │   loadgen    │                │
│  │ (data prep)  │  │ (HTTP client) │                │
│  └──────┬───────┘  └──────┬───────┘                │
│         │                   │                          │
│  ┌──────▼───────┐  ┌──────▼───────┐                │
│  │  scenario     │  │  _common     │                │
│  │  modules      │  │  (shared     │                │
│  │  (per-scenario│  │  run logic)  │                │
│  │   parse +    │  └──────────────┘                │
│  │   success)   │                                    │
│  └──────────────┘                                    │
└─────────────────────────┬───────────────────────────┘
                            │ HTTP
                            ▼
                   ┌─────────────────┐
                   │  Agent Service  │
                   │ (fast-service)  │
                   └─────────────────┘
```

## Build

```bash
cd request
bash scripts/build_image.sh
# Produces agsbench-request:v1 (~715 MB)
```

## Usage

### Prepare data

```bash
docker run --rm -v ./cache:/output \
  --entrypoint python agsbench-request:v1 \
  -m agsbench_request.prepare_data \
  --scenario swe --output_dir /output
```

Each scenario's `prepare_data.py` defines `extra_args(parser)` and `run(args)`:
- Downloads the dataset (pinned revision)
- Saves data files
- Generates a manifest file

### Run benchmark

```bash
docker run --rm --network <compose-net> agsbench-request:v1 \
  --scenario swe \
  --endpoint http://agent-service:1801 \
  --bm_file_path /data/manifest.txt \
  --bm_request_num 1 \
  --bm_mode one-by-one
```

Each scenario's `loadgen.py` defines:
- `line_to_request(line, args)` — parse manifest line → Request model
- `is_success(result)` — determine success from Result model
- `extra_args(parser)` — scenario-specific CLI args (optional)

## Available Scenarios

| Scenario | Dataset | Agent(s) |
|----------|---------|----------|
| qa | wiki_qa (pinned HF rev) | rag |
| multimodal | tiny-imagenet (HF) | hugginggpt |
| ai-search | hle/gaia/frames (pinned HF rev) | deep-research |
| swe | SWE-bench (pinned HF rev) | mini-swe-agent |
| terminal-use | SWE-bench (pinned HF rev) | codex |
| browser-use | MiniWoB++ (builtin) | webagent |
| computer-use | OSWorld (pinned commit) | gui-agent |
| tool-rich-assistant | PinchBench (pinned commit) | claude |
| office-work | WildClawBench (pinned commit) | openclaw |
| auto-research | Builtin tasks | pi-autoresearch |

## Wire Protocol

The loadgen and agent communicate through the `scenario-protocol` SDK package
(`sdk/scenario-protocol/`). Each scenario defines:
- `Request` / `Result` pydantic models
- `RPC_NAME` — the agent's fast-service endpoint path
- `PARAM_NAME` — the function parameter name for body wrapping

The loadgen never imports agent code — it only depends on the SDK's wire
protocol definitions.

## Extending: Add a New Scenario

1. Add wire protocol in `sdk/scenario-protocol/src/scenario_protocol/scenarios/<name>.py`
2. Add scenario module in `request/src/agsbench_request/<name>/`:
   - `loadgen.py` — `line_to_request`, `is_success`, `extra_args`
   - `prepare_data.py` — `extra_args`, `run` (optional)
   - `serving.py` — serving contract (optional)
3. Add deployment directories: `benchmark/deployment/<scenario>-<agent>/`

See `request/src/agsbench_request/swe/` for the pattern.
