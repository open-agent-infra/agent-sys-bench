# Terminal-Use x Codex

Benchmark the Codex CLI agent on [TerminalBench](https://github.com/laude-institute/terminal-bench) tasks, with a LiteLLM translation proxy for Responses API compatibility.

## Overview

| Dimension | Value |
|-----------|-------|
| Scenario | terminal-use |
| Agent | codex |
| Dataset | `laude-institute/terminal-bench` `original-tasks/` (pinned commit `d28711d0`, 241 tasks) |
| Evaluation | Each task's own `run-tests.sh`, run inside the task container right after the agent finishes |
| Hardware | CPU-only; one CPU-set per sandbox (configurable); the host must be able to build the task Dockerfiles (network access) |

## Prerequisites

- Docker 24+ and Docker Compose v2
- AgentSysBench repo cloned and set up (`bash scripts/setup.sh`)
- API keys configured (`bash scripts/configure-api.sh`)
- Agent image built (`cd agents/codex && bash scripts/build_image.sh`)
- Loadgen image built (`cd request && bash scripts/build_image.sh`)
- LiteLLM proxy image built (see `agents/codex/README.md`)

## Quick Start

### 1. Prepare data

```bash
cd benchmark/deployment/terminal-use-codex
bash scripts/1.prepare_data.sh
```

Clones the pinned TerminalBench commit into `.cache/terminal-bench/` and writes the manifest
`.cache/codex_data.txt` (one JSON line per task; `hello-world` first, the rest sorted).
The task dirs are mounted read-only into `codex-service`, which builds and starts each task's
own `docker-compose.yaml` per request.

Select tasks with `--tasks a,b,c` or `--task_filter <regex>`:

```bash
bash scripts/1.prepare_data.sh --task_filter '^(hello-world|fix-git)$'
```

An existing manifest is kept as is; pass `--force` to regenerate it (e.g. after changing the
task selection, or if `.cache/` still holds a manifest from an older SWE-bench-based prepare).

The default `CODEX_REQUEST_NUM=1` runs only `hello-world`. For the paper-scale run, set
`CODEX_REQUEST_NUM=220` (the loadgen stops at the end of the manifest).

### 2. Run the benchmark

```bash
bash scripts/2.run.sh
```

### 3. Observe

- Grafana: http://localhost:3000 (default `CODEX_GRAFANA_PORT`)
- Prometheus: http://localhost:9090 (default `CODEX_PROMETHEUS_PORT`)
- Loadgen logs: `docker logs -f codex-loadgen`

### 4. Cleanup

```bash
bash scripts/3.cleanup.sh
```

### 5. Analyze

From the repo root:

```bash
python benchmark/analysis/run.py terminal-use-codex
```

Report (figures, CSVs, `summary.txt`) goes to `benchmark/analysis/out/terminal-use-codex/`.
Raw data is in `benchmark/data/terminal-use-codex/`; the previous run is moved to
`benchmark/data/_archive/` on each `2.run.sh` (`AGSBENCH_KEEP_DATA=1` to append).
See [benchmark/analysis](../../analysis/README.md#raw-data-layout) for the data layout.

### 6. Evaluate

```bash
bash scripts/4.evaluate.sh
```

Tests already run inline during the benchmark; this summarises
`benchmark/data/terminal-use-codex/trajectories/results.json` (per-task tests passed/total,
exec/test time, overall accuracy).

## Configuration

Key environment variables (set in the agent's `config/.env` or override via shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_CPUSETS` | `0-1,2-3,4-5,6-7` | CPU-set pool for sandbox containers |
| `CODEX_CONCURRENCY` | `4` | Parallel loadgen threads |
| `CODEX_REQUEST_NUM` | `1` | Number of tasks to run (paper scale: 220) |
| `CODEX_BM_MODE` | `one-by-one` | Benchmark mode |
| `CODEX_STEP_TIMEOUT` | `0` | Agent timeout in seconds; `0` uses the task's `max_agent_timeout_sec` |
| `CODEX_TEST_TIMEOUT` | `0` | Test timeout in seconds; `0` uses the task's `max_test_timeout_sec` |
| `CODEX_PROXY_URL` | `http://host.docker.internal:4001/v1` | URL sandbox containers use to reach LiteLLM |
| `CODEX_LITELLM_PORT` | `4001` | Host port for the codex-proxy (LiteLLM frontend) |
| `CODEX_LITELLM_MASTER_KEY` | `sk-agsbench-codex-local` | Master key for the internal LiteLLM proxy (required by LiteLLM; not exposed outside the compose network) |
| `CODEX_GRAFANA_PORT` | `3000` | Host port for Grafana |
| `CODEX_PROMETHEUS_PORT` | `9090` | Host port for Prometheus |

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ codex-loadgen│────▶│ codex-service│     │  codex-proxy │────▶│ litellm-proxy│
│(agsbench-req)│ HTTP│ (codex:v1)   │     │  :4001→5000  │     │   :4000      │
└──────────────┘     └──────┬───────┘     └──────────────┘     └──────────────┘
                            │ docker.sock        ▲
                            ▼                    │
                 TerminalBench task ─────────────┘
                 containers (compose) (Responses API via host.docker.internal:4001)

┌──────────┐     ┌────────────┐     ┌──────────┐
│  grafana │────▶│ prometheus │────▶│ cadvisor │
│   :3000  │     │   :9090    │     │          │
└──────────┘     └────────────┘     └──────────┘
```

- **codex-service** -- Orchestrator that receives tasks on port 1903 (compose network only). Per task it runs `docker compose up --build` on the task dir, copies the codex binary in, runs `codex exec`, then `run-tests.sh`, and tears the project down (`--rmi all`).
- **litellm-proxy** -- Translates OpenAI Responses API calls to chat-completions for DashScope/DeepSeek backends.
- **codex-proxy** -- FastAPI middleware that scrubs unsupported tool types from Responses API requests before forwarding to LiteLLM.
- **codex-loadgen** -- Sends tasks from the manifest to codex-service.
- **cadvisor / prometheus / grafana** -- Resource monitoring stack with auto-provisioned dashboards.
