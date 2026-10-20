# SWE x Mini-SWE-Agent

Benchmark the Mini-SWE-Agent on SWE-bench software-engineering tasks with full resource monitoring.

## Overview

| Dimension | Value |
|-----------|-------|
| Scenario | swe |
| Agent | mini-swe-agent |
| Dataset | `princeton-nlp/SWE-bench` (pinned revision `e48e2bd1`) |
| Evaluation | SWE-bench harness (`swebench.harness.run_evaluation`) |
| Hardware | CPU-only; 4 CPU-sets + 4 GB RAM per sandbox (configurable) |

## Prerequisites

- Docker 24+ and Docker Compose v2
- AgentSysBench repo cloned and set up (`bash scripts/setup.sh`)
- API keys configured (`bash scripts/configure-api.sh`)
- Agent image built (`cd agents/mini-swe-agent && bash scripts/build_image.sh`)
- Loadgen image built (`cd request && bash scripts/build_image.sh`)

## Quick Start

### 1. Prepare data

```bash
cd benchmark/deployment/swe-mini-swe-agent
bash scripts/1.prepare_data.sh
```

Downloads the SWE-bench dataset from HuggingFace, writes per-instance JSON files and a manifest to `.cache/`.

### 2. Run the benchmark

```bash
bash scripts/2.run.sh
```

### 3. Observe

- Grafana: http://localhost:3000 (default `MINI_SWEAGENT_GRAFANA_PORT`)
- Prometheus: http://localhost:9090 (default `MINI_SWEAGENT_PROMETHEUS_PORT`)
- Loadgen logs: `docker logs -f mini-sweagent-loadgen`

### 4. Cleanup

```bash
bash scripts/3.cleanup.sh
```

### 5. Analyze

From the repo root:

```bash
python benchmark/analysis/run.py swe-mini-swe-agent
```

Report (figures, CSVs, `summary.txt`) goes to `benchmark/analysis/out/swe-mini-swe-agent/`.
Raw data is in `benchmark/data/swe-mini-swe-agent/`; the previous run is moved to
`benchmark/data/_archive/` on each `2.run.sh` (`AGSBENCH_KEEP_DATA=1` to append).
See [benchmark/analysis](../../analysis/README.md#raw-data-layout) for the data layout.

### 5. Evaluate

```bash
bash scripts/4.evaluate.sh
```

Runs the upstream SWE-bench harness against `trajectories/preds.json`. Produces a summary report and per-instance logs.

## Configuration

Key environment variables (set in the agent's `config/.env` or override via shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `MINI_SWE_CPUSETS` | `0-1,2-3,4-5,6-7` | CPU-set pool for sandbox containers |
| `MINI_SWE_MEMORY` | `4g` | Memory limit per sandbox container |
| `MINI_SWEAGENT_CONCURRENCY` | `4` | Parallel loadgen threads |
| `MINI_SWEAGENT_REQUEST_NUM` | `1` | Number of tasks to run (paper scale: 203) |
| `MINI_SWEAGENT_BM_MODE` | `one-by-one` | Benchmark mode (`one-by-one` or `rps`) |
| `MINI_SWEAGENT_BM_RPS` | `100` | Requests per second (when mode is `rps`) |
| `MINI_SWEAGENT_GRAFANA_PORT` | `3000` | Host port for Grafana |
| `MINI_SWEAGENT_PROMETHEUS_PORT` | `9090` | Host port for Prometheus |
| `MINI_SWE_REMOTE_ENV` | _(empty)_ | Set to `1` for multi-node mode |
| `MINI_SWE_ENV_HOST` | `127.0.0.1` | Remote env host (multi-node only) |
| `MINI_SWE_ENV_PORT` | `1802` | Remote env port (multi-node only) |

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐
│ mini-sweagent-loadgen│────▶│ mini-sweagent-service │
│(agsbench-request:v1)│ HTTP│ (mini-swe-agent:v1)   │
└─────────────────────┘:1801└──────────┬───────────┘
                                       │ docker.sock
                                       ▼
                              SWE-bench sandbox containers

┌─────────────────────┐     ┌──────────────────────┐     ┌────────────┐
│ mini-sweagent-grafana│────▶│mini-sweagent-prometheu│────▶│  cadvisor  │
│        :3000         │     │         :9090         │     │            │
└─────────────────────┘     └──────────────────────┘     └────────────┘
```

- **mini-sweagent-service** -- The agent, exposed on port 1801. Spawns per-task Docker sandbox containers via the host Docker socket.
- **mini-sweagent-loadgen** -- Sends tasks from the prepared manifest to the agent service one-by-one (or at a target RPS).
- **cadvisor / prometheus / grafana** -- Resource monitoring stack. Prometheus scrapes cAdvisor every 5 s; Grafana dashboards are auto-provisioned.
