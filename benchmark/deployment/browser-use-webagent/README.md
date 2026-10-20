# Browser-Use x WebAgent

Benchmark the WebAgent on browser-automation tasks using BrowserGym/MiniWoB++ with a live WebArena shopping-admin site.

## Overview

| Dimension | Value |
|-----------|-------|
| Scenario | browser-use |
| Agent | webagent |
| Dataset | BrowserGym MiniWoB++ task IDs (builtin, no external download) |
| Evaluation | Reward-based: `reward > 0` counts as resolved |
| Hardware | CPU-only; 2 GB shared memory for Chromium (`shm_size: 2gb`) |

## Prerequisites

- Docker 24+ and Docker Compose v2
- AgentSysBench repo cloned and set up (`bash scripts/setup.sh`)
- API keys configured (`bash scripts/configure-api.sh`)
- Agent image built (`cd agents/webagent && bash scripts/build_image.sh`)
- Loadgen image built (`cd request && bash scripts/build_image.sh`)

## Quick Start

### 1. Prepare data

```bash
cd benchmark/deployment/browser-use-webagent
bash scripts/1.prepare_data.sh
```

Generates a manifest of MiniWoB++ task IDs in `.cache/browsergym_data.txt`. No external download required.

### 2. Run the benchmark

```bash
bash scripts/2.run.sh
```

### 3. Observe

- Grafana: http://localhost:3000 (default `WEBAGENT_GRAFANA_PORT`)
- Prometheus: http://localhost:9090 (default `WEBAGENT_PROMETHEUS_PORT`)
- Loadgen logs: `docker logs -f webagent-benchmarking-browsergym-loadgen-1`

### 4. Cleanup

```bash
bash scripts/3.cleanup.sh
```

### 5. Analyze

From the repo root:

```bash
python benchmark/analysis/run.py browser-use-webagent
```

Report (figures, CSVs, `summary.txt`) goes to `benchmark/analysis/out/browser-use-webagent/`.
Raw data is in `benchmark/data/browser-use-webagent/`; the previous run is moved to
`benchmark/data/_archive/` on each `2.run.sh` (`AGSBENCH_KEEP_DATA=1` to append).
See [benchmark/analysis](../../analysis/README.md#raw-data-layout) for the data layout.

### 5. Evaluate

```bash
bash scripts/4.evaluate.sh
```

Aggregates per-task rewards from `trajectories/preds.json` into a resolve-rate report.

## Configuration

Key environment variables (set in the agent's `config/.env` or override via shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `WEBAGENT_CONCURRENCY` | `1` | Parallel loadgen threads |
| `WEBAGENT_REQUEST_NUM` | `1` | Number of tasks to run (paper scale: 83) |
| `WEBAGENT_BM_MODE` | `one-by-one` | Benchmark mode |
| `WEBAGENT_MAX_STEPS` | `30` | Maximum agent steps per task |
| `WEBAGENT_USE_SCREENSHOT` | `0` | Enable screenshot-based observation |
| `WEBAGENT_CHECK_URLS` | _(empty)_ | URLs to health-check before starting the agent |
| `WEBAGENT_MANIFEST` | `browsergym_data.txt` | Manifest file under `.cache/` |
| `COMPOSE_PROFILES` | _(empty)_ | Set to `webarena` to also start the WebArena shopping-admin site |
| `WEBAGENT_GRAFANA_PORT` | `3000` | Host port for Grafana |
| `WEBAGENT_PROMETHEUS_PORT` | `9090` | Host port for Prometheus |

## Architecture

```
┌────────────────────┐     ┌────────────────────┐     ┌────────────────┐
│ browsergym-loadgen │────▶│ browsergym-service  │────▶│ shopping-admin │
│ (agsbench-request) │ HTTP│ (webagent:v1)       │ HTTP│ (webarena)     │
└────────────────────┘:1902└────────────────────┘     └────────────────┘
                              │ headless Chromium          :80
                              │ + tinyproxy

┌──────────┐     ┌────────────┐     ┌──────────┐
│  grafana │────▶│ prometheus │────▶│ cadvisor │
│   :3000  │     │   :9090    │     │          │
└──────────┘     └────────────┘     └──────────┘
```

- **shopping-admin** -- WebArena verified shopping-admin site (Magento). Provides a real web UI for browser-based tasks.
- **browsergym-service** -- The WebAgent, running headless Chromium with BrowserGym. Exposed on port 1902. Uses tinyproxy for network isolation. Requires `SYS_ADMIN` capability and `seccomp=unconfined` for Chromium sandboxing.
- **browsergym-loadgen** -- Sends task IDs from the manifest to the agent service.
- **cadvisor / prometheus / grafana** -- Resource monitoring stack with auto-provisioned dashboards.
