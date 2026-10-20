# Office-Work x OpenClaw

Benchmark the OpenClaw agent on office-work tasks from WildClawBench, with optional sandbox and search proxy support.

## Overview

| Dimension | Value |
|-----------|-------|
| Scenario | office-work |
| Agent | openclaw |
| Dataset | WildClawBench `internlm/WildClawBench.git` (pinned commit `7aa1cc9b`) |
| Evaluation | Manual / custom (no built-in evaluate script) |
| Hardware | CPU-only; no GPU required |

## Prerequisites

- Docker 24+ and Docker Compose v2
- AgentSysBench repo cloned and set up (`bash scripts/setup.sh`)
- API keys configured (`bash scripts/configure-api.sh`)
- Agent image built (`cd agents/openclaw && bash scripts/build_image.sh`)
- Loadgen image built (`cd request && bash scripts/build_image.sh`)

## Quick Start

### 1. Prepare data

```bash
cd benchmark/deployment/office-work-openclaw
bash scripts/1.prepare_data.sh
```

Clones the WildClawBench repository (pinned commit) and generates a manifest of task `.md` files in `.cache/openclaw_tasks.txt`.

### 2. Run the benchmark

```bash
bash scripts/2.run.sh
```

### 3. Observe

- Grafana: http://localhost:3001 (default `OPENCLAW_GRAFANA_PORT`)
- Prometheus: http://localhost:9090 (default `OPENCLAW_PROMETHEUS_PORT`)
- Loadgen logs: `docker logs -f openclaw-loadgen`

### 4. Cleanup

```bash
bash scripts/3.cleanup.sh
```

### 5. Analyze

From the repo root:

```bash
python benchmark/analysis/run.py office-work-openclaw
```

Report (figures, CSVs, `summary.txt`) goes to `benchmark/analysis/out/office-work-openclaw/`.
Raw data is in `benchmark/data/office-work-openclaw/`; the previous run is moved to
`benchmark/data/_archive/` on each `2.run.sh` (`AGSBENCH_KEEP_DATA=1` to append).
See [benchmark/analysis](../../analysis/README.md#raw-data-layout) for the data layout.

## Configuration

Key environment variables (set in the agent's `config/.env` or override via shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | _(required)_ | API key for the LLM endpoint |
| `OPENCLAW_CONCURRENCY` | `4` | Parallel loadgen threads |
| `OPENCLAW_REQUEST_NUM` | `1` | Number of tasks to run (paper scale: 50) |
| `OPENCLAW_BM_MODE` | `one-by-one` | Benchmark mode |
| `OPENCLAW_MODEL` | `deepseek-instrumented/deepseek-v4-flash` | Model name |
| `OPENCLAW_MODEL_ID` | `deepseek-v4-flash` | Upstream model id registered with openclaw (set by `configure-api.sh`) |
| `OPENCLAW_SANDBOX` | `0` | Enable sandbox mode (`0`=off, `1`=on) |
| `OPENCLAW_SANDBOX_MODE` | _(empty)_ | Sandbox mode (`ssh` for multi-v2) |
| `OPENCLAW_DATA_DIR` | `./.cache` | Host dir with prepared task data (from `1.prepare_data.sh`), mounted at `/data/tasks` |
| `OPENCLAW_LLM_PROXY_PORT` | `1801` | Host port for the LLM proxy |
| `OPENCLAW_GRAFANA_PORT` | `3001` | Host port for Grafana |
| `OPENCLAW_PROMETHEUS_PORT` | `9090` | Host port for Prometheus |

## Architecture

```
┌────────────────┐     ┌────────────────┐     ┌───────────────────┐
│openclaw-loadgen│────▶│openclaw-service│────▶│ openclaw-llm-proxy│────▶ LLM API
│(agsbench-req)  │ HTTP│ (openclaw:v1)  │ HTTP│ (openclaw:v1)     │
└────────────────┘:1803└────────┬───────┘     │    :1801          │
                                │             └───────────────────┘
                                │
                                ├────▶ openclaw-search-proxy :1802 (profile: search)
                                │
                                └────▶ docker.sock (sandbox containers)

┌────────────────┐     ┌──────────────────┐     ┌──────────┐
│openclaw-grafana│────▶│openclaw-prometheu│────▶│ cadvisor │
│     :3001      │     │      :9090       │     │          │
└────────────────┘     └──────────────────┘     └──────────┘
```

- **openclaw-llm-proxy** -- Proxies LLM requests from the agent to the configured API endpoint. Exposed on port 1801.
- **openclaw-search-proxy** -- Optional search proxy (activate with `--profile search`). Exposed on port 1802.
- **openclaw-service** -- The OpenClaw agent. Receives tasks on port 1803. Can operate in sandbox mode (spawning Docker containers via the host socket) or non-sandbox mode.
- **openclaw-loadgen** -- Sends task paths from the WildClawBench manifest to the agent service.
- **cadvisor / prometheus / grafana** -- Resource monitoring stack with auto-provisioned dashboards.
