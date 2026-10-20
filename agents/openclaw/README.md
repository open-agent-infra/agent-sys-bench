# OpenClaw Agent

Benchmarks the OpenClaw CLI on office-work and general-purpose assistant tasks, with instrumented LLM and search proxies for full observability.

## Overview

| Dimension | Value |
|-----------|-------|
| Package | `agsbench_openclaw` |
| Image | `openclaw:v1` |
| Scenarios served | office-work |
| Hardware | CPU only; Docker-in-Docker optional (sandbox mode) |

## Build

```bash
cd /path/to/agent-sys-bench
bash scripts/setup.sh
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx
cd agents/openclaw
bash scripts/build_image.sh
```

## Serving Interface

This agent serves the following scenario(s) via fast-service RPC:

| Scenario | RPC Endpoint | Request Type | Result Type |
|----------|-------------|--------------|-------------|
| office-work | `agsbench_openclaw.run_task.run_task` | `OfficeTaskRequest` | `OfficeTaskResult` |

See `src/agsbench_openclaw/serving.py` for the full interface definition.

## Deployment

Experiments are configured and run from `benchmark/deployment/office-work-openclaw/`.
See the deployment README for each experiment setting:

- [`benchmark/deployment/office-work-openclaw/README.md`](../../benchmark/deployment/office-work-openclaw/README.md)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | *(required)* | API key for the DeepSeek LLM endpoint |
| `FIRECRAWL_API_KEY` | *(empty)* | API key for Firecrawl web search (optional) |
| `MOONSHOT_API_KEY` | *(empty)* | API key for Moonshot (optional) |
| `OPENCLAW_SANDBOX` | `0` | Toggle sandbox mode (`0`=off, `1`=on) |
| `OPENCLAW_CONCURRENCY` | `4` | Number of parallel task workers |
| `OPENCLAW_MODE` | `single` | Deployment mode: `single`, `multi`, or `multi-v2` |
| `OPENCLAW_REMOTE_HOST` | *(empty)* | Remote host for multi-node mode |
| `OPENCLAW_SANDBOX_MODE` | *(empty)* | Sandbox backend mode (e.g. `ssh` for multi-v2) |
| `OPENCLAW_MODEL` | `deepseek-instrumented/deepseek-v4-flash` | Model identifier for loadgen |
| `OPENCLAW_MODEL_ID` | `deepseek-v4-flash` | Upstream model id registered with openclaw (set by `configure-api.sh`) |
| `OPENCLAW_BM_MODE` | `one-by-one` | Loadgen arrival pattern |
| `OPENCLAW_BM_RPS` | `100` | Target requests per second |
| `OPENCLAW_REQUEST_NUM` | `1` | Number of tasks to run (paper scale: 50) |
| `OPENCLAW_MANIFEST` | `openclaw_tasks.txt` | Manifest filename inside `.cache/` |
| `OPENCLAW_DATA_DIR` | `./.cache` | Host dir with prepared task data (from `1.prepare_data.sh`), mounted at `/data/tasks` |
| `OPENCLAW_TASKS_DIR` | _(required in multi mode)_ | Local task data dir synced to the remote host |
| `OPENCLAW_LLM_PROXY_PORT` | `1801` | Host port for the LLM proxy |
| `OPENCLAW_SEARCH_PROXY_PORT` | `1802` | Host port for the search proxy |
| `OPENCLAW_PROMETHEUS_PORT` | `9090` | Host port for Prometheus |
| `OPENCLAW_GRAFANA_PORT` | `3001` | Host port for Grafana |
| `OPENCLAW_ENV_HOST` | `127.0.0.1` | Hostname for the remote env-manager (multi-v2) |
| `OPENCLAW_ENV_PORT` | `1804` | Port for the remote env-manager (multi-v2) |

### Config Files

| File | Purpose |
|------|---------|
| `config/.env.example` | Environment variable template |
| `config/openclaw-server.yml` | fast-service server config (port 1803, plus env-manager stubs) |
| `config/openclaw-client.yml` | fast-service client config for loadgen |
| `config/agsbench_openclaw.yaml` | OpenClaw model/provider config (patched via `openclaw config patch`) |
| `config/sandbox.yaml` | Sandbox agent config (Docker backend, applied via `openclaw config patch`) |
| `config/llm_proxy-server.yml` | fast-service config for the instrumented LLM proxy |
| `config/search_proxy-server.yml` | fast-service config for the instrumented search proxy |
| `config/run_task-server.yml` | fast-service config for workload driver timing |
| `config/openclaw-env.yml` | fast-service config for the remote env-manager (port 1804) |
| `monitoring/prometheus.yml` | Prometheus scrape configuration |

## Resource Controls

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCLAW_CONCURRENCY` | `4` | Maximum parallel in-flight tasks |
| `OPENCLAW_SANDBOX` | `0` | Enable Docker sandbox isolation per task |
| `OPENCLAW_SANDBOX_MODE` | *(empty)* | Sandbox backend: empty (local Docker), `ssh` (remote via SSH) |
| `MONITORING_UID` / `MONITORING_GID` | `1000` | UID/GID for Prometheus and Grafana containers |
