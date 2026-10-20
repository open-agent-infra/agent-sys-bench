# MiniSWE Agent

Lightweight SWE-bench agent that solves GitHub issues by iteratively reading, editing, and testing code inside Docker containers.

## Overview

| Dimension | Value |
|-----------|-------|
| Package | `agsbench_mini_swe_agent` |
| Image | `mini-swe-agent:v1` |
| Scenarios served | swe |
| Hardware | CPU only; Docker-in-Docker required (spawns per-task SWE-bench containers) |

## Build

```bash
cd /path/to/agent-sys-bench
bash scripts/setup.sh
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx
cd agents/mini-swe-agent
bash scripts/build_image.sh
```

## Serving Interface

This agent serves the following scenario(s) via fast-service RPC:

| Scenario | RPC Endpoint | Request Type | Result Type |
|----------|-------------|--------------|-------------|
| swe | `agsbench_mini_swe_agent.request.solve_instance` | `SWETaskRequest` | `SWETaskResult` |

See `src/agsbench_mini_swe_agent/serving.py` for the full interface definition.

## Deployment

Experiments are configured and run from `benchmark/deployment/swe-mini-swe-agent/`.
See the deployment README for each experiment setting:

- [`benchmark/deployment/swe-mini-swe-agent/README.md`](../../benchmark/deployment/swe-mini-swe-agent/README.md)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | API key for the LLM endpoint |
| `OPENAI_API_BASE` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Base URL for the OpenAI-compatible API |
| `MINI_SWE_MODEL` | `openai/deepseek-v4-pro` | LiteLLM model string (prefix with provider) |
| `MSWEA_COST_TRACKING` | `ignore_errors` | Disable cost calculation for models not in LiteLLM's price registry |
| `MINI_SWE_CPUSETS` | `0-1,2-3,4-5,6-7` | CPU core pool for sandbox containers |
| `MINI_SWE_MEMORY` | `4g` | Memory limit per SWE-bench sandbox |
| `MINI_SWEAGENT_CONCURRENCY` | `4` | Number of parallel loadgen threads |
| `MINI_SWEAGENT_BM_MODE` | `poisson` | Loadgen arrival pattern |
| `MINI_SWEAGENT_BM_RPS` | `100` | Target requests per second |
| `MINI_SWEAGENT_REQUEST_NUM` | `1` | Number of tasks to run (paper scale: 203) |
| `MINI_SWEAGENT_MANIFEST` | `mini_swe_agent_data.txt` | Manifest filename inside `.cache/` |
| `MINI_SWEAGENT_PROMETHEUS_PORT` | `9090` | Host port for Prometheus |
| `MINI_SWEAGENT_GRAFANA_PORT` | `3000` | Host port for Grafana |
| `MINI_SWE_REMOTE_ENV` | *(empty)* | Set to `1` to enable multi-node mode |
| `MINI_SWE_ENV_HOST` | `127.0.0.1` | Hostname of the remote env service |
| `MINI_SWE_ENV_PORT` | `1802` | Port of the remote env service |

### Config Files

| File | Purpose |
|------|---------|
| `config/.env.example` | Environment variable template |
| `config/default.yaml` | Agent config: system prompt, step/cost limits, model parameters, environment settings |
| `config/agsbench_mini_swe_agent-server.yml` | fast-service server config (port 1801, plus remote env stubs) |
| `config/agsbench_mini_swe_agent-client.yml` | fast-service client config for loadgen |
| `config/agsbench_mini_swe_agent-env.yml` | fast-service config for the remote env host (port 1802) |
| `monitoring/prometheus.yml` | Prometheus scrape configuration (supports remote cAdvisor targets) |

## Resource Controls

| Variable | Default | Description |
|----------|---------|-------------|
| `MINI_SWE_CPUSETS` | `0-1,2-3,4-5,6-7` | CPU core pool; each comma-separated range is one sandbox slot |
| `MINI_SWE_MEMORY` | `4g` | Memory limit per SWE-bench sandbox container |
| `MINI_SWEAGENT_CONCURRENCY` | `4` | Maximum parallel in-flight tasks |
| `step_limit` (in `default.yaml`) | `250` | Maximum agent steps per task |
| `cost_limit` (in `default.yaml`) | `3.0` | Maximum LLM cost (USD) per task |
| `timeout` (in `default.yaml`) | `60` | Per-command execution timeout (seconds) |
| `pull_timeout` (in `default.yaml`) | `1800` | Docker image pull timeout (seconds) |
| `MONITORING_UID` / `MONITORING_GID` | `1002` | UID/GID for Prometheus and Grafana containers |
