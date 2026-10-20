# Codex Agent

Benchmarks the OpenAI Codex CLI on TerminalBench terminal-use tasks, using LiteLLM as a Responses API translation proxy.

## Overview

| Dimension | Value |
|-----------|-------|
| Package | `agsbench_codex` |
| Image | `codex:v1` |
| Scenarios served | terminal-use |
| Hardware | CPU only; host Docker socket required (builds and runs each TerminalBench task's compose project) |

## Build

```bash
cd /path/to/agent-sys-bench
bash scripts/setup.sh
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx
cd agents/codex
bash scripts/build_image.sh
```

## Serving Interface

This agent serves the following scenario(s) via fast-service RPC:

| Scenario | RPC Endpoint | Request Type | Result Type |
|----------|-------------|--------------|-------------|
| terminal-use | `agsbench_codex.request.solve_terminal_bench` | `TerminalTaskRequest` | `TerminalTaskResult` |

See `src/agsbench_codex/serving.py` for the full interface definition.

## Deployment

Experiments are configured and run from `benchmark/deployment/terminal-use-codex/`.
See the deployment README for each experiment setting:

- [`benchmark/deployment/terminal-use-codex/README.md`](../../benchmark/deployment/terminal-use-codex/README.md)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | API key for the LLM endpoint |
| `OPENAI_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Base URL for the OpenAI-compatible API |
| `OPENAI_API_BASE` | *(same as BASE_URL)* | Alternate base URL env var |
| `CODEX_MODEL` | `deepseek-v4-pro` | Model name passed to Codex |
| `CODEX_CPUSETS` | `0-1,2-3,4-5,6-7` | Comma-separated CPU core pairs; pool size determines max concurrency |
| `CODEX_CONCURRENCY` | `4` | Number of parallel loadgen threads |
| `CODEX_BM_MODE` | `poisson` | Loadgen arrival pattern |
| `CODEX_BM_RPS` | `100` | Target requests per second |
| `CODEX_REQUEST_NUM` | `1` | Number of tasks to run (paper scale: 220) |
| `CODEX_STEP_TIMEOUT` | `0` | Agent timeout (seconds); `0` uses the task's `max_agent_timeout_sec` |
| `CODEX_TEST_TIMEOUT` | `0` | Test timeout (seconds); `0` uses the task's `max_test_timeout_sec` |
| `CODEX_LITELLM_PORT` | `4001` | Host port for the codex-proxy (LiteLLM frontend) |
| `CODEX_PROMETHEUS_PORT` | `9090` | Host port for Prometheus |
| `CODEX_GRAFANA_PORT` | `3000` | Host port for Grafana |
| `CODEX_MANIFEST` | `codex_data.txt` | Manifest filename inside `.cache/` |

### Config Files

| File | Purpose |
|------|---------|
| `config/.env.example` | Environment variable template |
| `config/codex-server.yml` | fast-service server config (port 1903) |
| `config/codex-client.yml` | fast-service client config for loadgen |
| `config/litellm-config.yaml` | LiteLLM proxy model routing (Responses API to chat-completions) |
| `scripts/codex_proxy.py` | Request-scrubbing proxy between Codex and LiteLLM |
| `monitoring/prometheus.yml` | Prometheus scrape configuration |

## Resource Controls

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_CPUSETS` | `0-1,2-3,4-5,6-7` | CPU core pool; each comma-separated range is one sandbox slot |
| `CODEX_CONCURRENCY` | `4` | Maximum parallel in-flight tasks |
| `MONITORING_UID` / `MONITORING_GID` | `1002` | UID/GID for Prometheus and Grafana containers |
