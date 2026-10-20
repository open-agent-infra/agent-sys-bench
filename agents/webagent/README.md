# WebAgent

BrowserGym-based web agent that completes browser interaction tasks (form filling, click sequences, drag-and-drop) against MiniWoB++ and WebArena environments.

## Overview

| Dimension | Value |
|-----------|-------|
| Package | `agsbench_webagent` |
| Image | `webagent:v1` |
| Scenarios served | browser-use |
| Hardware | CPU only; 2 GB shared memory for Chromium |

## Build

```bash
cd /path/to/agent-sys-bench
bash scripts/setup.sh
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx
cd agents/webagent
bash scripts/build_image.sh
```

## Serving Interface

This agent serves the following scenario(s) via fast-service RPC:

| Scenario | RPC Endpoint | Request Type | Result Type |
|----------|-------------|--------------|-------------|
| browser-use | `agsbench_webagent.request.solve_task` | `BrowserTaskRequest` | `BrowserTaskResult` |

See `src/agsbench_webagent/serving.py` for the full interface definition.

## Deployment

Experiments are configured and run from `benchmark/deployment/browser-use-webagent/`.
See the deployment README for each experiment setting:

- [`benchmark/deployment/browser-use-webagent/README.md`](../../benchmark/deployment/browser-use-webagent/README.md)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | API key for the OpenAI-compatible LLM endpoint |
| `OPENAI_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | LLM endpoint base URL |
| `WEBAGENT_MODEL` | `deepseek-v4-pro` | Model name passed to chat completions |
| `WEBAGENT_CONCURRENCY` | `4` | Number of parallel tasks (loadgen thread pool) |
| `WEBAGENT_REQUEST_NUM` | `1` | Number of tasks to run from the manifest (paper scale: 83) |
| `WEBAGENT_MAX_STEPS` | `30` | Maximum browser interaction steps per task |
| `WEBAGENT_BM_MODE` | `poisson` | Loadgen scheduler mode |
| `WEBAGENT_BM_RPS` | `100` | Target arrival rate |
| `WEBAGENT_MANIFEST` | `browsergym_data.txt` | Manifest file under `.cache/` |
| `WEBAGENT_PROMETHEUS_PORT` | `9090` | Host port for Prometheus |
| `WEBAGENT_GRAFANA_PORT` | `3000` | Host port for Grafana |
| `MINIWOB_URL` | `file:///opt/miniwob-plusplus/miniwob/html/miniwob/` | MiniWoB++ static HTML path (set by Dockerfile) |
| `WA_SHOPPING_ADMIN` | `http://shopping-admin:80/admin` | Magento admin panel URL |
| `WA_SHOPPING` | `http://shopping-admin:80` | Magento storefront URL |

### Config Files

| File | Purpose |
|------|---------|
| `config/.env.example` | Environment variable template (API keys, model, endpoint) |
| `config/webagent-server.yml` | fast-service server config (port 1902, `solve_task` RPC) |
| `config/webagent-client.yml` | fast-service client config (connects to `browsergym-service:1902`) |

## Resource Controls

| Resource | Value | Notes |
|----------|-------|-------|
| GPU | None | CPU-only workload |
| `shm_size` | `2gb` | Required by Chromium for rendering |
| `cap_add` | `SYS_ADMIN` | Required by Chromium sandboxing |
| `security_opt` | `seccomp=unconfined` | Chromium compatibility |
