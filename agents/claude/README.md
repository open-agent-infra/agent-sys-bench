# Claude Code Agent

Benchmarks the Claude Code CLI on tool-rich assistant tasks using WildClawBench, PinchBench, and TerminalBench datasets.

## Overview

| Dimension | Value |
|-----------|-------|
| Package | `agsbench_claude` |
| Image | `claude:v1` |
| Scenarios served | tool-rich-assistant |
| Hardware | CPU only |

## Build

```bash
cd /path/to/agent-sys-bench
bash scripts/setup.sh
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx
cd agents/claude
bash scripts/build_image.sh
```

## Serving Interface

This agent serves the following scenario(s) via fast-service RPC:

| Scenario | RPC Endpoint | Request Type | Result Type |
|----------|-------------|--------------|-------------|
| tool-rich-assistant | `agsbench_claude.run_task.run_task` | `ToolRichTaskRequest` | `ToolRichTaskResult` |

See `src/agsbench_claude/serving.py` for the full interface definition.

## Deployment

Experiments are configured and run from `benchmark/deployment/tool-rich-assistant-claude/`.
See the deployment README for each experiment setting:

- [`benchmark/deployment/tool-rich-assistant-claude/README.md`](../../benchmark/deployment/tool-rich-assistant-claude/README.md)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | API key for the LLM endpoint |
| `ANTHROPIC_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Base URL for the Anthropic-compatible API |
| `CLAUDE_MODEL` | `deepseek-v4-pro` | Model name passed to the Claude Code CLI |
| `CLAUDE_CONCURRENCY` | `4` | Number of parallel task workers |
| `CLAUDE_PROMETHEUS_PORT` | `9092` | Host port for Prometheus |
| `CLAUDE_GRAFANA_PORT` | `3002` | Host port for Grafana |
| `CLAUDE_BM_MODE` | `poisson` | Loadgen arrival pattern (`poisson` or `closed`) |
| `CLAUDE_BM_RPS` | `100` | Target requests per second (poisson mode) |
| `CLAUDE_REQUEST_NUM` | `1` | Number of tasks to run (paper scale: 500) |
| `CLAUDE_MANIFEST` | `claude_tasks.txt` | Manifest filename inside `.cache/` |
| `CLAUDE_DATA_DIR` | `./.cache` (deployment dir) | Host dir with the manifest and task repos from `1.prepare_data.sh`; mounted at `/data/tasks` |
| `CLAUDE_EFFORT` | *(empty)* | Optional effort level passed to Claude Code |

### Config Files

| File | Purpose |
|------|---------|
| `config/.env.example` | Environment variable template |
| `config/claude-server.yml` | fast-service server config (port 1703) |
| `config/claude-client.yml` | fast-service client config for loadgen |
| `config/claude-mcp-server.yml` | MCP-mode server config (port 1707) |
| `config/claude-mcp-client.yml` | MCP-mode client config |
| `monitoring/prometheus.yml` | Prometheus scrape configuration |

## Resource Controls

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_CONCURRENCY` | `4` | Maximum parallel in-flight tasks |
| `CLAUDE_UID` / `CLAUDE_GID` | `1002` | UID/GID for the service container process |
| `MONITORING_UID` / `MONITORING_GID` | `1000` | UID/GID for Prometheus and Grafana containers |
