# Tool-Rich-Assistant x Claude

Benchmark Claude Code on tool-rich coding assistant tasks from PinchBench, with an Anthropic-to-OpenAI translation proxy.

## Overview

| Dimension | Value |
|-----------|-------|
| Scenario | tool-rich-assistant |
| Agent | claude |
| Dataset | PinchBench `pinchbench/skill.git` (pinned commit `819384ae`) |
| Evaluation | Manual / custom (no built-in evaluate script) |
| Hardware | CPU-only; no GPU required |

## Prerequisites

- Docker 24+ and Docker Compose v2
- AgentSysBench repo cloned and set up (`bash scripts/setup.sh`)
- API keys configured (`bash scripts/configure-api.sh`)
- Agent image built (`cd agents/claude && bash scripts/build_image.sh`)
- Loadgen image built (`cd request && bash scripts/build_image.sh`)

## Quick Start

### 1. Prepare data

```bash
cd benchmark/deployment/tool-rich-assistant-claude
bash scripts/1.prepare_data.sh
```

Clones the PinchBench repository (pinned commit) and generates a manifest of task `.md` files in `.cache/claude_tasks.txt`.

### 2. Run the benchmark

```bash
bash scripts/2.run.sh
```

### 3. Observe

- Grafana: http://localhost:3002 (default `CLAUDE_GRAFANA_PORT`)
- Prometheus: http://localhost:9092 (default `CLAUDE_PROMETHEUS_PORT`)
- Loadgen logs: `docker logs -f claude-loadgen`

### 4. Cleanup

```bash
bash scripts/3.cleanup.sh
```

### 5. Analyze

From the repo root:

```bash
python benchmark/analysis/run.py tool-rich-assistant-claude
```

Report (figures, CSVs, `summary.txt`) goes to `benchmark/analysis/out/tool-rich-assistant-claude/`.
Raw data is in `benchmark/data/tool-rich-assistant-claude/`; the previous run is moved to
`benchmark/data/_archive/` on each `2.run.sh` (`AGSBENCH_KEEP_DATA=1` to append).
See [benchmark/analysis](../../analysis/README.md#raw-data-layout) for the data layout.

## Configuration

Key environment variables (set in the agent's `config/.env` or override via shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | _(required)_ | API key (used by the translation proxy) |
| `UPSTREAM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Upstream OpenAI-compatible LLM endpoint |
| `UPSTREAM_API_KEY` | `${ANTHROPIC_API_KEY}` | API key for the upstream endpoint |
| `CLAUDE_MODEL` | `deepseek-v4-pro` | Model name passed to the agent |
| `CLAUDE_CONCURRENCY` | `4` | Parallel loadgen threads |
| `CLAUDE_REQUEST_NUM` | `1` | Number of tasks to run (paper scale: 500) |
| `CLAUDE_BM_MODE` | `one-by-one` | Benchmark mode |
| `CLAUDE_EFFORT` | _(empty)_ | Agent effort level |
| `CLAUDE_GRAFANA_PORT` | `3002` | Host port for Grafana |
| `CLAUDE_PROMETHEUS_PORT` | `9092` | Host port for Prometheus |

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│ claude-loadgen│────▶│claude-service│────▶│ anthropic-proxy  │────▶ Upstream LLM API
│(agsbench-req)│ HTTP│ (claude:v1)  │ HTTP│ (claude:v1)      │      (DashScope)
└──────────────┘:1703└──────────────┘     │  :8090           │
                                          └──────────────────┘

┌──────────────┐     ┌────────────────┐     ┌──────────┐
│claude-grafana│────▶│claude-prometheu│────▶│ cadvisor │
│    :3002     │     │     :9092      │     │          │
└──────────────┘     └────────────────┘     └──────────┘
```

- **anthropic-proxy** -- Translates Anthropic SDK calls (`/v1/messages`) to OpenAI chat-completions and forwards to the upstream API (DashScope/DeepSeek).
- **claude-service** -- The Claude Code agent. Receives tasks on port 1703 and executes them using the Claude CLI. Task results and session data are written to `runs/latest/`.
- **claude-loadgen** -- Sends task paths from the PinchBench manifest to the agent service.
- **cadvisor / prometheus / grafana** -- Resource monitoring stack. Ports are offset (9092/3002) to avoid collisions with other deployments.
