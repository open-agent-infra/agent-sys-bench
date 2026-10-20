# AI-Search x Deep-Research

Benchmark the Deep-Research (LegoAgent) agent on AI-powered deep search and research tasks, using pure API mode with no local infrastructure.

## Overview

| Dimension | Value |
|-----------|-------|
| Scenario | ai-search |
| Agent | deep-research |
| Dataset | `cais/hle` (default, rev `5a81a4c7`), also supports `gaia-benchmark/GAIA` (rev `682dd723`), `google/frames-benchmark` (rev `58d9fb63`) |
| Evaluation | Manual / custom (no built-in evaluate script) |
| Hardware | CPU-only; configurable limits (default 4 CPUs, 16 GB RAM for agent) |

## Prerequisites

- Docker 24+ and Docker Compose v2
- AgentSysBench repo cloned and set up (`bash scripts/setup.sh`)
- API keys configured (`bash scripts/configure-api.sh`)
- Agent image built (`cd agents/deep-research && bash scripts/build_image.sh`)
- Loadgen image built (`cd request && bash scripts/build_image.sh`)

## Quick Start

### 1. Prepare data

```bash
cd benchmark/deployment/ai-search-deep-research
bash scripts/1.prepare_data.sh
```

Downloads the selected dataset (`--dataset hle|gaia|frames`, default `hle`) from HuggingFace and writes a JSON-per-line manifest to `.cache/<dataset>.txt`.
`hle` and `gaia` are gated: accept their terms on huggingface.co with your account and `export HF_TOKEN=<token>` first.
`frames` is public. To run a non-default dataset, pass the same name at run time, e.g. `DATASET=frames bash scripts/2.run.sh`.

### 2. Run the benchmark

```bash
bash scripts/2.run.sh
```

### 3. Observe

- Agent API docs: http://localhost:20000/docs (inside the compose network)
- Loadgen logs: `docker logs -f deep-research-benchmarking-loadgen-1`

### 4. Cleanup

```bash
bash scripts/3.cleanup.sh
```

### 5. Analyze

From the repo root:

```bash
python benchmark/analysis/run.py ai-search-deep-research
```

Report (figures, CSVs, `summary.txt`) goes to `benchmark/analysis/out/ai-search-deep-research/`.
Raw data is in `benchmark/data/ai-search-deep-research/`; the previous run is moved to
`benchmark/data/_archive/` on each `2.run.sh` (`AGSBENCH_KEEP_DATA=1` to append).
See [benchmark/analysis](../../analysis/README.md#raw-data-layout) for the data layout.

## Configuration

Key environment variables (set via shell or in a `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(required)_ | API key for LLM and embedding (DashScope) |
| `FIRECRAWL_API_KEY` | _(optional)_ | Firecrawl remote API key for web search |
| `LEGOAGENT_DEFAULT_LLM` | `deepseek-v4-pro` | Default LLM model name |
| `LEGOAGENT_DEFAULT_EMBEDDING` | `text-embedding-v3` | Default embedding model |
| `DATASET` | `hle` | Dataset name; reads `.cache/<DATASET>.txt` (prepare with `1.prepare_data.sh --dataset <name>`) |
| `REQUEST_NUM` | `1` | Number of requests to run (paper scale: 102) |
| `BENCHMARK_MODE` | `one-by-one` | Benchmark mode |
| `AGENT_CPUS` | `4.0` | CPU limit for the agent container |
| `AGENT_MEMORY` | `16G` | Memory limit for the agent container |
| `LOADGEN_CPUS` | `4.0` | CPU limit for the loadgen container |
| `LOADGEN_MEMORY` | `8G` | Memory limit for the loadgen container |
| `LOG_LEVEL` | `DEBUG` | Agent log level |

## Architecture

This is the lightest deployment -- only two containers, zero local infrastructure (no vector DB, no search engine, no Redis). Everything uses remote APIs.

```
┌──────────────┐     ┌──────────────────────┐
│   loadgen    │────▶│    agent_server      │────▶ DashScope API (LLM + Embedding)
│(agsbench-req)│ HTTP│ (deep-research:v1)   │────▶ Firecrawl API (Web Search)
└──────────────┘:20000└──────────────────────┘
```

- **agent_server** -- The LegoAgent deep-research server. Receives search/research queries on port 20000. Uses an in-memory vector store and file-based caching. Configured via YAML files in `agents/deep-research/config/api-mode/`.
- **loadgen** -- Sends queries from the prepared manifest to the agent server.
