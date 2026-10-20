# QA x RAG

Benchmark the RAG agent on question-answering tasks using the wiki_qa dataset, with Milvus as the vector store.

## Overview

| Dimension | Value |
|-----------|-------|
| Scenario | qa |
| Agent | rag |
| Dataset | `microsoft/wiki_qa` (pinned revision `3f104672`) |
| Evaluation | Manual / custom (no built-in evaluate script) |
| Hardware | CPU-only; no GPU required (API-mode: LLM + embedding via remote API) |

## Prerequisites

- Docker 24+ and Docker Compose v2
- AgentSysBench repo cloned and set up (`bash scripts/setup.sh`)
- API keys configured (`bash scripts/configure-api.sh`)
- Agent image built (`cd agents/rag && bash scripts/build_image.sh`)
- Loadgen image built (`cd request && bash scripts/build_image.sh`)

## Quick Start

### 1. Prepare data

```bash
cd benchmark/deployment/qa-rag
bash scripts/1.prepare_data.sh
```

Downloads the wiki_qa dataset from HuggingFace and writes a tab-separated manifest to `.cache/natural_questions.txt`.

### 2. Run the benchmark

```bash
bash scripts/2.run.sh
```

### 3. Observe

- RAG service docs: http://localhost:20000/docs
- Milvus health: http://localhost:9091/healthz
- Loadgen logs: `docker logs -f rag-loadgen`

### 4. Cleanup

```bash
bash scripts/3.cleanup.sh
```

### 5. Analyze

From the repo root:

```bash
python benchmark/analysis/run.py qa-rag
```

Report (figures, CSVs, `summary.txt`) goes to `benchmark/analysis/out/qa-rag/`.
Raw data is in `benchmark/data/qa-rag/`; the previous run is moved to
`benchmark/data/_archive/` on each `2.run.sh` (`AGSBENCH_KEEP_DATA=1` to append).
See [benchmark/analysis](../../analysis/README.md#raw-data-layout) for the data layout.

## Configuration

Key environment variables (set in the agent's `config/.env` or via shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | _(required)_ | API key for LLM and embedding endpoints |
| `AGSBENCH_HOME` | _(required)_ | Root path of the AgentSysBench repository |
| `RAG_REQUEST_NUM` | `1` | Number of requests the loadgen sends (paper scale: 3010) |
| `RAG_BM_MODE` | `one-by-one` | Arrival mode |

The RAG agent's behavior is configured through YAML files in `agents/rag/config/api-mode/`:
- `server.yml` -- Service deployment configuration
- `settings.yml` -- LLM model, embedding model, and retrieval parameters

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  rag-loadgen │────▶│ rag-service  │────▶│ milvus-service│
│(agsbench-req)│ HTTP│ (rag:v1)     │gRPC │ (milvus:v2.5) │
└──────────────┘:20000└──────────────┘     └───────────────┘
                                                :19530
```

- **milvus-service** -- Standalone Milvus vector database (v2.5.3) with embedded etcd. Exposed on ports 19530 (gRPC) and 9091 (health).
- **rag-service** -- The RAG coordinator. Launches via `launch_services.py`, indexes documents into Milvus, and serves QA queries on port 20000.
- **rag-loadgen** -- Sends questions from the wiki_qa manifest to the RAG service one-by-one.
