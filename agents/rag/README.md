# RAG

Retrieval-Augmented Generation pipeline for question-answering, with per-stage telemetry across the full retrieval-generation stack.

## Overview

| Dimension | Value |
|-----------|-------|
| Package | `agsbench_rag` |
| Image | `rag:v1` |
| Scenarios served | qa |
| Hardware | GPU required (vLLM for LLM, embedding, reranking, situating); API mode available (no GPU) |

## Build

```bash
cd /path/to/agent-sys-bench
bash scripts/setup.sh
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx
cd agents/rag
bash scripts/build_image.sh
```

## Serving Interface

This agent serves the following scenario(s) via fast-service RPC:

| Scenario | RPC Endpoint | Request Type | Result Type |
|----------|-------------|--------------|-------------|
| qa | `agsbench_rag.rag_e2e.naive_rag` | `QARequest` | `QAResult` |

See `src/agsbench_rag/serving.py` for the full interface definition.

## Deployment

Experiments are configured and run from `benchmark/deployment/qa-rag/`.
See the deployment README for each experiment setting:

- [`benchmark/deployment/qa-rag/README.md`](../../benchmark/deployment/qa-rag/README.md)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | API key for the LLM endpoint |
| `HF_HOME` | *(required)* | Host path to Hugging Face model cache |
| `HUGGING_FACE_HUB_TOKEN` | *(required)* | Hugging Face Hub API token |
| `AGENT_DATA` | *(required)* | Host path to the dataset directory |
| `AGENT_CONFIG` | *(required)* | Host path to the config directory |
| `AGENT_RESULTS` | *(required)* | Host path for output/results |
| `AGENT_CACHE` | *(required)* | Host path for caches (tiktoken, etc.) |
| `AGENT_CPUS` | `4.0` | CPU limit for the RAG coordinator |
| `AGENT_MEMORY` | `16G` | Memory limit for the RAG coordinator |
| `AGENT_PORT` | `20000` | Host port for the RAG coordinator |
| `LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | vLLM LLM model |
| `LLM_DEVICE` | *(required)* | GPU device ID for the LLM service |
| `LLM_CPUS` | `8.0` | CPU limit for vLLM LLM service |
| `LLM_MEMORY` | `16G` | Memory limit for vLLM LLM service |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory fraction for vLLM LLM |
| `EMBEDDING_MODEL` | `Alibaba-NLP/gte-Qwen2-1.5B-instruct` | Embedding model |
| `EMBEDDING_DEVICE` | *(required)* | GPU device ID for the embedding service |
| `EMBEDDING_CPUS` | `8.0` | CPU limit for embedding service |
| `EMBEDDING_MEMORY` | `16G` | Memory limit for embedding service |
| `EMBEDDING_ENGINE` | `vllm` | Embedding backend: `vllm`, `tei`, or `infinity` |
| `RERANKING_MODEL` | `BAAI/bge-reranker-large` | Reranking model |
| `RERANKING_DEVICE` | `0` | GPU device ID for the reranking service |
| `RERANKING_CPUS` | `8.0` | CPU limit for reranking service |
| `RERANKING_MEMORY` | `16G` | Memory limit for reranking service |
| `SITUATING_LLM_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Situating (contextual) LLM model |
| `SITUATING_LLM_DEVICE` | *(required)* | GPU device ID for the situating service |
| `MILVUS_CPUS` | `32.0` | CPU limit for Milvus |
| `MILVUS_MEMORY` | `64G` | Memory limit for Milvus |
| `LOADGEN_CPUS` | `32.0` | CPU limit for the loadgen |
| `LOADGEN_MEMORY` | `64G` | Memory limit for the loadgen |
| `NLTK_DATA` | `/data/nltk_data` | Host path to NLTK data |

### Config Files

| File | Purpose |
|------|---------|
| `config/settings.yml` | RAG pipeline config: LLM model/key/endpoint, vector DB settings, embedding model/device, chunking params, retriever type, generation method |
| `config/settings-api.yml` | API-mode template: same structure as `settings.yml` but configured for remote DashScope endpoints (LLM, embedding, situating). Copy to `AGENT_CONFIG/settings.yml` when using API mode. |
| `config/rag-server.yml` | fast-service server config (port 20100, 3 RPCs: `naive_rag`, `advanced_rag`, `dynamic_rag`) |
| `config/rag-client.yml` | fast-service client config |

## Resource Controls

### GPU Mode

| Resource | Value | Notes |
|----------|-------|-------|
| GPU | **Required** | Minimum 4 GPUs recommended (LLM, embedding, reranking, situating each on separate device) |
| Milvus stack | etcd + MinIO + Milvus standalone | Managed via `milvus-standalone.yaml` include; `milvus-services.yaml` for resource limits |
| Embedding backends | vLLM, TEI, Infinity | Selectable via `EMBEDDING_ENGINE` env var |
| Dataset | `microsoft/wiki_qa` (and others) | Specified via `BENCHMARK_ARGS --dataset` |

### API Mode

| Resource | Value | Notes |
|----------|-------|-------|
| GPU | **Not required** | All inference handled by remote API |
| Services | Milvus + rag-service + loadgen | Only 3 services instead of 7+ |
| API provider | DashScope (default) | Any OpenAI-compatible API works |
| Config | `config/settings-api.yml` | Template for API-mode settings |
