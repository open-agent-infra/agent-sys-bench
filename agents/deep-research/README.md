# Deep Research

Multi-step web research agent with iterative search, summarization, and report generation, instrumented with per-stage telemetry across the full research pipeline.

## Overview

| Dimension | Value |
|-----------|-------|
| Package | `agsbench_deep_research` |
| Image | `deep-research:v1` |
| Scenarios served | ai-search |
| Hardware | GPU required (vLLM for LLM and embedding); API mode and API-only mode available (no GPU) |

## Build

```bash
cd /path/to/agent-sys-bench
bash scripts/setup.sh
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx
cd agents/deep-research
bash scripts/build_image.sh
```

## Serving Interface

This agent serves the following scenario(s) via fast-service RPC:

| Scenario | RPC Endpoint | Request Type | Result Type |
|----------|-------------|--------------|-------------|
| ai-search | `agsbench_deep_research.agents.deep_research.langchain_drs.workflow.deep_research.deep_research_workflow` | `AISearchRequest` | `AISearchResult` |

See `src/agsbench_deep_research/serving.py` for the full interface definition.

## Deployment

Experiments are configured and run from `benchmark/deployment/ai-search-deep-research/`.
See the deployment README for each experiment setting:

- [`benchmark/deployment/ai-search-deep-research/README.md`](../../benchmark/deployment/ai-search-deep-research/README.md)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | `EMPTY` | API key for the LLM endpoint |
| `HF_HOME` | *(required)* | Host path to Hugging Face model cache |
| `HUGGING_FACE_HUB_TOKEN` | *(required)* | Hugging Face Hub API token |
| `TAVILY_API_KEY` | *(optional)* | Tavily search API key (if using Tavily backend) |
| `AGENT_DATA` | *(required)* | Host path to the dataset directory |
| `AGENT_CONFIG` | *(required)* | Host path to the config directory |
| `AGENT_RESULTS` | *(required)* | Host path for output/results |
| `AGENT_CACHE` | *(required)* | Host path for caches (tiktoken, etc.) |
| `AGENT_CPUS` | `4.0` | CPU limit for the agent server |
| `AGENT_MEMORY` | `16G` | Memory limit for the agent server |
| `LLM_MODEL` | *(required)* | vLLM model name (e.g. `Qwen/Qwen3-4B`) |
| `LLM_DEVICE` | *(required)* | GPU device ID for the LLM service |
| `LLM_CPUS` | `8.0` | CPU limit for the vLLM LLM service |
| `LLM_MEMORY` | `32G` | Memory limit for the vLLM LLM service |
| `LLM_GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory fraction for vLLM LLM |
| `EMBEDDING_MODEL` | `jinaai/jina-embeddings-v3` | Embedding model name |
| `EMBEDDING_DEVICE` | *(required)* | GPU device ID for the embedding service |
| `EMBEDDING_CPUS` | `8.0` | CPU limit for the embedding service |
| `EMBEDDING_MEMORY` | `32G` | Memory limit for the embedding service |
| `VLLM_IMAGE` | `agsbench/vllm-log:v0.9.0.1` | Docker image for vLLM services |
| `LEGOAGENT_IMAGE` | `deep-research:v1` | Docker image for the agent server |
| `LOADGEN_CPUS` | `32.0` | CPU limit for the loadgen |
| `LOADGEN_MEMORY` | `64G` | Memory limit for the loadgen |
| `SEARXNG_ENDPOINT` | `http://searxng:8080` | SearXNG search endpoint |
| `WARM_UP` | `0` | Number of warm-up requests before benchmarking |
| `REQUEST_VERSION` | `latest` | Dataset version filter |
| `BENCHMARK_ARGS` | *(required)* | Loadgen arguments: `--agent`, `--dataset`, `--mode`, `--lambda_rate`, `--request_num` |

### Config Files

| File | Purpose |
|------|---------|
| `config/core.service.llm.yaml` | LLM service instances: model name, endpoint, API key (DashScope or local vLLM) |
| `config/core.service.llm-api.yaml` | API-mode LLM config template: DashScope `deepseek-v4-pro` endpoint, no local vLLM needed |
| `config/core.service.embedding.yaml` | Embedding service instances (vLLM or local sentence-transformers) |
| `config/core.service.embedding-api.yaml` | API-mode embedding config template: DashScope `text-embedding-v3` endpoint, no local vLLM needed |
| `config/core.service.search.yaml` | Search engine instances: 11 backends (tavily, arxiv, duckduckgo, wiki, perplexity, googlesearch, exa, linkup, serper, searxng, dummy) with caching |
| `config/core.service.search-api.yaml` | API-only search config: Firecrawl remote API, no local SearXNG/Redis |
| `config/core.service.vdb.yaml` | Vector DB instances (Milvus, Milvus Lite, or simple in-memory) |
| `config/core.service.caching.yaml` | Caching backends (in-memory, file, Redis) |
| `config/deep_research.langchain_drs.yaml` | Research workflow params: `max_search_depth`, `number_of_queries`, `search_api`, per-role model names and generation kwargs, post-processing (summarize, chunk, rerank) |

## Resource Controls

### GPU Mode

| Resource | Value | Notes |
|----------|-------|-------|
| GPU | **Required** | Minimum 2 GPUs (LLM + embedding); multi-model compose uses 4 GPUs |
| Agent server | 4 CPUs / 16 GB RAM (default) | No GPU needed; LLM calls go to vLLM or remote API |
| Loadgen | 32 CPUs / 64 GB RAM (default) | Needs significant resources for concurrent report generation |
| Search backends | 11 configurable | SearXNG (self-hosted, default), Tavily, Serper, Exa, Linkup, DuckDuckGo, Google, Wikipedia, Arxiv, Perplexity, dummy |
| Firecrawl stack | firecrawl + worker + playwright-service + redis | Web scraping and content extraction |

### API Mode

| Resource | Value | Notes |
|----------|-------|-------|
| GPU | **Not required** | All LLM and embedding inference handled by remote API |
| Agent server | 4 CPUs / 16 GB RAM (default) | No GPU needed |
| Services | 7 (no GPU containers) | agent_server, loadgen, firecrawl, worker, playwright-service, searxng, redis |
| API provider | DashScope (default) | Any OpenAI-compatible API works |
| Config templates | `config/core.service.llm-api.yaml`, `config/core.service.embedding-api.yaml` | Copy to `AGENT_CONFIG/setup/` |
