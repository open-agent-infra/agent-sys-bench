# HuggingGPT

LLM-orchestrated multi-model agent that decomposes multimodal requests into task plans, selects specialist Hugging Face models, runs inference, and synthesizes results.

## Overview

| Dimension | Value |
|-----------|-------|
| Package | `agsbench_hugginggpt` |
| Image (coordinator) | `hugginggpt:v1` |
| Image (models server) | `hugginggpt_models_server:v1` |
| Scenarios served | multimodal |
| Hardware | Multi-GPU: 7 local HF models on 3 GPUs (~18 GB VRAM total; verified on 8× RTX 4090D) |

## Build

```bash
cd /path/to/agent-sys-bench
bash scripts/setup.sh
bash scripts/configure-api.sh --llm-preset deepseek --llm-key sk-xxx
cd agents/hugginggpt
bash scripts/build_coordinator.sh
bash scripts/build_models_server.sh
```

## Serving Interface

| Scenario | RPC Endpoint | Request Type | Result Type |
|----------|-------------|--------------|-------------|
| multimodal | `agsbench_hugginggpt.hugginggpt_e2e.hugginggpt_e2e` | `MultimodalRequest` | `MultimodalResult` |

## Architecture

Two containers:
- **coordinator** (`hugginggpt:v1`): LLM orchestration (task planning → model selection → inference routing → response generation). CPU only.
- **models_server** (`hugginggpt_models_server:v1`): 7 local HF models on GPU (DETR, SDXL, ViT-GPT2, VilT, RMBG, NSFW detector, SDXL-refiner).

## Deployment

See [deployment README](../../benchmark/deployment/multimodal-hugginggpt/README.md).

## Configuration

| File | Purpose |
|------|---------|
| `config/settings.yml` | LLM API config (4 stages: task_planning, model_selection, model_inference, response_generation) |
| `config/models_server_config.yaml` | GPU device assignments for 7 HF models |
| `config/hugginggpt_server.yml` | fast-service server config (port 20101) |
| `config/hugginggpt_client.yml` | fast-service client config |
