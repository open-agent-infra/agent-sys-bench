# Multimodal × HuggingGPT

Benchmark HuggingGPT — an LLM-orchestrated multi-model agent that decomposes
multimodal requests into task plans, selects specialist Hugging Face models,
runs inference, and synthesizes results.

## Overview

| Dimension | Value |
|-----------|-------|
| Scenario | multimodal |
| Agent | hugginggpt |
| Dataset | HuggingGPT hybrid image-text benchmark |
| Evaluation | Response quality (LLM judge) |
| Hardware | Multi-GPU: 7 local HF models on 3 GPUs (~18 GB VRAM total; verified on 8× RTX 4090D) |

## Prerequisites

- Docker 24+ with NVIDIA Container Toolkit
- 3+ NVIDIA GPUs (RTX 4090 or better)
- ~50 GB disk for HuggingFace model cache
- An OpenAI-compatible LLM API key (DeepSeek, OpenAI)

## Architecture

```
hugginggpt-loadgen ──HTTP──▶ hugginggpt-coordinator ──HTTP──▶ hugginggpt-models-server
(agsbench-request:v1)         (hugginggpt:v1)                  (hugginggpt_models_server:v1)
                                  │                                    │
                                  ▼                                    ▼
                              LLM API                          7 HF models on GPU
                              (DeepSeek)                        (DETR, SDXL, ViT-GPT2, ...)
```

## Quick Start

### 1. Build images
```bash
cd agents/hugginggpt
bash scripts/build_coordinator.sh
bash scripts/build_models_server.sh
```

### 2. Prepare data
```bash
cd benchmark/deployment/multimodal-hugginggpt
bash scripts/1.prepare_data.sh
```

### 3. Run
```bash
bash scripts/2.run.sh
```

### 4. Observe
- Grafana: http://localhost:3004 (`HUGGINGGPT_GRAFANA_PORT`)
- Prometheus: http://localhost:9095 (`HUGGINGGPT_PROMETHEUS_PORT`)
- Logs: `docker logs -f hugginggpt-loadgen`

### 5. Cleanup
```bash
bash scripts/3.cleanup.sh
```

### 6. Analyze

From the repo root:

```bash
python benchmark/analysis/run.py multimodal-hugginggpt
```

Report (figures, CSVs, `summary.txt`) goes to `benchmark/analysis/out/multimodal-hugginggpt/`.
Raw data is in `benchmark/data/multimodal-hugginggpt/`; the previous run is moved to
`benchmark/data/_archive/` on each `2.run.sh` (`AGSBENCH_KEEP_DATA=1` to append).
See [benchmark/analysis](../../analysis/README.md#raw-data-layout) for the data layout.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_HOME` | `~/.cache/huggingface` | Host HuggingFace model cache (mounted at `/mnt/data/huggingface` in models-server) |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HF mirror for model downloads |
| `HUGGINGGPT_REQUEST_NUM` | `1` | Number of requests the loadgen sends (paper scale: 104) |
| `HUGGINGGPT_GRAFANA_PORT` / `HUGGINGGPT_PROMETHEUS_PORT` | `3004` / `9095` | Host ports |
| LLM API | settings.yml | DeepSeek API (task planning, model selection, inference, response) |

## Notes

- Models server uses `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` base
- 7 HF models are loaded across GPUs 0, 2, 3 (configurable in `models_server_config.yaml`)
- First run downloads ~50 GB of HF models — subsequent runs use cache

## Image Data Preparation

The default task (`image-to-text`) downloads tiny-imagenet from Hugging Face
(via `HF_ENDPOINT`) and saves the images to `.cache/images/`, with a manifest
at `.cache/hugginggpt_data.txt`:

```bash
bash scripts/1.prepare_data.sh --task image-to-text --num_samples 5
```

For text-only testing (no images needed):
```bash
bash scripts/1.prepare_data.sh --task text-generation --num_samples 5
```

Images are mounted at `/data/images` in both coordinator and models_server
containers. The manifest references this container-internal path.
