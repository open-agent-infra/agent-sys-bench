# PiAutoResearch Agent

The [pi](https://pi.dev/) coding agent with the [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) extension -- an autonomous experiment loop that edits code, measures a metric, keeps improvements and reverts regressions. Here it does AutoML research on [MLE-bench](https://github.com/openai/mle-bench) (Kaggle) competitions, following the reference pi-autoresearch MLE-bench agent: build a baseline `submission.csv`, then iterate on models/features with the autoresearch tools, using a cross-validation score as the metric.

Per task the runner builds a workspace with the competition's public data (`data/`, read-only), `submission/`, the MLE-bench instructions and a `validate_submission.sh` helper; starts pi with the `autoresearch-create` skill under the `TASK_TIMEOUT` time limit, capping the loop at `MAX_ITERATIONS` experiments (`.auto/config.json`); then grades `submission/submission.csv` with the MLE-bench grader. The grader runs as a sidecar (`pi-autoresearch-grader:v1`, built from `grader/`) that holds the private test answers: the agent can only call its format-checking `/validate`, while `/grade` needs a token the agent's environment never gets.

## Overview

| Dimension | Value |
|-----------|-------|
| Package | `agsbench_pi_autoresearch` |
| Images | `pi-autoresearch:v1` (agent + service), `pi-autoresearch-grader:v1` (MLE-bench grader / data preparer) |
| Scenarios served | auto-research |
| Hardware | 1 NVIDIA GPU for the agent's training code (`PI_AUTORESEARCH_GPU`); the LLM is remote |

The agent's training code runs in a separate Python env, `/opt/agent-env` (put first on pi's `PATH`), with the MLE-bench agent image's versions of numpy/pandas/scikit-learn/LightGBM/XGBoost/CatBoost/PyTorch/transformers etc. (`agent-env-requirements.txt`).

## Build

```bash
cd /path/to/agent-sys-bench
bash scripts/setup.sh
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx
cd agents/pi-autoresearch
bash scripts/build_image.sh   # builds pi-autoresearch:v1 and pi-autoresearch-grader:v1
```

## Serving Interface

This agent serves the following scenario(s) via fast-service RPC:

| Scenario | RPC Endpoint | Request Type | Result Type |
|----------|-------------|--------------|-------------|
| auto-research | `agsbench_pi_autoresearch.request.solve_task` | `AutoResearchTaskRequest` | `AutoResearchTaskResult` |

See `src/agsbench_pi_autoresearch/serving.py` for the full interface definition.

## Deployment

Experiments are configured and run from `benchmark/deployment/auto-research-pi-autoresearch/`.
See the deployment README for each experiment setting:

- [`benchmark/deployment/auto-research-pi-autoresearch/README.md`](../../benchmark/deployment/auto-research-pi-autoresearch/README.md)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PI_PROVIDER` | `openai` | LLM provider name for the Pi agent (use `openai` with LiteLLM proxy) |
| `PI_MODEL` | `deepseek-v3` | Model name passed to the Pi agent |
| `DEEPSEEK_API_KEY` | *(required)* | API key for the DeepSeek endpoint |
| `OPENAI_API_KEY` | *(required)* | API key for OpenAI-compatible sub-components |
| `OPENAI_API_BASE` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | Base URL for OpenAI-compatible sub-components |
| `PI_THINKING` | `high` | Extended thinking / reasoning mode level |
| `MAX_ITERATIONS` | `5` | Maximum research iterations per task |
| `TASK_TIMEOUT` | `3600` | Per-competition time limit in seconds (told to the agent) |
| `PI_AUTORESEARCH_CONCURRENCY` | `1` | Number of parallel task workers |
| `FIRECRAWL_API_KEY` | *(empty)* | API key for Firecrawl web search (optional) |
| `KAGGLE_API_TOKEN` | *(required for data prep)* | Kaggle API token, read from the shell by `1.prepare_data.sh` (or `~/.kaggle/kaggle.json`) |
| `PI_AUTORESEARCH_GPU` | `0` | GPU index given to the agent service |
| `PI_AUTORESEARCH_SHM` | `16g` | Shared memory for the agent service (PyTorch data loaders) |
| `PI_AUTORESEARCH_GRADER_PORT` | `5005` | Host port of the MLE-bench grader (`127.0.0.1` only) |
| `PI_AUTORESEARCH_BM_MODE` | `one-by-one` | Loadgen arrival pattern |
| `PI_AUTORESEARCH_BM_RPS` | `1` | Target requests per second |
| `PI_AUTORESEARCH_REQUEST_NUM` | `1` | Number of competitions to run (paper scale: 22, MLE-bench Lite) |
| `PI_AUTORESEARCH_MANIFEST` | `pi_autoresearch_data.txt` | Manifest filename inside `.cache/` |
| `PI_AUTORESEARCH_CADVISOR_PORT` | `8081` | Host port for cAdvisor |
| `PI_AUTORESEARCH_PROMETHEUS_PORT` | `9091` | Host port for Prometheus |
| `PI_AUTORESEARCH_GRAFANA_PORT` | `3001` | Host port for Grafana |

### Config Files

| File | Purpose |
|------|---------|
| `config/.env.example` | Environment variable template |
| `config/pi-autoresearch-server.yml` | fast-service server config (port 1902) |
| `config/pi-autoresearch-client.yml` | fast-service client config for loadgen |
| `config/openai-provider-patch.js` | Patches the Pi OpenAI provider to use `OPENAI_BASE_URL` as the base URL, enabling LiteLLM proxy routing |
| `config/models.json` | Provider config override; sets `openai` provider's `baseUrl` to the LiteLLM proxy (`http://localhost:4000/v1`) |
| `monitoring/prometheus.yml` | Prometheus scrape configuration |

## Resource Controls

| Variable | Default | Description |
|----------|---------|-------------|
| `PI_AUTORESEARCH_CONCURRENCY` | `1` | Maximum parallel in-flight tasks |
| `MAX_ITERATIONS` | `5` | Maximum research loop iterations per task |
| `TASK_TIMEOUT` | `3600` | Time limit per competition (seconds); pi and its training processes are killed at the limit and the current submission is graded |
| `PI_THINKING` | `high` | Reasoning effort level (`low`, `medium`, `high`) |
| `MONITORING_UID` / `MONITORING_GID` | `1002` | UID/GID for Prometheus and Grafana containers |
