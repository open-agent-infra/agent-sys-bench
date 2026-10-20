# Auto-Research x Pi-AutoResearch

Benchmark the Pi-AutoResearch agent on AutoML research: solving [MLE-bench](https://github.com/openai/mle-bench) Kaggle competitions by iteratively training models, with LiteLLM for model routing.

## Overview

| Dimension | Value |
|-----------|-------|
| Scenario | auto-research |
| Agent | pi-autoresearch |
| Dataset | MLE-bench @ `507f92e1` competitions. Default: `spooky-author-identification` (2 MB, text classification); `--lite`: the 22 MLE-bench Lite competitions (~158 GB) |
| Evaluation | MLE-bench grader on its held-out test split (score, medal, above-median), run in-line after each competition |
| Hardware | 1 NVIDIA GPU for the agent's training code (+ disk for the competition data). Uses `network_mode: host`. |

## Prerequisites

- Docker 24+ and Docker Compose v2
- AgentSysBench repo cloned and set up (`bash scripts/setup.sh`)
- API keys configured (`bash scripts/configure-api.sh`)
- Agent + grader images built (`cd agents/pi-autoresearch && bash scripts/build_image.sh`)
- NVIDIA Container Toolkit (the agent service reserves a GPU)
- A Kaggle account with an API token, and the rules of each competition accepted on kaggle.com (e.g. https://www.kaggle.com/c/spooky-author-identification/rules)
- Loadgen image built (`cd request && bash scripts/build_image.sh`)
- LiteLLM proxy image built (see `agents/pi-autoresearch/README.md`)

## Quick Start

### 1. Prepare data

```bash
cd benchmark/deployment/auto-research-pi-autoresearch
export KAGGLE_API_TOKEN=...            # or put kaggle.json in ~/.kaggle/
bash scripts/1.prepare_data.sh         # default: spooky-author-identification
# bash scripts/1.prepare_data.sh --force --competitions leaf-classification random-acts-of-pizza
# bash scripts/1.prepare_data.sh --force --lite        # all 22 Lite competitions, ~158 GB
```

Writes the manifest `.cache/pi_autoresearch_data.txt`, then uses the grader image to download each competition from Kaggle and build MLE-bench's train/test split in `.cache/mlebench/data/<id>/prepared/{public,private}`. The public part is hard-linked to `.cache/mlebench/public/<id>`, the only data the agent container mounts. Already-prepared competitions are skipped; an existing manifest is kept unless `--force`.

### 2. Run the benchmark

```bash
bash scripts/2.run.sh
```

### 3. Observe

- Grafana: http://localhost:3001 (default `PI_AUTORESEARCH_GRAFANA_PORT`)
- Prometheus: http://localhost:9091 (default `PI_AUTORESEARCH_PROMETHEUS_PORT`)
- Loadgen logs: `docker logs -f pi-autoresearch-loadgen`

### 4. Cleanup

```bash
bash scripts/3.cleanup.sh
```

### 5. Analyze

From the repo root:

```bash
python benchmark/analysis/run.py auto-research-pi-autoresearch
```

Report (figures, CSVs, `summary.txt`) goes to `benchmark/analysis/out/auto-research-pi-autoresearch/`.
Raw data is in `benchmark/data/auto-research-pi-autoresearch/`; the previous run is moved to
`benchmark/data/_archive/` on each `2.run.sh` (`AGSBENCH_KEEP_DATA=1` to append).
See [benchmark/analysis](../../analysis/README.md#raw-data-layout) for the data layout.

### 6. Evaluate

```bash
bash scripts/4.evaluate.sh
```

Grading already happens during the run; this prints each competition's score, the median threshold, medal and above-median from `trajectories/preds.json`, plus the valid-submission, medal and above-median rates. Each `trajectories/<id>/` also keeps the agent's workspace (code, `submission/`, `.auto/log.jsonl`), the pi session, and `<id>.traj.json` with the full MLE-bench grade report.

## Configuration

Key environment variables (set in the agent's `config/.env` or override via shell):

| Variable | Default | Description |
|----------|---------|-------------|
| `PI_PROVIDER` | `openai` | LLM provider (`openai` via LiteLLM proxy) |
| `PI_MODEL` | `deepseek-v3` | Model name |
| `OPENAI_API_KEY` | _(required)_ | API key for LiteLLM backend |
| `OPENAI_BASE_URL` | `http://localhost:4000/v1` | LiteLLM proxy URL |
| `PI_THINKING` | `off` | Enable/disable chain-of-thought |
| `MAX_ITERATIONS` | `5` | Max autoresearch experiments per competition |
| `TASK_TIMEOUT` | `3600` | Time limit per competition in seconds (told to the agent; at the limit pi is killed and the current submission graded) |
| `PI_AUTORESEARCH_CONCURRENCY` | `1` | Parallel loadgen threads (each competition trains on the same GPU) |
| `PI_AUTORESEARCH_REQUEST_NUM` | `1` | Number of competitions to run (paper scale: 22, MLE-bench Lite) |
| `PI_AUTORESEARCH_GPU` | `0` | GPU index for the agent service |
| `PI_AUTORESEARCH_GRADER_PORT` | `5005` | Grader port (`127.0.0.1`) |
| `KAGGLE_API_TOKEN` | _(shell, data prep only)_ | Kaggle API token for `1.prepare_data.sh` |
| `PI_AUTORESEARCH_GRAFANA_PORT` | `3001` | Host port for Grafana |
| `PI_AUTORESEARCH_PROMETHEUS_PORT` | `9091` | Host port for Prometheus |

## Architecture

All services use `network_mode: host`, so they communicate via `localhost` ports rather than Docker DNS.

```
┌──────────────────────┐     ┌──────────────────────────┐     ┌──────────────┐
│pi-autoresearch-loadge│────▶│ pi-autoresearch-service   │────▶│ litellm-proxy│
│  (agsbench-request)  │ HTTP│ (pi-autoresearch:v1, GPU) │ HTTP│   :4000      │
└──────────────────────┘:1902└────────────┬─────────────┘     └──────┬───────┘
                                          │ /validate (agent)        │
                                          │ /grade (runner, token)   ▼
                                          ▼                   Upstream LLM API
                             ┌──────────────────────────┐      (DashScope)
                             │ pi-autoresearch-grader    │
                             │ (MLE-bench, private data) │
                             │        :5005              │
                             └──────────────────────────┘

┌──────────────────────┐     ┌──────────────────────────┐     ┌──────────┐
│pi-autoresearch-grafan│────▶│pi-autoresearch-prometheus │────▶│ cadvisor │
│        :3001         │     │          :9091            │     │  :8081   │
└──────────────────────┘     └──────────────────────────┘     └──────────┘
```

- **litellm-proxy** -- Routes LLM calls through LiteLLM to the configured backend (DashScope). Runs on host network port 4000.
- **pi-autoresearch-service** -- The Pi coding agent. Receives one competition per request on port 1902, trains and iterates on models in a per-competition workspace (public data only), then has the submission graded.
- **pi-autoresearch-grader** -- MLE-bench grading server with the private test answers: `/validate` checks the submission format for the agent; `/grade` (token set by `2.run.sh`) returns the score and medal to the runner.
- **pi-autoresearch-loadgen** -- Sends competitions from the manifest to the agent service.
- **cadvisor / prometheus / grafana** -- Resource monitoring stack. cAdvisor exposed on port 8081; Prometheus on 9091; Grafana on 3001.
