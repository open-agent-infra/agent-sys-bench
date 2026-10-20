# Experiment Deployments

Each subdirectory is one **experiment setting**: a scenario (request dataset +
evaluation) paired with an agent (tool + model + orchestrator), deployed with
Docker Compose.

| Scenario | Agent | Deployment | Hardware | Paper scale |
|----------|-------|------------|----------|---------:|
| QA | RAG | [`qa-rag`](qa-rag/README.md) | CPU | 3010 |
| MultiModal | HuggingGPT | [`multimodal-hugginggpt`](multimodal-hugginggpt/README.md) | 3 GPUs (~18 GB VRAM total) | 104 |
| AISearch | DeepResearch | [`ai-search-deep-research`](ai-search-deep-research/README.md) | CPU | 102 |
| SWE | MiniSWE | [`swe-mini-swe-agent`](swe-mini-swe-agent/README.md) | CPU | 203 |
| TerminalUse | Codex | [`terminal-use-codex`](terminal-use-codex/README.md) | CPU | 220 |
| BrowserUse | WebAgent | [`browser-use-webagent`](browser-use-webagent/README.md) | CPU | 83 |
| ComputerUse | GUIAgent | [`computer-use-gui-agent`](computer-use-gui-agent/README.md) | CPU + KVM (`/dev/kvm`) | 358 |
| ToolRichAssistant | Claude | [`tool-rich-assistant-claude`](tool-rich-assistant-claude/README.md) | CPU | 500 |
| OfficeWork | OpenClaw | [`office-work-openclaw`](office-work-openclaw/README.md) | CPU | 50 |
| AutoResearch | PiAutoResearch | [`auto-research-pi-autoresearch`](auto-research-pi-autoresearch/README.md) | 1 GPU | 22 |

**Hardware** assumes a remote LLM API (e.g., DashScope, DeepSeek, OpenAI).
HuggingGPT's GPUs serve its 7 local Hugging Face models; Pi-AutoResearch's GPU
trains models for MLE-bench competitions. If you self-host the LLM (e.g.,
vLLM), every setting needs GPU hardware for it.

Every setting defaults to **1 request**: a hello-world run that checks the
deployment works end to end. **Paper scale** is the request count behind the
baseline results in the [AgentSysBench paper](../../README.md#citation).

## Layout of a deployment

```
<scenario>-<agent>/
├── docker-compose.yaml   # agent service(s) + loadgen + monitoring
├── monitoring/           # Prometheus + Grafana configs
├── scripts/
│   ├── 1.prepare_data.sh # download the dataset, write the request manifest
│   ├── 2.run.sh          # docker compose up; loadgen sends the requests
│   ├── 3.cleanup.sh      # docker compose down; make outputs readable
│   └── 4.evaluate.sh     # task-level evaluation (where the scenario has one)
└── README.md             # dataset, prerequisites, configuration
```

The images come from `agents/<name>/scripts/build_image.sh` and
`request/scripts/build_image.sh`. Experiment-specific variables (request count,
concurrency, resource limits, ports) are set in the agent's `config/.env` or
overridden on the command line; each deployment README lists them.

## Experiment data

Outputs persist in `benchmark/data/<setting>/` after the containers exit.
Each `2.run.sh` first moves the previous run's data to
`benchmark/data/_archive/<setting>-<timestamp>/` so traces from different runs
are never mixed (set `AGSBENCH_KEEP_DATA=1` to append instead). `3.cleanup.sh`
makes the root-owned files readable on the host. Resource metrics (CPU /
memory per container) are in each deployment's Prometheus / Grafana while it
is running.

See [benchmark/analysis](../analysis/README.md) for the report and the
[raw data layout](../analysis/README.md#raw-data-layout) of each setting.

## Running at full scale

The loadgen sends `min(*_REQUEST_NUM, manifest lines)` requests; it stops at the end of the manifest and does not wrap. To run more requests:

1. Prepare a manifest at least as large as the target. Extra arguments to `scripts/1.prepare_data.sh` pass through to the scenario's `prepare_data.py`; add `--force` to regenerate an existing manifest.
2. Set the request count, e.g. `MINI_SWEAGENT_REQUEST_NUM=203 bash scripts/2.run.sh`.

| Deployment | Request count | Manifest size from `1.prepare_data.sh` |
|------------|---------------|-------------------------------|
| `qa-rag` | `RAG_REQUEST_NUM` | All of wiki_qa (3047); `--nrows N` to limit |
| `multimodal-hugginggpt` | `HUGGINGGPT_REQUEST_NUM` | `--num_samples N` (default 5) |
| `ai-search-deep-research` | `REQUEST_NUM` | Whole split of `--dataset {hle,gaia,frames}` |
| `swe-mini-swe-agent` | `MINI_SWEAGENT_REQUEST_NUM` | Whole `--split` of SWE-bench (test: 2294) |
| `terminal-use-codex` | `CODEX_REQUEST_NUM` | All TerminalBench `original-tasks` (241); narrow with `--tasks` / `--task_filter` |
| `browser-use-webagent` | `WEBAGENT_REQUEST_NUM` | 8 built-in tasks; `--tasks ID ...` for others |
| `computer-use-gui-agent` | `OS_WORLD_REQUEST_NUM` | `--num_samples N` (default 1), `--domain` to filter |
| `tool-rich-assistant-claude` | `CLAUDE_REQUEST_NUM` | All PinchBench tasks (147); `--task_filter` to filter |
| `office-work-openclaw` | `OPENCLAW_REQUEST_NUM` | All WildClawBench tasks (60); `--task_filter` to filter |
| `auto-research-pi-autoresearch` | `PI_AUTORESEARCH_REQUEST_NUM` | `--lite`: the 22 MLE-bench Lite competitions (~158 GB of Kaggle data); or `--competitions ID ...` |

Where the built-in dataset is smaller than the paper scale, extend the scenario's `prepare_data.py` (under `request/src/agsbench_request/<scenario>/`) with your own task source. Table 2 of the paper lists the datasets used for each setting.

To add a setting of your own, see
[Configure a new experiment setting](../../docs/development.md#1-configure-a-new-experiment-setting).
