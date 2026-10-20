# AgentSysBench

A system-level benchmark suite for LLM agent serving systems.

> **Methodology — *ceteris paribus*.** Agent system performance is shaped by many coupled factors: the workload, the agent, the model, and the serving stack. AgentSysBench isolates them: vary exactly one dimension of the serving system, hold everything else fixed, and attribute the change in system behavior to that dimension alone.

## Concept Model

```
┌─────────────────────────────────────────────────────────────────┐
│                          Workload                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐     │
│  │ Request  │  │  Tool    │  │  Model   │  │ Orchestrator │     │
│  │(scenario │  │ (search, │  │ (GPT-4o, │  │ (ReAct loop, │     │
│  │ dataset) │  │  shell,  │  │ DeepSeek,│  │  pipeline,   │     │
│  │          │  │  browse) │  │  Claude) │  │  DAG, …)     │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘     │
└────────────────────────────┬────────────────────────────────────┘
                             │ defines
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Serving System                             │
│  ┌──────────┐  ┌──────────────┐  ┌────────────────────┐         │
│  │ Hardware │  │  Component   │  │   Architecture     │         │
│  │(x86/ARM, │  │ (vLLM, Milvus│  │   (single-node,    │         │
│  │ GPU/CPU, │  │  Playwright, │  │    multi-node,     │         │
│  │  memory) │  │  fast-svc)   │  │    edge+cloud, …)  │         │
│  └──────────┘  └──────────────┘  └────────────────────┘         │
└────────────────────────────┬────────────────────────────────────┘
                             │ produces
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      System Behavior                            │
│  latency · per-op breakdown · throughput · resource usage       │
│  token consumption · cost estimation · cache hit rate · …       │
└─────────────────────────────────────────────────────────────────┘
```

## How It Works

An **experiment setting** pairs a **scenario** (request dataset + evaluation)
with an **agent** (tool + model + orchestrator) and deploys them on a serving
system with Docker Compose. Every setting runs the same pipeline:

```
 request/                         agents/<name>/                      external
┌──────────────────┐   HTTP     ┌──────────────────────────┐       ┌───────────────┐
│ prepare_data     │ scenario-  │ agent service            │──────►│ LLM API       │
│   → manifest     │ protocol   │ (fast-service: traces    │       │ search, models│
│ loadgen ─────────┼───────────►│  every LLM / tool call)  │──────►│ sandboxes, …  │
└──────────────────┘            └────────────┬─────────────┘       └───────────────┘
                                             │ trajectories + latency traces
                                             ▼
                              benchmark/data/<setting>/
                                             │
                                             ▼
                 sdk/analysis + benchmark/analysis: normalise → 4 metrics → report
```

Three design choices make settings comparable:

- **Decoupled workload and agent.** The loadgen and the agent both depend on
  `sdk/scenario-protocol`, never on each other. Any agent that speaks a
  scenario's request/result types can serve that scenario, so one dimension can
  change while the rest stays fixed.
- **Uniform instrumentation.** Agents are served through `sdk/fast-service`,
  which records end-to-end latency and the call tree of every request,
  alongside each agent's own trajectory.
- **Agent-agnostic analysis.** `sdk/analysis` maps every agent's raw output onto
  one schema (`requests`, `latency`, `io`, `tokens`, joined on `request_id`)
  and one taxonomy of operations (LLM, Sandbox, Search, Embedding, VecDB,
  Model, MCP, Other). The same four metrics come out for every setting:
  e2e latency, latency breakdown, IO size and token consumption.

| Package | Role |
|---------|------|
| [`sdk/scenario-protocol`](sdk/scenario-protocol/README.md) | Wire protocol between loadgen and agent: `ScenarioClient`, per-scenario request/result models |
| [`sdk/fast-service`](sdk/fast-service/README.md) | Serving framework: `@fast_service` HTTP/gRPC services, latency tracking, call trees, load generation |
| [`sdk/agent-commons`](sdk/agent-commons/README.md) | Shared agent utilities: LLM client, task loader, manifest format |
| [`sdk/analysis`](sdk/analysis/README.md) | Extraction and metric computation over the unified schema |

## Default Experiments

AgentSysBench ships with 10 default settings:

| Scenario | Agent | Deployment |
|----------|-------|------------|
| QA | RAG | [`qa-rag`](benchmark/deployment/qa-rag/README.md) |
| MultiModal | HuggingGPT | [`multimodal-hugginggpt`](benchmark/deployment/multimodal-hugginggpt/README.md) |
| AISearch | DeepResearch | [`ai-search-deep-research`](benchmark/deployment/ai-search-deep-research/README.md) |
| SWE | MiniSWE | [`swe-mini-swe-agent`](benchmark/deployment/swe-mini-swe-agent/README.md) |
| TerminalUse | Codex | [`terminal-use-codex`](benchmark/deployment/terminal-use-codex/README.md) |
| BrowserUse | WebAgent | [`browser-use-webagent`](benchmark/deployment/browser-use-webagent/README.md) |
| ComputerUse | GUIAgent | [`computer-use-gui-agent`](benchmark/deployment/computer-use-gui-agent/README.md) |
| ToolRichAssistant | Claude | [`tool-rich-assistant-claude`](benchmark/deployment/tool-rich-assistant-claude/README.md) |
| OfficeWork | OpenClaw | [`office-work-openclaw`](benchmark/deployment/office-work-openclaw/README.md) |
| AutoResearch | PiAutoResearch | [`auto-research-pi-autoresearch`](benchmark/deployment/auto-research-pi-autoresearch/README.md) |

Each setting defaults to a 1-request hello-world run. With a remote LLM API,
most settings need only a CPU machine; HuggingGPT, Pi-AutoResearch and
GUIAgent need GPUs or KVM. Hardware per setting, the request counts used in
the paper and how to run at full scale are in
[benchmark/deployment](benchmark/deployment/README.md).

## Quick Start

This walks through one setting, `swe-mini-swe-agent`, from setup to report.
The other settings follow the same steps; their READMEs list any extra
prerequisites.

### Prerequisites

- Docker 24+ and Docker Compose v2
- An OpenAI-compatible LLM API key (e.g., [DashScope](https://dashscope.aliyuncs.com), DeepSeek, OpenAI)

### 1. Setup

```bash
git clone https://github.com/open-agent-infra/agent-sys-bench.git
cd agent-sys-bench
bash scripts/setup.sh                          # writes .env with pip/npm mirrors
bash scripts/configure-api.sh                  # interactive: configure LLM + search keys
```

Non-interactive alternative:
```bash
bash scripts/configure-api.sh \
    --llm-preset dashscope --llm-key sk-xxx --llm-model deepseek-v4-pro \
    --search-provider firecrawl --search-key fc-xxx
```

See [docs/api-providers.md](docs/api-providers.md) for the supported LLM,
embedding and search providers.

> **China network**: `setup.sh` defaults to aliyun pip mirror and npmmirror.
> See [docs/china-mirrors.md](docs/china-mirrors.md) for Docker Hub, HuggingFace,
> and Playwright mirror configuration.

### 2. Build images

Build the agent image and the unified loadgen image:

```bash
(cd agents/mini-swe-agent && bash scripts/build_image.sh)
(cd request && bash scripts/build_image.sh)
```

### 3. Run the experiment

```bash
cd benchmark/deployment/swe-mini-swe-agent
bash scripts/1.prepare_data.sh    # download dataset + generate manifest
bash scripts/2.run.sh             # docker compose up; loadgen sends the requests
bash scripts/3.cleanup.sh         # docker compose down
bash scripts/4.evaluate.sh        # task-level evaluation (SWE-bench harness)
cd ../../..
```

Trajectories and latency traces are written to `benchmark/data/swe-mini-swe-agent/`.

### 4. Analyze results

```bash
pip install pandas numpy matplotlib
python benchmark/analysis/run.py swe-mini-swe-agent
```

Figures, CSV tables and `summary.txt` land in
`benchmark/analysis/out/swe-mini-swe-agent/`. `tables/` holds the normalised
per-request data for your own analyses. See
[benchmark/analysis](benchmark/analysis/README.md) for the metric
definitions, the raw data layout of each setting and reference outputs.

## Repository Structure

```
agent-sys-bench/
├── scripts/                    # setup.sh, configure-api.sh
├── sdk/                        # scenario-protocol, fast-service, agent-commons, analysis
├── request/                    # Workload: prepare_data + unified loadgen (agsbench-request image)
├── agents/<name>/              # Agent implementations: package, config, Dockerfile, build script
├── benchmark/
│   ├── deployment/             # Experiment settings, one per scenario × agent
│   ├── analysis/               # Report generation and reference outputs
│   └── data/                   # Experiment output (gitignored)
└── docs/                       # API providers, China mirrors, developer guide, limitations
```

## Extending AgentSysBench

See [docs/development.md](docs/development.md) for the developer guide covering:
- [Configure a new experiment setting](docs/development.md#1-configure-a-new-experiment-setting)
- [Add a new scenario](docs/development.md#2-add-a-new-scenario)
- [Add a new agent](docs/development.md#3-add-a-new-agent)

Known limitations and workarounds are in [docs/limitations.md](docs/limitations.md).

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Citation

The paper below reports baseline results for the default experiments at their standard scale. If you use AgentSysBench, please cite it:

```bibtex
@article{chang2026agentsysbench,
  title         = {From LLM Inference to Agentic Workloads: Characterization and Implications for Serving Systems},
  author        = {Chang, Chaokun and Zhou, Yukun and Fu, Kaihua and An, Dakai and Feng, Tianyu and Lu, Hanfeng and Yao, Sheng and Guo, Pu and Yu, Yinghao and Shan, Yizhou and Li, Bo and Yuan, Binhang and Wang, Wei},
  journal       = {arXiv preprint arXiv:2608.15127},
  year          = {2026},
  eprint        = {2608.15127},
  archivePrefix = {arXiv},
  primaryClass  = {cs.OS},
  url           = {https://arxiv.org/abs/2608.15127}
}
```
