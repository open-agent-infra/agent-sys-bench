# Developer Guide

This guide covers three common development tasks:
1. [Configure a new experiment setting](#1-configure-a-new-experiment-setting)
2. [Add a new scenario](#2-add-a-new-scenario)
3. [Add a new agent](#3-add-a-new-agent)

It also defines the project-wide [naming conventions](#4-naming-conventions).

---

## 1. Configure a New Experiment Setting

An "experiment setting" pairs an existing scenario with an existing agent. For
example, running the `swe` scenario with `codex` instead of `mini-swe-agent`.

### Steps

1. **Create the deployment directory:**

```bash
mkdir -p benchmark/deployment/<scenario>-<agent>/{monitoring,scripts}
```

2. **Create `docker-compose.yaml`:**

Use an existing deployment as a template (e.g., `benchmark/deployment/swe-mini-swe-agent/`).
Key sections to customize:

- `services`: agent service (image, volumes, env_file, healthcheck)
- Loadgen service: `image: agsbench-request:v1`, `--scenario <name>`, `--endpoint http://<agent>:<port>`
- Monitoring stack (prometheus + grafana + cadvisor)

3. **Copy monitoring configs:**

```bash
cp -r benchmark/deployment/swe-mini-swe-agent/monitoring/* benchmark/deployment/<scenario>-<agent>/monitoring/
```

Edit `monitoring/prometheus.yml` to adjust the service name filter if needed.

4. **Create deployment scripts:**

- `scripts/1.prepare_data.sh` — calls `agsbench_request.prepare_data --scenario <name>`
- `scripts/2.run.sh` — `docker compose up -d`
- `scripts/3.cleanup.sh` — `docker compose down`

5. **Create `.env`:**

Copy from the agent's `config/.env.example`, fill in API keys, set `AGSBENCH_HOME`.

6. **Create `README.md`** documenting prerequisites, quick start, and configuration.

### Important: Volume mount paths

Experiment output should land in `benchmark/data/<setting>/` (gitignored). Map:
- `trajectories/` — agent session/pred JSON
- `traces/` — fast-service latency CSVs + intermediate data
- `monitoring/prometheus/` — TSDB resource metrics
- `monitoring/grafana/` — dashboard state

The `.cache/` mounts (input data, manifests) should stay on `agents/<name>/.cache`.

### Image path convention

If the agent needs to pass file paths to another container (e.g., hugginggpt
coordinator → models_server), mount the files at a **shared path** like `/data/images`
in both containers. The manifest must reference this container-internal path.

---

## 2. Add a New Scenario

A scenario defines the **Request dimension** — the dataset format, request schema,
and success criteria. It lives in two places: the SDK wire protocol and the
request package.

### Step 1: Wire Protocol (SDK)

Create `sdk/scenario-protocol/src/scenario_protocol/scenarios/<name>.py`:

```python
"""<Scenario> wire protocol."""

from pydantic import BaseModel, Field
from scenario_protocol.interface import ScenarioRequest, ScenarioResult


class MyRequest(ScenarioRequest):
    """Request payload sent from loadgen to agent."""
    task_id: str = Field(description="Task identifier")
    input_text: str = Field(description="Task input")


class MyResult(ScenarioResult):
    """Response returned by the agent."""
    answer: str = ""
    score: float = 0.0


# The agent's fast-service RPC endpoint path (module.function)
RPC_NAME = "agsbench_<agent>.request.solve_task"
PARAM_NAME = "req"  # or "request", "item" — must match the agent's function signature
```

### Step 2: Loadgen (request package)

Create `request/src/agsbench_request/<name>/loadgen.py`:

```python
"""<Scenario> loadgen — scenario-specific logic only."""

from scenario_protocol.scenarios.<name> import (
    MyRequest, MyResult, RPC_NAME, PARAM_NAME,
)

Request = MyRequest
Result = MyResult

def extra_args(parser):
    parser.add_argument("--max_steps", default=10, type=int)

def line_to_request(line, args):
    # Parse a manifest line into a Request
    import json
    data = json.loads(line.strip())
    return MyRequest(task_id=data["id"], input_text=data["prompt"])

def is_success(result):
    return bool(result.answer)
```

### Step 3: Prepare Data (optional)

Create `request/src/agsbench_request/<name>/prepare_data.py`:

```python
import os

MANIFEST_NAME = "<name>_data.txt"

def extra_args(parser):
    parser.add_argument("--num_samples", type=int, default=5)

def run(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    # Download dataset, save files, generate manifest
    # Pin dataset version for reproducibility
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    with open(manifest_path, "w") as f:
        for item in dataset:
            f.write(f"{item['path']}\n")
    print(f"Wrote manifest to {manifest_path}")
```

### Step 4: Deployment

Create `benchmark/deployment/<name>-<agent>/` with compose + scripts + README
(see [Section 1](#1-configure-a-new-experiment-setting)).

### Checklist

- [ ] Wire protocol in SDK (`scenarios/<name>.py`)
- [ ] Loadgen in request package (`<name>/loadgen.py`)
- [ ] Prepare data (`<name>/prepare_data.py`, optional)
- [ ] Deployment directory (`benchmark/deployment/<name>-<agent>/`)
- [ ] At least one agent implements the scenario's RPC

---

## 3. Add a New Agent

An agent implements the **Tool + Model + Orchestrator** dimensions — it serves
one or more scenario RPCs via fast-service.

### Step 1: Agent package

```
agents/<name>/
├── src/
│   └── agsbench_<name>/
│       ├── __init__.py
│       ├── fsm.py              # FastServiceManager instance
│       ├── request.py         # Request/Result models + @fsm.fast_service RPC function
│       └── serving.py         # Scenario-protocol integration (SUPPORTED_SCENARIOS)
├── config/
│   ├── .env.example
│   └── <name>-server.yml      # fast-service server config
├── scripts/
│   └── build_image.sh
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

### Step 2: Implement the RPC

```python
# agents/<name>/src/agsbench_<name>/request.py

from pydantic import BaseModel
from fast_service import RequestContext
from .fsm import my_fsm

class SolveTaskRequest(BaseModel):
    task_id: str
    input_text: str

class SolveTaskResponse(BaseModel):
    answer: str
    score: float = 0.0

@my_fsm.fast_service
def solve_task(
    req: SolveTaskRequest,
    fsm_context: RequestContext = None,
) -> SolveTaskResponse:
    # Agent logic here
    return SolveTaskResponse(answer="...", score=1.0)
```

### Step 3: Serving interface

```python
# agents/<name>/src/agsbench_<name>/serving.py

from scenario_protocol.scenarios.<scenario_name> import (
    MyRequest, MyResult, RPC_NAME, PARAM_NAME,
)

AGENT_NAME = "<name>"
SUPPORTED_SCENARIOS = {
    "<scenario>": (RPC_NAME, PARAM_NAME, MyRequest, MyResult),
}
```

### Step 4: Dockerfile

```dockerfile
FROM python:3.12

ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

# fast-service + scenario-protocol via BuildKit
WORKDIR /workspace/fast-service
COPY --from=fast-service requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY --from=fast-service src src
COPY --from=fast-service pyproject.toml pyproject.toml
COPY --from=fast-service setup.py setup.py
RUN pip install -e .

WORKDIR /workspace/scenario-protocol
COPY --from=scenario-protocol pyproject.toml pyproject.toml
COPY --from=scenario-protocol requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY --from=scenario-protocol src src
RUN pip install -e .

# Agent package
WORKDIR /workspace/agsbench_<name>
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY src src
COPY pyproject.toml pyproject.toml
RUN pip install -e .
```

### Step 5: build_image.sh

```bash
#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
AGSBENCH_ENV="${AGENT_DIR}/../../.env"
[ -f "$AGSBENCH_ENV" ] && set -a && . "$AGSBENCH_ENV" && set +a
cd "$AGENT_DIR"
docker build \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple/}" \
  --build-context fast-service=../../sdk/fast-service \
  --build-context scenario-protocol=../../sdk/scenario-protocol \
  -t <name>:v1 .
```

### Step 6: Deployment

Create `benchmark/deployment/<scenario>-<name>/` with:
- `docker-compose.yaml` (agent service + agsbench-request loadgen + monitoring)
- `scripts/` (prepare_data, run, cleanup)
- `.env` (from agent's `.env.example`)
- `README.md`

### Checklist

- [ ] `src/agsbench_<name>/` package with `fsm.py`, `request.py`, `serving.py`
- [ ] `config/.env.example` + fast-service yml
- [ ] `Dockerfile` with `--build-context fast-service` + `scenario-protocol`
- [ ] `scripts/build_image.sh`
- [ ] `pyproject.toml` with `name = "agsbench-<name>"` (import name `agsbench_<name>`)
- [ ] `README.md` (Build + Serving Interface + Deployment sections)
- [ ] Deployment directory with compose + scripts + README
- [ ] RPC_NAME in wire protocol matches `agsbench_<name>.<module>.<function>`

---

## 4. Naming Conventions

The project name has three forms. Pick by **audience**, never by taste:

| Form | Audience | Use for |
|---|---|---|
| `AgentSysBench` | External: readers, papers, announcements | Brand and prose — README / docs titles and text, docstrings, banners printed by scripts, author fields (`AgentSysBench contributors`) |
| `agent-sys-bench` | Local / global directory management | Repository name and GitHub URL, the clone directory, shared host-side data directories, example paths in docs |
| `agsbench` / `AGSBENCH` / `AGSBench` | Developers: anything code reads | Prefix for Python packages and `src/` dirs, distribution names, Docker images, env vars, identifiers, container-internal paths, generated file names |

`agsbench` is short for **Ag**ent**S**ys**Bench**. It is the only abbreviation
of the project name. Do not introduce others in code, configs or docs.

The code prefix comes in several case styles, chosen by what the target syntax allows:

| Target | Style | Example |
|---|---|---|
| Python import package / `src/` directory | `agsbench_<name>` | `agents/rag/src/agsbench_rag/`, `import agsbench_rag` |
| Distribution name in `pyproject.toml` | `agsbench-<name>` | `name = "agsbench-rag"`, `name = "agsbench-mini-swe-agent"` |
| Module paths in fast-service / RPC configs | `agsbench_<name>.<module>.<function>` | `agsbench_openclaw.run_task.run_task` |
| Identifiers in code | `agsbench_<...>` | `agsbench_rag_fsm` |
| Python class names | `AGSBench<...>` (acronym upper-cased, PEP 8) | `AGSBenchClient` |
| Environment variables | `AGSBENCH_<NAME>` | `AGSBENCH_HOME`, `AGSBENCH_ENV` |
| Docker images | `agsbench-<name>:<tag>` or `agsbench/<name>:<tag>` | `agsbench-request:v1`, `agsbench/vllm-log:v0.9.0.1` |
| Config files named after a package | `agsbench_<name>-<role>.yml` | `agsbench_mini_swe_agent-server.yml` |

**Host directory vs. container directory.** The two look alike but follow
different rules:

- A **shared host-side directory** is a global directory, so it uses
  `agent-sys-bench`. Example: the OSWorld VM image cache
  `OS_WORLD_DATA_DIR` defaults to `/data/agent-sys-bench`, and the repository
  is cloned to `/path/to/agent-sys-bench`.
- A **container-internal working directory** mirrors the Python package, so it
  uses `agsbench_<name>`. Example: the gui-agent container works in
  `/workspace/agsbench_gui_agent`, and mini-swe-agent writes to
  `/workspace/agsbench_mini_swe_agent/trajectories`.

Upstream names stay as they are. For example, `mini-swe-agent`, `openclaw` and
the `cais/hle` dataset are not renamed. The prefix is what keeps
our wrappers from colliding with them.

---

## Key Design Principles

1. **Loadgen and agent never import each other** — they only share the
   `scenario-protocol` SDK (wire protocol types + `ScenarioClient`).

2. **All configuration is in deployment directories** — agent directories
   contain only source code + build scripts + fast-service yml templates.

3. **Experiment data centralized** — all output lands in
   `benchmark/data/<setting>/` (trajectories, traces, monitoring).

4. **Dataset versions pinned** — every `prepare_data.py` pins a specific
   git commit or HF revision for reproducibility.

5. **`agsbench_` prefix** — all agent Python packages use `agsbench_<name>`
   to avoid collisions with upstream packages (see
   [Naming Conventions](#4-naming-conventions)).
