"""Shared constants for the AgentSysBench analysis SDK.

Operation/component taxonomy (8 categories, ported from agent-benchmark-bin
all-agent branch ``config.COMPONENT_COLORS``):
  LLM, Sandbox, Search, Embedding, VecDB, Model, MCP, Other

  - LLM        : LLM API calls (planning / generation / selection / judge)
  - Sandbox    : tool / environment execution (bash, file ops, GUI action)
  - Search     : web-search tool calls (web_fetch, web_search, firecrawl)
  - Embedding  : vector embedding
  - VecDB      : vector DB store / search / chunking / indexing
  - Model      : local HF model inference (non-API, e.g. HuggingGPT stages)
  - MCP        : claude mcp__* tool calls (built-in Bash/Read/... are Sandbox)
  - Other      : untracked / residual
"""

import os

# ── Categories ───────────────────────────────────────────────────────
CATEGORIES = ["LLM", "Sandbox", "Search", "Embedding", "VecDB", "Model", "MCP", "Other"]

CATEGORY_COLORS = {
    "LLM":        "#4C72B0",
    "Sandbox":    "#55A868",
    "Search":     "#C44E52",
    "Embedding":  "#8172B2",
    "VecDB":      "#CCB974",
    "Model":      "#64B5CD",
    "MCP":        "#DD8452",
    "Other":      "#AAAAAA",
}

# Agent "type" decides the classification strategy:
#   react    -> timestamp-delta attribution over trajectory events
#   pipeline -> function-name substring attribution over fast-service trace
AGENT_TYPE_REACT = "react"
AGENT_TYPE_PIPELINE = "pipeline"

# ── Agent metadata ───────────────────────────────────────────────────
# key = canonical agent id; matches agent-benchmark-bin naming where possible.
AGENT_META = {
    "mini_sweagent":   {"label": "SWE-agent",     "type": AGENT_TYPE_REACT,    "setting": "swe-mini-swe-agent"},
    "codex":           {"label": "Codex",         "type": AGENT_TYPE_REACT,    "setting": "terminal-use-codex"},
    "webagent":        {"label": "WebAgent",      "type": AGENT_TYPE_REACT,    "setting": "browser-use-webagent"},
    # LLM time is tracked per call; the untracked residual is VM boot + evaluation.
    "guiagent":        {"label": "GUIAgent",      "type": AGENT_TYPE_REACT,    "setting": "computer-use-gui-agent",
                        "breakdown_residual": "Other"},
    "claude":          {"label": "Claude",        "type": AGENT_TYPE_REACT,    "setting": "tool-rich-assistant-claude"},
    "openclaw":        {"label": "OpenClaw",      "type": AGENT_TYPE_REACT,    "setting": "office-work-openclaw"},
    "pi_autoresearch": {"label": "Pi-AutoResearch","type": AGENT_TYPE_REACT,   "setting": "auto-research-pi-autoresearch"},
    "rag":             {"label": "RAG",           "type": AGENT_TYPE_PIPELINE, "setting": "qa-rag"},
    "deepresearch":    {"label": "DeepResearch",  "type": AGENT_TYPE_PIPELINE, "setting": "ai-search-deep-research"},
    "hugginggpt":      {"label": "HuggingGPT",    "type": AGENT_TYPE_PIPELINE, "setting": "multimodal-hugginggpt"},
}

AGENTS = list(AGENT_META.keys())

# setting name -> agent id
SETTING_AGENT = {m["setting"]: a for a, m in AGENT_META.items()}

# ── Per-agent data-layout descriptors ────────────────────────────────
# Each layout tells the extractor where to find things relative to
# benchmark/data/<setting>/. None = not applicable / unknown.
#
#   traj_dir      : directory holding per-task trajectory files
#   traj_glob     : glob (relative to traj_dir) for the main trajectory file
#   session_glob  : glob for the per-task session jsonl (react agents)
#   preds         : path to preds.json (if any)
#   e2e_csv       : glob for fast-service e2e_delay.csv
#   intermediate  : glob for the fast-service intermediate "requests" dir
#   summary       : glob for run_summary.json (claude/openclaw)
AGENT_LAYOUT = {
    "mini_sweagent": {
        "traj_dir": "trajectories", "traj_glob": "*/*.traj.json",
        "preds": "trajectories/preds.json",
        "e2e_csv": "traces/mini-swe-agent/server/e2e_delay.csv",
        "intermediate": "traces/mini-swe-agent/intermediate/server/requests",
    },
    "codex": {
        "traj_dir": "trajectories", "traj_glob": "*/*.traj.json",
        "session_glob": "*/*.codex_stdout.jsonl",
        "e2e_csv": "traces/codex/server/e2e_delay.csv",
        "intermediate": "traces/codex/intermediate/server/requests",
    },
    "webagent": {
        "traj_dir": "trajectories", "traj_glob": "*/*.traj.json",
        "session_glob": "*/*.session.jsonl",
        "preds": "trajectories/preds.json",
        "e2e_csv": "monitoring/server/e2e_delay.csv",
        "intermediate": "monitoring/intermediate/server/requests",
    },
    "guiagent": {
        "traj_dir": "trajectories", "traj_glob": "*/*.traj.json",
        "session_glob": "*/*.session.jsonl",
        "preds": "trajectories/preds.json",
        "e2e_csv": "traces/monitoring/os-world-service/e2e_delay.csv",
        "intermediate": "traces/intermediate/os-world-service/requests",
    },
    "claude": {
        "traj_dir": "runs/latest/tasks", "traj_glob": "*/*/agent.stream.jsonl",
        "summary_glob": "*/*/run_summary.json",
        "e2e_csv": "runs/latest/fast-service-monitor/server/e2e_delay.csv",
    },
    "openclaw": {
        "traj_dir": "output", "session_glob": "*/*/chat.jsonl",
        "summary_glob": "*/*/run_summary.json",
    },
    "pi_autoresearch": {
        "traj_dir": "trajectories", "traj_glob": "*/*.traj.json",
        "session_glob": "*/pi-sessions/*.jsonl",
        "preds": "trajectories/preds.json",
    },
    "rag": {
        "e2e_csv": "runs/results/server/monitor/rag_server/e2e_delay.csv",
        "intermediate": "runs/results/server/intermediate/requests",
    },
    "deepresearch": {
        "e2e_csv": "runs/results/server/monitor/server/e2e_delay.csv",
        "intermediate": "runs/results/server/intermediate/requests",
    },
    "hugginggpt": {
        "e2e_csv": "monitoring/coordinator/hugginggpt_server/e2e_delay.csv",
        "intermediate": "traces/requests",
        "models_log": "monitoring/models_server/hugginggpt_models_server/model_server_execution.log",
    },
}

# ── Tool-name -> category overrides (react agents) ───────────────────
# Tools in these sets are classified as "Search" instead of "Sandbox".
SEARCH_TOOLS = {
    "web_fetch", "web_search", "firecrawl_search", "firecrawl_scrape",
    "search", "tavily_search", "duckduckgo",
    "WebSearch", "WebFetch",  # claude code built-ins
}

# ── Layout helper ────────────────────────────────────────────────────
def data_dir(setting):
    """Return the benchmark/data/<setting> path for a setting name."""
    return os.path.join(_BENCH_DATA_ROOT, setting)


# Root of benchmark/data. Resolved lazily so the package is relocatable.
# Default: <repo>/benchmark/data (three levels up from this file).
_BENCH_DATA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "benchmark", "data")
)
