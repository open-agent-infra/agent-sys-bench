"""Operation classification — two strategies, ported from
agent-benchmark-bin all-agent ``extract_time_breakdown.py``.

Strategy A (react agents): timestamp-delta attribution over trajectory
events. The time between event N and N+1 is charged to event N+1's role:
  - assistant            -> LLM
  - toolResult / tool    -> Sandbox (or Search if the tool name is a known
                            web-search tool)

Strategy B (pipeline agents): function-name substring matching over the
fast-service ``service_name``:
  - task_planning / model_selection / response_generation / llm_task /
    *_generation / generate_queries / judge  -> LLM
  - search_task / *_search                   -> Search
  - embed_documents / embed_query / embedding_task -> Embedding
  - vdb_store / vdb_search / vdb_task / split_docs / chunking / indexing -> VecDB
  - model_inference                          -> Model (local HF, non-API)
  - solve_task / solve_instance / solve_terminal_bench / *_e2e / naive_rag / *_workflow (top) -> e2e (top-level)
"""

from . import config as C

# ── Strategy A: react event classification ───────────────────────────
def react_event_category(role: str, tool_name: str = "") -> str:
    """Map a trajectory event's role (+ tool name) to a component category."""
    role = (role or "").lower()
    if role == "assistant":
        return "LLM"
    if role in ("tool", "toolresult", "tool_result"):
        if tool_name and tool_name in C.SEARCH_TOOLS:
            return "Search"
        return "Sandbox"
    # user / system / environment-step-without-tool
    if role in ("user", "system"):
        return "Other"
    return "Other"


def claude_tool_category(tool_name: str) -> str:
    """Claude Code tool -> category: mcp__* bridge tools are MCP, web
    tools are Search, built-ins (Bash/Read/Write/Edit/...) are Sandbox."""
    name = tool_name or ""
    if name in C.SEARCH_TOOLS:
        return "Search"
    if name.startswith("mcp__"):
        return "MCP"
    return "Sandbox"


# ── Strategy B: op classification (pipeline + react trace funcs) ─────
def pipeline_op_category(service_name: str) -> str:
    """Map a fast-service service_name (dotted) to a component category.
    Handles BOTH pipeline function names (llm_task/search_task/...) AND
    react-agent trace functions (model_query_call/env_execute_call/...).
    Falls back to 'Other'."""
    op = (service_name or "").lower()
    last = op.split(".")[-1]
    # top-level e2e entry?
    if any(k in last for k in ("solve_task", "solve_instance", "solve_terminal_bench", "naive_rag",
                                "_e2e", "e2e_", "deep_research_workflow",
                                "hugginggpt_e2e", "run_task")):
        return "e2e"
    # local HF model inference (hugginggpt)
    if "model_inference" in op:
        return "Model"
    # search
    if "search_task" in op or last.endswith("_search") or last == "search":
        return "Search"
    # embedding
    if "embed" in op or "embedding_task" in op:
        return "Embedding"
    # vecdb / indexing / chunking
    if any(k in op for k in ("vdb_", "vdb", "split_doc", "splitting_task",
                              "chunking", "indexing", "load_docs", "situate")):
        return "VecDB"
    # sandbox / env / tool execution (react agents)
    if any(k in op for k in ("env_execute", "env_step", "codex_exec",
                              "bash", "shell", "tool_call", "execute_call")):
        return "Sandbox"
    # LLM stages (pipeline + react)
    if any(k in op for k in ("task_planning", "model_selection", "response_generation",
                              "llm_task", "generation", "generate_queries",
                              "planner", "judge", "model_query_call", "llm_call",
                              "llm_pick", "normal_generation", "direct_generation")):
        return "LLM"
    return "Other"


# ── Normalization (fraction of e2e, with Other residual) ────────────
def normalize_breakdown(component_seconds: dict, total_e2e: float) -> dict:
    """Return {category: fraction}. Untracked residual (< 0.99 covered) -> Other.
    Ported from extract_time_breakdown._normalize."""
    if total_e2e <= 0:
        return {}
    result = {cat: t / total_e2e for cat, t in component_seconds.items() if t > 0}
    tracked = sum(result.values())
    if tracked < 0.99:
        result["Other"] = 1.0 - tracked
    return result
