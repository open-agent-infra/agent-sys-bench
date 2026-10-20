# AgentSysBench Analysis SDK

Data extraction + metric computation layer that aligns each agent's
heterogeneous raw output to a unified record schema, so the 4 standard
metrics can be computed agent-agnostically.

**Scope**: data preparation + computation only. Presentation lives in
`benchmark/analysis/`.

## Quick map

```
sdk/analysis/
├── config.py          8-category taxonomy + per-agent layout + setting↔agent map
├── schema.py          AnalysisData (4 pandas DataFrames, fixed columns)
├── task_registry.py   per-agent request-validity filtering (≥2 assistant turns)
├── classify.py        two classification strategies (timestamp-delta / function-substring)
├── metrics.py         4 metrics (latency / breakdown / IO / tokens), agent-agnostic
├── extract.py         per-agent extractors: raw files → AnalysisData
├── _parse.py          low-level CSV/JSONL/timestamp parsers
└── _kmeans.py         numpy-only k-means for breakdown example selection
```

## 1. Data distribution format (per agent)

All data lands under `benchmark/data/<setting>/`. Two production endpoints:

| endpoint | what it writes | format |
|---|---|---|
| **agent service** (fast-service FSM) | trajectory files + fast-service trace (e2e_delay.csv, intermediate JSONL) | JSON / JSONL / CSV |
| **loadgen** (agsbench-request) | nothing (output_file not set — Result folded to bool) | — |

### Per-agent layout (after the trace-persistence fix)

| agent | type | trajectory | trace (fast-service) | token source |
|---|---|---|---|---|
| **swe** | react | `trajectories/<id>/<id>.traj.json` (messages with `extra.response.usage`) | `traces/mini-swe-agent/server/e2e_delay.csv` + `intermediate/server/requests/<func>/` | `traj.json` messages[].extra.response.usage (per-call) |
| **codex** | react | `trajectories/<id>/<id>.traj.json` + `<id>.codex_stdout.jsonl` (turn.completed usage) | `traces/codex/server/e2e_delay.csv` + `intermediate/` | `codex_stdout.jsonl` turn.completed.usage |
| **webagent** | react | `trajectories/<id>/<id>.traj.json` + `<id>.session.jsonl` (llm_call.usage) | `monitoring/server/e2e_delay.csv` + `intermediate/` | `session.jsonl` llm_call.usage |
| **openclaw** | react | `output/<task>/<run>/chat.jsonl` (message.usage; exported by `run_task` from openclaw's SQLite `transcript_events`) | **no fast-service trace** (FSM RequestContext=None) | `chat.jsonl` message.usage |
| **pi** | react | `trajectories/<id>/<id>.traj.json` + `pi-sessions/*.jsonl` (message.usage) | **no fast-service trace** (monitor_only) | `pi-sessions` message.usage |
| **claude** | react | `runs/latest/tasks/<task>/<run>/agent.stream.jsonl` (result.usage aggregate) | `runs/latest/fast-service-monitor/server/e2e_delay.csv` | `stream.jsonl` result.usage (aggregate) |
| **rag** | pipeline | **no trajectory** (answer in intermediate) | `runs/results/server/monitor/rag_server/e2e_delay.csv` + `intermediate/requests/<20+func>/` | **N/A** (generation fn doesn't return usage) |
| **ai-search** | pipeline | **no trajectory** (final_report in intermediate) | `runs/results/server/monitor/server/e2e_delay.csv` + `intermediate/` | **N/A** (llm_task fn doesn't return usage) |
| **hugginggpt** | pipeline | **no trajectory** | `monitoring/coordinator/hugginggpt_server/e2e_delay.csv` + `traces/requests/<func>/` | **N/A** (LLM stages don't return usage; model_inference is local HF) |

> **webagent**: blocked on environment (MiniWoB download). Extractor code is ready but unverified.
> **gui-agent**: tasks run in a spawned subprocess without a RequestContext, so the trace has only `solve_task` rows; the breakdown comes from `session.jsonl` (`llm_call.duration_s` → LLM, llm_call→env_step → Sandbox), with VM boot + evaluation as the Other residual.

### Fast-service trace format (shared, from `sdk/fast-service`)

**`e2e_delay.csv` / `remote_delay.csv`** — no header, 7 positional columns:
```
end_time, request_id, parent_service_name, service_name, delay, call_id, parent_call_id
```
(declared header is 5 cols but `to_tuple()` returns 7 — known sdk bug, see `docs/data-format-survey.md` §3-A)

**`stats/e2e_delay.csv`** — with header (auto-generated on close):
```
service_name, throughput, avg_delay, std_delay, max_delay, min_delay,
25%, 50%, 75%, 90%, 95%, 99%, start_ts, end_ts, total_time, total_records
```

**`intermediate/requests/<function>/{requests,responses}.txt`** — JSONL:
```json
{"request_id": "...", "content": <json>}
```
`content` for BaseModel returns = `model_dump()`; for dict/list/None = `str()`-wrapped (known sdk inconsistency, §3-D).

## 2. Unified record schema (`schema.py`)

Each extractor returns `AnalysisData` — 4 DataFrames with fixed columns:

| DataFrame | columns |
|---|---|
| `latency` | request_id, call_id, parent_call_id, function_name, **category**, start_time, end_time, delay, delay_type, is_top_level |
| `io` | request_id, call_id, function_name, **category**, direction(input/output), bytes, tokens |
| `tokens` | request_id, call_id, function_name, **category**, model, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens, total_tokens, cost_usd |
| `requests` | request_id, e2e_latency, success, n_steps, n_llm_calls, total_input_tokens, total_output_tokens, total_cost_usd, error |

The `category` column is the cross-agent classification axis (filled by `classify.py`).

## 3. Request-validity filtering (`task_registry.py`)

Ported from `agent-benchmark-bin` all-agent branch. **Single source of truth** —
all extractors call `get_valid_tasks(agent, data_dir)` first.

**Common rule**: a request is valid only if the agent did multi-step work
(≥ 2 assistant turns), not a single errored response. Per-agent specifics:

| agent | validity criteria |
|---|---|
| swe | traj.json has `messages` + ≥2 assistant messages (with `extra`) |
| codex | `codex_stdout.jsonl` has ≥1 `turn.completed` event |
| webagent | `session.jsonl` `session_end.n_steps` ≥ 2 |
| gui-agent | `session.jsonl` ≥2 `llm_call` events |
| claude | `agent.stream.jsonl` ≥2 `assistant` events |
| openclaw | `chat.jsonl` ≥2 assistant messages + `run_summary.wall_s` > 0 |
| pi | `pi-sessions/*.jsonl` ≥2 assistant messages |
| rag / ai-search / hugginggpt | top-level `e2e_delay.csv` row with delay > 0 (pipeline) |

Also computes `e2e_s` per task (from trajectory timestamps or trace top-level).

## 4. Classification (`classify.py`)

8 categories (ported from `agent-benchmark-bin` `config.COMPONENT_COLORS`):
`LLM, Sandbox, Search, Embedding, VecDB, Model, MCP, Other`.
Claude Code tools: `mcp__*` → MCP, `WebSearch`/`WebFetch` → Search, built-ins
(Bash/Read/Write/Edit/...) → Sandbox.

**Two strategies** (chosen per-agent by `config.AGENT_META[agent]["type"]`):

- **Strategy A — timestamp-delta** (react agents): the time between
  event N and N+1 is charged to event N+1's role:
  `assistant → LLM`, `toolResult → Sandbox` (or `Search` if tool name ∈
  `SEARCH_TOOLS`). Used when the fast-service trace is thin or absent
  (openclaw, pi, claude, webagent).
- **Strategy B — function-substring** (pipeline agents + react trace):
  `service_name` matched against substrings:
  `model_query_call/llm_task/generation → LLM`, `env_execute/codex_exec → Sandbox`,
  `search_task → Search`, `embed → Embedding`, `vdb_/split_doc → VecDB`,
  `model_inference → Model`, `solve_instance/*_e2e → e2e` (top-level).

**Residual**: for react agents, untracked time = `LLM` (the LLM call is
external, not in the trace). For pipeline agents, residual = `Other`.

## 5. Extraction (`extract.py`)

Each agent has an `@_register("agent")` extractor implementing:
1. `task_registry.get_valid_tasks()` → valid task ids + paths.
2. Build `latency` (from trace or trajectory timestamps).
3. Build `io` (from intermediate JSONL content bytes, or trajectory message bytes).
4. Build `tokens` (from each agent's usage source — see table in §1).
5. Build `requests` (e2e_latency per valid task).

**Token extraction** handles multiple usage field shapes:
- OpenAI: `prompt_tokens` / `completion_tokens` / `prompt_tokens_details.cached_tokens`
- Codex: `input_tokens` / `output_tokens` / `cached_input_tokens` / `reasoning_output_tokens`
- openclaw/pi (pi-ai): bare `input` / `output` / `cacheRead` / `reasoning`

`input_tokens` always **includes** cache reads (OpenAI semantics), so
`cache_rate = cache_read / input` is in [0, 1]. Usage shapes that exclude
them (Anthropic `input_tokens`, pi-ai `input`) are summed with the cache
reads on extraction.

**Request ids**: all four tables are keyed by the task's `request_id`. When the
fast-service trace uses loadgen UUIDs instead (codex, swe, claude), trace ids are
rewritten to task ids — exactly via the intermediate request payload when it
carries the task id, otherwise by pairing each task with the top-level trace row
whose delay matches its e2e (within 5% / 1 s). `_fill_e2e_from_trace()` fills
`e2e_latency` from the top-level trace delay when the trajectory has none, with
an order-based fallback only for tasks that could not be aligned.

**success / error**: whether the agent run finished cleanly (`run_summary`
exit_code / timed_out for claude/openclaw, traj `info.exit_status` for
swe/codex). It does not say whether the task was solved — use the
deployment's evaluate script for that.

## 6. Metrics (`metrics.py`) — agent-agnostic

| metric | computation |
|---|---|
| **e2e latency** | mean / std / p50 / p90 / p95 / p99 / histogram (from `requests.e2e_latency`) |
| **breakdown** | per-category delay as fraction of e2e mean (sum-of-means); k-means cluster selection of representative per-request breakdowns |
| **IO size** | per-category input/output byte distribution (mean / p95) from `io` |
| **token breakdown** | per-category token sums (input/output/cache_read/reasoning) + cache_rate from `tokens` |

## 7. Verified status

See `benchmark/analysis/README.md` (Verified output) for the per-setting
numbers; the reference reports are in `benchmark/analysis/examples/`.
