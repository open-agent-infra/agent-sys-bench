"""Per-agent extractors: raw files -> AnalysisData (unified records).

Each extractor:
  1. calls task_registry for valid tasks
  2. builds LatencyRecord (from fast-service trace, or trajectory timestamps)
  3. builds IORecord (from fast-service intermediate requests/responses)
  4. builds TokenRecord (from each agent's token source — the most heterogeneous)
  5. builds RequestSummary (e2e_latency per valid request)

All extractors return AnalysisData (never raise); shortfalls go to
.diagnostics so the runner can surface them as data problems.
"""

import glob
import json
import os
from typing import List

import pandas as pd

from . import config as C
from ._parse import (content_bytes, parse_e2e_csv, parse_intermediate,
                     parse_iso_ts, read_json, read_jsonl)
from .classify import claude_tool_category, pipeline_op_category, react_event_category
from .schema import AnalysisData


# ── Shared: latency from fast-service trace ──────────────────────────
def _latency_from_trace(data_dir: str, layout: dict, diags: List[str]) -> pd.DataFrame:
    """Build LatencyRecord from e2e_delay.csv (all rows, classified)."""
    e2e_csv = os.path.join(data_dir, layout["e2e_csv"]) if layout.get("e2e_csv") else None
    rows = parse_e2e_csv(e2e_csv) if e2e_csv else []
    if not rows:
        diags.append(f"no e2e_delay rows at {e2e_csv}")
        return pd.DataFrame()
    recs = []
    for r in rows:
        cat = pipeline_op_category(r["service_name"])
        is_top = (r["parent_service_name"] in ("None", "", "None,None"))
        recs.append({
            "request_id": r["request_id"],
            "call_id": r["call_id"],
            "parent_call_id": r["parent_call_id"],
            "function_name": r["service_name"],
            "category": "e2e" if is_top else cat,
            "start_time": r["end_time"] - r["delay"],
            "end_time": r["end_time"],
            "delay": r["delay"],
            "delay_type": "e2e",
            "is_top_level": is_top,
        })
    df = pd.DataFrame(recs)
    n_sub = (~df["is_top_level"]).sum()
    diags.append(f"trace rows={len(df)} top={int(df['is_top_level'].sum())} sub={int(n_sub)}")
    if n_sub == 0:
        diags.append("trace has only top-level rows -> breakdown from trace unavailable")
    return df


# ── Shared: IO from fast-service intermediate ────────────────────────
def _io_from_intermediate(data_dir: str, layout: dict, diags: List[str]) -> pd.DataFrame:
    req_dir = os.path.join(data_dir, layout["intermediate"]) if layout.get("intermediate") else None
    inter = parse_intermediate(req_dir)
    if not inter:
        diags.append(f"no intermediate at {req_dir}")
        return pd.DataFrame()
    recs = []
    for fn, entry in inter.items():
        cat = pipeline_op_category(fn)
        # pair requests/responses by request_id
        resp_map = {r.get("request_id"): r for r in entry["responses"]}
        for req in entry["requests"]:
            rid = req.get("request_id", "")
            # Use the RAW content (before _maybe_parse_content) for byte
            # measurement — the raw content is what was actually sent/received
            # over the wire. json.dumps serializes any type (str/dict/list)
            # to a consistent byte count.
            raw_req = req.get("content")
            in_bytes = len(json.dumps(raw_req, default=str).encode("utf-8")) if raw_req is not None else 0
            out_bytes = 0
            r = resp_map.get(rid)
            if r:
                raw_resp = r.get("content")
                out_bytes = len(json.dumps(raw_resp, default=str).encode("utf-8")) if raw_resp is not None else 0
            recs.append({"request_id": rid, "call_id": "", "function_name": fn,
                         "category": cat, "direction": "input", "bytes": in_bytes, "tokens": 0})
            recs.append({"request_id": rid, "call_id": "", "function_name": fn,
                         "category": cat, "direction": "output", "bytes": out_bytes, "tokens": 0})
    diags.append(f"intermediate functions={len(inter)} io_rows={len(recs)}")
    return pd.DataFrame(recs)


# ── Shared: request summary from valid tasks ─────────────────────────
def _requests_from_tasks(task_data: dict) -> pd.DataFrame:
    recs = []
    for tid, td in task_data.items():
        rid = td.get("request_id", tid)
        recs.append({
            "request_id": rid, "e2e_latency": td.get("e2e_s", 0.0),
            "success": td.get("success", True), "n_steps": td.get("n_steps", 0),
            "n_llm_calls": td.get("n_llm_calls", 0),
            "total_input_tokens": 0, "total_output_tokens": 0,
            "total_cost_usd": td.get("cost", 0.0), "error": td.get("error"),
        })
    return pd.DataFrame(recs)


def _find_id(content, ids: set):
    """First task id quoted inside ``content`` (dict or a possibly-truncated
    JSON string, as the intermediate recorder stores both)."""
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    for i in ids:
        if f'"{i}"' in text:
            return i
    return None


def _rewrite_ids(data: AnalysisData, mapping: dict):
    for df in (data.latency, data.io):
        if not df.empty and "request_id" in df:
            df["request_id"] = df["request_id"].map(lambda x: mapping.get(x, x))


def _top_ids_match(data: AnalysisData) -> bool:
    top = data.latency[data.latency["is_top_level"] == True]  # noqa: E712
    return bool(set(top["request_id"]) & set(data.requests["request_id"]))


def _align_via_intermediate(data: AnalysisData, data_dir: str):
    """Exact trace-id -> task-id mapping from the intermediate request payloads
    (e.g. codex/swe: the loadgen UUID's request body carries the task/instance id)."""
    if data.requests.empty or data.latency.empty or "is_top_level" not in data.latency:
        return
    if _top_ids_match(data):
        return
    rel = C.AGENT_LAYOUT.get(data.agent, {}).get("intermediate")
    inter = parse_intermediate(os.path.join(data_dir, rel)) if rel else {}
    ids = set(data.requests["request_id"])
    mapping = {}
    for entry in inter.values():
        for req in entry["requests"]:
            trid = req.get("request_id")
            if trid and trid not in mapping:
                hit = _find_id(req.get("content"), ids)
                if hit:
                    mapping[trid] = hit
    if mapping:
        _rewrite_ids(data, mapping)
        data.diagnostics.append(f"aligned trace ids to tasks via intermediate payload ({len(mapping)})")


def _fill_e2e_from_trace(data: AnalysisData, data_dir: str = ""):
    """If request e2e_latency is 0/unset, fill from the top-level trace rows.
    Tries request_id match first; falls back to order-based assignment
    (trace top-level rows <-> tasks) when ids don't line up (e.g. codex
    where the task is keyed by task id but the trace by a loadgen UUID)."""
    if data.requests.empty or data.latency.empty:
        return
    if data_dir:
        _align_via_intermediate(data, data_dir)
    top = data.latency[data.latency.get("is_top_level", False)].sort_values("start_time")
    if top.empty:
        return
    by_rid = dict(zip(top["request_id"], top["delay"]))
    unmatched_idx = []
    for i, row in data.requests.iterrows():
        if (row.get("e2e_latency") or 0) <= 0 and row["request_id"] in by_rid:
            data.requests.at[i, "e2e_latency"] = by_rid[row["request_id"]]
        elif (row.get("e2e_latency") or 0) <= 0:
            unmatched_idx.append(i)
    if unmatched_idx:
        # order-based fallback: zip unmatched requests with sorted top-level delays
        delays = list(top["delay"])
        for k, i in enumerate(unmatched_idx):
            if k < len(delays):
                data.requests.at[i, "e2e_latency"] = delays[k]
        data.diagnostics.append(f"filled e2e_latency from trace top-level (order-match, {len(unmatched_idx)})")
    else:
        data.diagnostics.append("filled e2e_latency from trace top-level")


# ── Token extraction helpers ─────────────────────────────────────────
def _token_row(rid, fn, cat, model, usage: dict):
    """Build a TokenRecord row from an OpenAI-style usage dict."""
    if not usage:
        return None
    pdetails = usage.get("prompt_tokens_details") or {}
    cdetails = usage.get("completion_tokens_details") or {}
    # LangChain usage_metadata style
    idetails = usage.get("input_token_details") or {}
    odetails = usage.get("output_token_details") or {}
    # pi-ai style ({input, cacheRead}) excludes cached tokens from `input`;
    # normalise to OpenAI semantics where input includes cache reads.
    pi_extra = int(usage.get("cacheRead") or 0) if "input" in usage and "prompt_tokens" not in usage else 0
    return {
        "request_id": rid, "call_id": "", "function_name": fn, "category": cat,
        "model": model or "",
        "input_tokens": int(usage.get("input_tokens") or usage.get("prompt_tokens") or usage.get("input") or 0) + pi_extra,
        "output_tokens": int(usage.get("output_tokens") or usage.get("completion_tokens") or usage.get("output") or 0),
        "cache_read_tokens": int(usage.get("cache_read_tokens")
                                 or usage.get("cached_tokens")
                                 or usage.get("cacheRead")
                                 or pdetails.get("cached_tokens")
                                 or usage.get("prompt_cache_hit_tokens")
                                 or usage.get("cached_input_tokens")
                                 or idetails.get("cache_read")
                                 or 0),
        "cache_write_tokens": int(usage.get("cache_write_tokens")
                                   or usage.get("cacheWrite")
                                   or pdetails.get("cache_creation_tokens")
                                   or usage.get("prompt_cache_miss_tokens")
                                   or 0),
        "reasoning_tokens": int(usage.get("reasoning_tokens")
                                or usage.get("reasoning_output_tokens")
                                or cdetails.get("reasoning_tokens")
                                or odetails.get("reasoning")
                                or 0),
        "total_tokens": int(usage.get("total_tokens") or
                            usage.get("totalTokens") or 0),
        "cost_usd": float(usage.get("cost", {}).get("total") if isinstance(usage.get("cost"), dict)
                          else usage.get("cost", 0.0) or 0.0),
    }


# ════════════════════════════════════════════════════════════════════
# Per-agent extractors
# ════════════════════════════════════════════════════════════════════
_LOADERS = {}


def _register(agent):
    def deco(fn):
        _LOADERS[agent] = fn
        return fn
    return deco


def extract(agent: str, data_dir: str) -> AnalysisData:
    """Run the extractor for ``agent`` against ``data_dir``. Never raises."""
    data = AnalysisData(agent=agent)
    fn = _LOADERS.get(agent)
    if fn is None:
        data.diagnostics.append(f"no extractor for agent '{agent}'")
        return data
    if not os.path.isdir(data_dir):
        data.diagnostics.append(f"data dir missing: {data_dir}")
        return data
    try:
        fn(data, data_dir)
        _align_via_intermediate(data, data_dir)
        _align_trace_ids(data)
        _fill_request_totals(data)
    except Exception as e:  # surface as a data problem, don't crash the runner
        data.diagnostics.append(f"extractor error: {type(e).__name__}: {e}")
    return data


def _align_trace_ids(data: AnalysisData):
    """Rewrite trace request_ids (loadgen UUIDs) to the task request_id so all
    four tables join on one key. Only runs when no top-level trace id matches
    a task id; pairs each task with the top-level trace row whose delay is
    closest to the task's e2e latency (within 5% or 1s)."""
    if data.requests.empty or data.latency.empty or "is_top_level" not in data.latency:
        return
    top = data.latency[data.latency["is_top_level"] == True]  # noqa: E712
    if top.empty or _top_ids_match(data):
        return
    pairs = []
    for _, r in data.requests.iterrows():
        e2e = float(r.get("e2e_latency") or 0)
        if e2e <= 0:
            continue
        for trid, d in zip(top["request_id"], top["delay"]):
            pairs.append((abs(float(d) - e2e), r["request_id"], trid, e2e))
    mapping, used_req = {}, set()
    for diff, rid, trid, e2e in sorted(pairs, key=lambda x: x[0]):
        if rid in used_req or trid in mapping or diff > max(1.0, 0.05 * e2e):
            continue
        mapping[trid] = rid
        used_req.add(rid)
    if not mapping:
        data.diagnostics.append("trace ids not aligned to tasks (no e2e match)")
        return
    _rewrite_ids(data, mapping)
    data.diagnostics.append(f"aligned trace ids to tasks by e2e ({len(mapping)}/{len(data.requests)})")


def _fill_request_totals(data: AnalysisData):
    """Fill per-request token totals / LLM-call counts from the tokens table
    when the extractor left them at 0."""
    if data.requests.empty or data.tokens.empty:
        return
    llm = data.tokens[data.tokens["category"] == "LLM"]
    g = llm.groupby("request_id").agg(i=("input_tokens", "sum"), o=("output_tokens", "sum"),
                                      n=("input_tokens", "size"))
    for i, row in data.requests.iterrows():
        if row["request_id"] not in g.index:
            continue
        t = g.loc[row["request_id"]]
        if not row.get("total_input_tokens"):
            data.requests.at[i, "total_input_tokens"] = int(t["i"])
        if not row.get("total_output_tokens"):
            data.requests.at[i, "total_output_tokens"] = int(t["o"])
        if not row.get("n_llm_calls"):
            data.requests.at[i, "n_llm_calls"] = int(t["n"])


# ── swe (react, trace has model_query_call+env_execute_call) ─────────
@_register("mini_sweagent")
def _extract_mini_sweagent(data: AnalysisData, data_dir: str):
    from . import task_registry as tr
    layout = C.AGENT_LAYOUT["mini_sweagent"]
    tids, td, diags = tr.get_valid_tasks("mini_sweagent", data_dir)
    data.diagnostics += diags
    if not tids:
        data.diagnostics.append("no valid tasks (need >=2 assistant turns)")
        return
    data.latency = _latency_from_trace(data_dir, layout, data.diagnostics)
    data.io = _io_from_intermediate(data_dir, layout, data.diagnostics)
    data.requests = _requests_from_tasks(td)
    # tokens + LLM IO from traj.json messages[].extra.response.usage
    tok_rows, llm_io = [], []
    for tid, t in td.items():
        d = read_json(t["traj_path"]) or {}
        msgs = d.get("messages", [])
        for m in msgs:
            if m.get("role") != "assistant":
                continue
            ex = m.get("extra") or {}
            resp = ex.get("response") or {}
            usage = resp.get("usage") or {}
            if usage:
                row = _token_row(tid, "model_query_call", "LLM", resp.get("model"), usage)
                if row:
                    row["cost_usd"] = float(ex.get("cost", 0) or 0)
                    tok_rows.append(row)
            # LLM IO: input = full message history up to here, output = assistant content
            # (approx: use this message's content as output; input = prior messages)
            out_bytes = content_bytes(m.get("content"))
            llm_io.append({"request_id": tid, "call_id": "", "function_name": "model_query_call",
                           "category": "LLM", "direction": "output", "bytes": out_bytes, "tokens": 0})
        # input side: each assistant turn's prompt = all prior messages
        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            in_bytes = sum(content_bytes(mm.get("content")) for mm in msgs[:i])
            llm_io.append({"request_id": tid, "call_id": "", "function_name": "model_query_call",
                           "category": "LLM", "direction": "input", "bytes": in_bytes, "tokens": 0})
    if tok_rows:
        data.tokens = pd.DataFrame(tok_rows)
        data.diagnostics.append(f"per-call tokens from traj.usage: {len(tok_rows)} rows")
    else:
        # fall back to aggregate model_stats cost only
        rows = []
        for tid, t in td.items():
            d = read_json(t["traj_path"]) or {}
            ms = (d.get("info") or {}).get("model_stats") or {}
            if ms:
                rows.append({"request_id": tid, "call_id": "", "function_name": "model_query_call",
                             "category": "LLM", "model": "", "input_tokens": 0, "output_tokens": 0,
                             "cache_read_tokens": 0, "cache_write_tokens": 0, "reasoning_tokens": 0,
                             "total_tokens": 0, "cost_usd": float(ms.get("instance_cost", 0) or 0)})
        if rows:
            data.tokens = pd.DataFrame(rows)
        data.diagnostics.append("no per-call usage in traj; aggregate cost only")
    # augment IO with LLM message bytes
    if llm_io:
        llm_df = pd.DataFrame(llm_io)
        data.io = pd.concat([data.io, llm_df], ignore_index=True) if not data.io.empty else llm_df


# ── codex (react, per-turn usage in codex_stdout) ────────────────────
@_register("codex")
def _extract_codex(data: AnalysisData, data_dir: str):
    from . import task_registry as tr
    layout = C.AGENT_LAYOUT["codex"]
    tids, td, diags = tr.get_valid_tasks("codex", data_dir)
    data.diagnostics += diags
    if not tids:
        data.diagnostics.append("no valid tasks (need >=2 turn.completed)")
        return
    data.latency = _latency_from_trace(data_dir, layout, data.diagnostics)
    data.io = _io_from_intermediate(data_dir, layout, data.diagnostics)
    data.requests = _requests_from_tasks(td)
    _fill_e2e_from_trace(data, data_dir)
    rows = []
    for tid, t in td.items():
        for ev in read_jsonl(t.get("stdout_path", "")):
            if ev.get("type") == "turn.completed":
                usage = ev.get("usage") or {}
                row = _token_row(tid, "codex_turn", "LLM", ev.get("model"), usage)
                if row:
                    rows.append(row)
    if rows:
        data.tokens = pd.DataFrame(rows)
    else:
        data.diagnostics.append("no turn.completed usage in codex_stdout")


# ── webagent (react, session.jsonl has llm_call.usage) ───────────────
@_register("webagent")
def _extract_webagent(data: AnalysisData, data_dir: str):
    from . import task_registry as tr
    layout = C.AGENT_LAYOUT["webagent"]
    tids, td, diags = tr.get_valid_tasks("webagent", data_dir)
    data.diagnostics += diags
    if not tids:
        data.diagnostics.append("no valid tasks (need n_steps>=2)")
        return
    data.latency = _latency_from_trace(data_dir, layout, data.diagnostics)
    data.io = _io_from_intermediate(data_dir, layout, data.diagnostics)
    data.requests = _requests_from_tasks(td)
    # tokens + trajectory-based breakdown fallback
    rows = []
    traj_lat = []
    for tid, t in td.items():
        events = read_jsonl(t["session_path"])
        for ev in events:
            if ev.get("event") == "llm_call":
                usage = ev.get("usage") or {}
                if not usage:
                    # webagent stores usage inside completion_response.usage
                    cr = ev.get("completion_response") or {}
                    usage = cr.get("usage") or {}
                row = _token_row(tid, "llm_call", "LLM", ev.get("model"), usage)
                if row:
                    rows.append(row)
        # trajectory breakdown (strategy A) — pair timestamps by event order
        ts_events = [(parse_iso_ts(e.get("timestamp")), e.get("event"))
                     for e in events if e.get("timestamp")]
        ts_events = [(t0, r) for t0, r in ts_events if t0]
        for i in range(1, len(ts_events)):
            delta = ts_events[i][0] - ts_events[i-1][0]
            if delta < 0:
                continue
            role = ts_events[i][1]
            cat = "LLM" if role == "llm_call" else ("Sandbox" if role == "env_step" else "Other")
            traj_lat.append({"request_id": tid, "call_id": "", "parent_call_id": "",
                             "function_name": role, "category": cat,
                             "start_time": ts_events[i-1][0], "end_time": ts_events[i][0],
                             "delay": delta, "delay_type": "traj", "is_top_level": False})
    if rows:
        data.tokens = pd.DataFrame(rows)
    else:
        data.diagnostics.append("no llm_call.usage in session.jsonl")
    # if trace was thin, supplement breakdown with trajectory deltas
    if not data.latency.empty and not (~data.latency["is_top_level"]).any() and traj_lat:
        data.diagnostics.append("supplementing breakdown from trajectory timestamps")
        data.latency = pd.concat([data.latency, pd.DataFrame(traj_lat)], ignore_index=True)


# ── guiagent (react, session.jsonl same shape) ───────────────────────
@_register("guiagent")
def _extract_guiagent(data: AnalysisData, data_dir: str):
    from . import task_registry as tr
    layout = C.AGENT_LAYOUT["guiagent"]
    tids, td, diags = tr.get_valid_tasks("guiagent", data_dir)
    data.diagnostics += diags
    if not tids:
        data.diagnostics.append("no valid tasks (need n_llm>=2)")
        return
    data.latency = _latency_from_trace(data_dir, layout, data.diagnostics)
    data.io = _io_from_intermediate(data_dir, layout, data.diagnostics)
    data.requests = _requests_from_tasks(td)
    rows = []
    traj_lat = []
    for tid, t in td.items():
        prev_llm_end = None
        for ev in read_jsonl(t["session_path"]):
            ts = parse_iso_ts(ev.get("timestamp"))
            if ev.get("event") == "llm_call":
                usage = ev.get("usage") or {}
                row = _token_row(tid, "llm_call", "LLM", ev.get("model"), usage)
                if row:
                    rows.append(row)
                # timestamp is logged after the call returns; duration_s is its wall time
                dur = float(ev.get("duration_s") or 0)
                if ts and dur > 0:
                    traj_lat.append({"request_id": tid, "call_id": "", "parent_call_id": "",
                                     "function_name": "llm_call", "category": "LLM",
                                     "start_time": ts - dur, "end_time": ts,
                                     "delay": dur, "delay_type": "traj", "is_top_level": False})
                prev_llm_end = ts
            elif ev.get("event") == "env_step" and ts and prev_llm_end and ts >= prev_llm_end:
                # VM boot (before the first call) and evaluation (after the last
                # step) are left out, so they land in the Other residual.
                traj_lat.append({"request_id": tid, "call_id": "", "parent_call_id": "",
                                 "function_name": "env_step", "category": "Sandbox",
                                 "start_time": prev_llm_end, "end_time": ts,
                                 "delay": ts - prev_llm_end, "delay_type": "traj", "is_top_level": False})
                prev_llm_end = ts
    if rows:
        data.tokens = pd.DataFrame(rows)
    else:
        data.diagnostics.append("no llm_call.usage in session.jsonl")
    # solve_task runs each task in a spawned subprocess with fsm_context=None,
    # so the trace has no sub-calls; take the breakdown from session.jsonl.
    if not data.latency.empty and not (~data.latency["is_top_level"]).any() and traj_lat:
        data.diagnostics.append("supplementing breakdown from session.jsonl durations")
        data.latency = pd.concat([data.latency, pd.DataFrame(traj_lat)], ignore_index=True)


# ── rag (pipeline, tokens in intermediate generation/embedding) ──────
@_register("rag")
def _extract_rag(data: AnalysisData, data_dir: str):
    from . import task_registry as tr
    layout = C.AGENT_LAYOUT["rag"]
    tids, td, diags = tr.get_valid_tasks("rag", data_dir)
    data.diagnostics += diags
    if not tids:
        data.diagnostics.append("no valid top-level e2e calls")
        return
    data.latency = _latency_from_trace(data_dir, layout, data.diagnostics)
    data.io = _io_from_intermediate(data_dir, layout, data.diagnostics)
    data.requests = _requests_from_tasks(td)
    # tokens: scan intermediate responses for usage blocks
    rows = _tokens_from_intermediate_usage(data_dir, layout, data.diagnostics)
    if rows:
        data.tokens = pd.DataFrame(rows)
    else:
        data.diagnostics.append("no usage in intermediate responses (rag)")


def _tokens_from_intermediate_usage(data_dir, layout, diags) -> list:
    """Scan intermediate responses.txt for an OpenAI-style usage dict in the
    parsed content. Works for rag/deepresearch/hugginggpt LLM-stage functions."""
    req_dir = os.path.join(data_dir, layout["intermediate"]) if layout.get("intermediate") else None
    inter = parse_intermediate(req_dir)
    rows = []
    for fn, entry in inter.items():
        cat = pipeline_op_category(fn)
        if cat not in ("LLM", "Embedding"):
            continue
        for r in entry["responses"]:
            rid = r.get("request_id", "")
            c = r.get("_content_parsed")
            usage = _find_usage(c)
            if usage:
                row = _token_row(rid, fn, cat, usage.get("model"), usage)
                if row:
                    rows.append(row)
    diags.append(f"intermediate-usage token rows={len(rows)}")
    return rows


def _find_usage(obj):
    """Recursively look for a usage-shaped dict {prompt_tokens|input_tokens, ...}."""
    if isinstance(obj, dict):
        if any(k in obj for k in ("prompt_tokens", "input_tokens", "completion_tokens", "output_tokens")):
            return obj
        for v in obj.values():
            found = _find_usage(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_usage(v)
            if found:
                return found
    return None


# ── deepresearch (pipeline) ──────────────────────────────────────────
@_register("deepresearch")
def _extract_deepresearch(data: AnalysisData, data_dir: str):
    from . import task_registry as tr
    layout = C.AGENT_LAYOUT["deepresearch"]
    tids, td, diags = tr.get_valid_tasks("deepresearch", data_dir)
    data.diagnostics += diags
    if not tids:
        data.diagnostics.append("no valid top-level e2e calls")
        return
    data.latency = _latency_from_trace(data_dir, layout, data.diagnostics)
    data.io = _io_from_intermediate(data_dir, layout, data.diagnostics)
    data.requests = _requests_from_tasks(td)
    rows = _tokens_from_intermediate_usage(data_dir, layout, data.diagnostics)
    if rows:
        data.tokens = pd.DataFrame(rows)
    else:
        data.diagnostics.append("no usage in intermediate responses (deepresearch)")


# ── hugginggpt (pipeline; model_inference = local, no tokens) ────────
@_register("hugginggpt")
def _extract_hugginggpt(data: AnalysisData, data_dir: str):
    from . import task_registry as tr
    layout = C.AGENT_LAYOUT["hugginggpt"]
    tids, td, diags = tr.get_valid_tasks("hugginggpt", data_dir)
    data.diagnostics += diags
    if not tids:
        data.diagnostics.append("no valid top-level e2e calls")
        return
    data.latency = _latency_from_trace(data_dir, layout, data.diagnostics)
    data.io = _io_from_intermediate(data_dir, layout, data.diagnostics)
    data.requests = _requests_from_tasks(td)
    rows = _tokens_from_intermediate_usage(data_dir, layout, data.diagnostics)
    if rows:
        data.tokens = pd.DataFrame(rows)
    else:
        data.diagnostics.append("no usage in intermediate (hugginggpt LLM stages); "
                                "model_inference is local HF (no API tokens)")


# ── claude (react; postprocess CSVs for breakdown, stream for IO/tokens) ─
@_register("claude")
def _extract_claude(data: AnalysisData, data_dir: str):
    from . import task_registry as tr
    import csv as _csv
    layout = C.AGENT_LAYOUT["claude"]
    tids, td, diags = tr.get_valid_tasks("claude", data_dir)
    data.diagnostics += diags
    if not tids:
        data.diagnostics.append("no valid tasks (need >=2 assistant events)")
        return
    data.latency = _latency_from_trace(data_dir, layout, data.diagnostics)
    if not data.latency.empty:
        # claude-service records every run_task stage with parent=None; only
        # run_task itself is the request e2e (the stages are inside it).
        data.latency = data.latency[
            data.latency["function_name"].str.endswith(".run_task")
        ].reset_index(drop=True)
    data.requests = _requests_from_tasks(td)
    _fill_e2e_from_trace(data, data_dir)
    rows, io_rows = [], []
    lat_from_csv = []
    for tid, t in td.items():
        run_dir = t.get("run_dir", "")
        # ── breakdown from postprocess CSVs (llm_calls.csv + tool_calls.csv) ──
        llm_csv = os.path.join(run_dir, "llm_calls.csv")
        tool_csv = os.path.join(run_dir, "tool_calls.csv")
        if os.path.exists(llm_csv):
            with open(llm_csv) as f:
                for row in _csv.DictReader(f):
                    lat_from_csv.append({
                        "request_id": tid, "call_id": "", "parent_call_id": "",
                        "function_name": "llm_call", "category": "LLM",
                        "start_time": 0, "end_time": 0,
                        "delay": float(row.get("latency_s", 0) or 0),
                        "delay_type": "postprocess", "is_top_level": False,
                    })
                    # per-step token data from llm_calls.csv. Anthropic usage
                    # excludes cache hits from input_tokens; add them back so
                    # input means the full prompt, as for the OpenAI agents.
                    usage = {
                        "prompt_tokens": int(row.get("input_tokens", 0) or 0)
                                         + int(row.get("cache_read_tokens", 0) or 0),
                        "completion_tokens": int(row.get("output_tokens", 0) or 0),
                        "cached_tokens": int(row.get("cache_read_tokens", 0) or 0),
                    }
                    tok_row = _token_row(tid, "llm_call", "LLM", row.get("model_id", ""), usage)
                    if tok_row:
                        rows.append(tok_row)
            data.diagnostics.append(f"breakdown from llm_calls.csv ({len([r for r in lat_from_csv if r['request_id']==tid])} LLM turns)")
        if os.path.exists(tool_csv):
            with open(tool_csv) as f:
                for row in _csv.DictReader(f):
                    lat_from_csv.append({
                        "request_id": tid, "call_id": "", "parent_call_id": "",
                        "function_name": row.get("tool", "tool_call"),
                        "category": claude_tool_category(row.get("tool", "")),
                        "start_time": 0, "end_time": 0,
                        "delay": float(row.get("duration_s", 0) or 0),
                        "delay_type": "postprocess", "is_top_level": False,
                    })
            data.diagnostics.append(f"breakdown from tool_calls.csv ({sum(1 for r in lat_from_csv if r['request_id']==tid and r['category']!='LLM')} tool calls)")

        # ── IO from stream content blocks ──
        events = read_jsonl(t["stream_path"])
        all_msgs = []
        for ev in events:
            if ev.get("type") in ("assistant", "user"):
                m = ev.get("message", {}) or {}
                if m.get("role"):
                    all_msgs.append(m)
        for i, m in enumerate(all_msgs):
            role = m.get("role", "")
            if role == "assistant":
                out = content_bytes(m.get("content"))
                inp = sum(content_bytes(mm.get("content")) for mm in all_msgs[:i])
                io_rows.append({"request_id": tid, "call_id": "", "function_name": "assistant",
                                "category": "LLM", "direction": "output", "bytes": out, "tokens": 0})
                io_rows.append({"request_id": tid, "call_id": "", "function_name": "assistant",
                                "category": "LLM", "direction": "input", "bytes": inp, "tokens": 0})
            elif role == "user":
                inp = content_bytes(m.get("content"))
                io_rows.append({"request_id": tid, "call_id": "", "function_name": "tool_result",
                                "category": "Sandbox", "direction": "output", "bytes": inp, "tokens": 0})

        # ── fallback: tokens from result event (aggregate) if no llm_calls.csv ──
        if not os.path.exists(llm_csv):
            for ev in events:
                if ev.get("type") == "result":
                    usage = ev.get("usage") or {}
                    row = _token_row(tid, "claude_run", "LLM", ev.get("model"), usage)
                    if row:
                        rows.append(row)

    # Use postprocess CSV latency for breakdown (supplements/replaces trace)
    if lat_from_csv:
        lat_df = pd.DataFrame(lat_from_csv)
        if not data.latency.empty:
            data.latency = pd.concat([data.latency, lat_df], ignore_index=True)
        else:
            data.latency = lat_df
    else:
        data.diagnostics.append("no llm_calls.csv/tool_calls.csv in task dirs -> breakdown N/A")

    if io_rows:
        data.io = pd.DataFrame(io_rows)
    if rows:
        data.tokens = pd.DataFrame(rows)
    else:
        data.diagnostics.append("no token data (neither llm_calls.csv nor stream result.usage)")


# ── openclaw (react; no trace; latency from chat.jsonl timestamps) ───
@_register("openclaw")
def _extract_openclaw(data: AnalysisData, data_dir: str):
    from . import task_registry as tr
    tids, td, diags = tr.get_valid_tasks("openclaw", data_dir)
    data.diagnostics += diags
    if not tids:
        data.diagnostics.append("no valid tasks (need >=2 assistant msgs)")
        return
    data.requests = _requests_from_tasks(td)
    rows, lat = [], []
    for tid, t in td.items():
        events = []
        for ev in read_jsonl(t["chat_path"]):
            if ev.get("type") == "message" and ev.get("timestamp"):
                msg = ev.get("message", {})
                role = msg.get("role", "")
                tool = msg.get("toolName", "") if role == "toolResult" else ""
                if role:
                    events.append((parse_iso_ts(ev["timestamp"]), role, tool))
        for i in range(1, len(events)):
            delta = events[i][0] - events[i-1][0]
            if delta < 0:
                continue
            cat = react_event_category(events[i][1], events[i][2])
            lat.append({"request_id": tid, "call_id": "", "parent_call_id": "",
                        "function_name": events[i][1], "category": cat,
                        "start_time": events[i-1][0], "end_time": events[i][0],
                        "delay": delta, "delay_type": "traj", "is_top_level": False})
        # tokens from assistant message.usage
        for ev in read_jsonl(t["chat_path"]):
            if ev.get("type") == "message":
                msg = ev.get("message", {})
                if msg.get("role") == "assistant" and msg.get("usage"):
                    row = _token_row(tid, "openclaw_turn", "LLM", msg.get("model"), msg["usage"])
                    if row:
                        rows.append(row)
    if lat:
        data.latency = pd.DataFrame(lat)
    else:
        data.diagnostics.append("no latency reconstructable from chat.jsonl")
    # IO from chat.jsonl content bytes
    io_rows = []
    for tid, t in td.items():
        events = list(read_jsonl(t["chat_path"]))
        msgs = [e.get("message", {}) for e in events if e.get("type") == "message"]
        for i, m in enumerate(msgs):
            role = m.get("role", "")
            if role == "assistant":
                # output = assistant content; input = full prior history
                out = content_bytes(m.get("content"))
                inp = sum(content_bytes(mm.get("content")) for mm in msgs[:i])
                io_rows.append({"request_id": tid, "call_id": "", "function_name": "assistant",
                                "category": "LLM", "direction": "output", "bytes": out, "tokens": 0})
                io_rows.append({"request_id": tid, "call_id": "", "function_name": "assistant",
                                "category": "LLM", "direction": "input", "bytes": inp, "tokens": 0})
            elif role == "toolResult":
                inp = len(json.dumps(m.get("content", "")).encode("utf-8")) if not isinstance(m.get("content"), list) else content_bytes(m.get("content"))
                io_rows.append({"request_id": tid, "call_id": "", "function_name": "toolResult",
                                "category": "Sandbox", "direction": "output", "bytes": inp, "tokens": 0})
    if io_rows:
        data.io = pd.DataFrame(io_rows)
    if rows:
        data.tokens = pd.DataFrame(rows)
    else:
        data.diagnostics.append("no assistant.usage in chat.jsonl")


# ── pi_autoresearch (react; no trace; pi-sessions) ───────────────────
@_register("pi_autoresearch")
def _extract_pi(data: AnalysisData, data_dir: str):
    from . import task_registry as tr
    tids, td, diags = tr.get_valid_tasks("pi_autoresearch", data_dir)
    data.diagnostics += diags
    if not tids:
        data.diagnostics.append("no valid tasks (need >=2 assistant msgs)")
        return
    data.requests = _requests_from_tasks(td)
    rows, lat = [], []
    for tid, t in td.items():
        events = []
        for ev in read_jsonl(t["session_path"]):
            if ev.get("type") == "message" and ev.get("timestamp"):
                role = ev.get("message", {}).get("role", "")
                if role:
                    events.append((parse_iso_ts(ev["timestamp"]), role))
        for i in range(1, len(events)):
            delta = events[i][0] - events[i-1][0]
            if delta < 0:
                continue
            cat = react_event_category(events[i][1])
            lat.append({"request_id": tid, "call_id": "", "parent_call_id": "",
                        "function_name": events[i][1], "category": cat,
                        "start_time": events[i-1][0], "end_time": events[i][0],
                        "delay": delta, "delay_type": "traj", "is_top_level": False})
        for ev in read_jsonl(t["session_path"]):
            if ev.get("type") == "message":
                msg = ev.get("message", {})
                if msg.get("role") == "assistant" and msg.get("usage"):
                    row = _token_row(tid, "pi_turn", "LLM", msg.get("model"), msg["usage"])
                    if row:
                        rows.append(row)
    if lat:
        data.latency = pd.DataFrame(lat)
    else:
        data.diagnostics.append("no latency from pi-sessions")
    # IO from pi-sessions content bytes
    io_rows = []
    for tid, t in td.items():
        events = list(read_jsonl(t["session_path"]))
        msgs = [e.get("message", {}) for e in events if e.get("type") == "message"]
        for i, m in enumerate(msgs):
            role = m.get("role", "")
            if role == "assistant":
                out = content_bytes(m.get("content"))
                inp = sum(content_bytes(mm.get("content")) for mm in msgs[:i])
                io_rows.append({"request_id": tid, "call_id": "", "function_name": "assistant",
                                "category": "LLM", "direction": "output", "bytes": out, "tokens": 0})
                io_rows.append({"request_id": tid, "call_id": "", "function_name": "assistant",
                                "category": "LLM", "direction": "input", "bytes": inp, "tokens": 0})
            elif role == "toolResult":
                inp = content_bytes(m.get("content"))
                io_rows.append({"request_id": tid, "call_id": "", "function_name": "toolResult",
                                "category": "Sandbox", "direction": "output", "bytes": inp, "tokens": 0})
    if io_rows:
        data.io = pd.DataFrame(io_rows)
    if rows:
        data.tokens = pd.DataFrame(rows)
    else:
        data.diagnostics.append("no assistant.usage in pi-sessions")
