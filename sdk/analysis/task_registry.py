"""Per-agent request-validity filtering — single source of truth for which
tasks/requests are "kept" vs "excluded".

Ported from agent-benchmark-bin all-agent branch ``task_registry.py``,
adapted to AgentSysBench's ``benchmark/data/<setting>/`` layout.

Common rule: a request is valid only if the agent actually did multi-step
work (>= 2 assistant turns), not a single errored response. Per-agent
existence checks augment this. Pipeline agents (no trajectory) fall back to
"top-level fast-service call present" in the e2e_delay.csv.
"""

import glob
import os
from typing import Dict, List, Tuple

from . import config as C
from ._parse import parse_e2e_csv, parse_iso_ts, read_json, read_jsonl


# ── Public API ───────────────────────────────────────────────────────
def get_valid_tasks(agent: str, data_dir: str) -> Tuple[List[str], Dict, List[str]]:
    """Return (task_ids, task_data, diagnostics) for one agent.

    task_data[id] = {paths..., e2e_s, n_steps, n_llm_calls, ...}
    diagnostics  = human-readable strings about what was found / missed.
    """
    loader = _LOADERS.get(agent)
    if loader is None:
        return [], [], [f"no loader registered for agent '{agent}'"]
    return loader(data_dir)


# ── Registry ─────────────────────────────────────────────────────────
_LOADERS = {}


def _register(agent):
    def deco(fn):
        _LOADERS[agent] = fn
        return fn
    return deco


# "success" = the agent run finished cleanly (no crash / timeout / upstream
# error); whether the task was actually solved is the evaluate step's job.
def _status_from_summary(summary: dict) -> dict:
    code = summary.get("exit_code")
    ok = code in (0, None) and not summary.get("timed_out") and not summary.get("error")
    err = summary.get("error") or ("timed out" if summary.get("timed_out") else
                                    None if ok else f"exit_code={code}")
    return {"success": ok, "error": err}


def _status_from_traj(traj: dict) -> dict:
    status = (traj.get("info") or {}).get("exit_status")
    ok = status in (None, "Submitted")
    return {"success": ok, "error": None if ok else status}


def _status_from_tb_traj(traj: dict) -> dict:
    """TerminalBench traj.json (codex): clean when codex exited 0 and the
    task's tests ran, whether or not they passed."""
    info = traj.get("info") or {}
    status = info.get("exit_status") or ""
    ok = info.get("error") is None and (status == "Resolved" or status.startswith("Failed("))
    return {"success": ok, "error": None if ok else (info.get("error") or status)}


def _status_from_pi_traj(traj: dict) -> dict:
    """pi-autoresearch traj.json: clean when the run ended without error
    (reaching the time limit or maxIterations is a normal finish) and the
    submission was graded, whether or not it scored well."""
    err = (traj.get("info") or {}).get("error")
    return {"success": err is None, "error": err}


def _count_assistant_messages_jsonl(path: str) -> int:
    """Count messages with message.role == 'assistant' in a jsonl of
    {type:'message', message:{role}} or {event:'...'} rows."""
    n = 0
    for ev in read_jsonl(path):
        msg = ev.get("message") if isinstance(ev.get("message"), dict) else None
        if msg and msg.get("role") == "assistant":
            n += 1
    return n


# ── React agents (trajectory-based validity) ─────────────────────────

@_register("mini_sweagent")
def _load_mini_sweagent(data_dir: str):
    layout = C.AGENT_LAYOUT["mini_sweagent"]
    traj_dir = os.path.join(data_dir, layout["traj_dir"])
    diags: List[str] = []
    if not os.path.isdir(traj_dir):
        return [], [], [f"traj_dir missing: {traj_dir}"]
    task_ids, task_data = [], {}
    scanned = 0
    for tj in sorted(glob.glob(os.path.join(traj_dir, layout["traj_glob"]))):
        scanned += 1
        d = read_json(tj)
        if not d:
            continue
        messages = d.get("messages", [])
        asst = [m for m in messages if m.get("role") == "assistant" and "extra" in m]
        if len(asst) < 2:
            continue
        # e2e from message timestamps
        ts = [float(m["extra"]["timestamp"]) for m in messages
              if "extra" in m and m["extra"].get("timestamp") is not None]
        try:
            ts = [float(x) for x in ts]
        except (TypeError, ValueError):
            ts = []
        e2e_s = (ts[-1] - ts[0]) if len(ts) >= 2 else 0.0
        tid = os.path.basename(tj).rsplit(".traj.json", 1)[0]
        task_ids.append(tid)
        task_data[tid] = {"traj_path": tj, "n_steps": len(asst), "e2e_s": e2e_s,
                          **_status_from_traj(d)}
    diags.append(f"scanned={scanned} kept={len(task_ids)}")
    # also expose e2e from fast-service trace for breakdown attribution
    return task_ids, task_data, diags


@_register("codex")
def _load_codex(data_dir: str):
    layout = C.AGENT_LAYOUT["codex"]
    traj_dir = os.path.join(data_dir, layout["traj_dir"])
    diags: List[str] = []
    if not os.path.isdir(traj_dir):
        return [], [], [f"traj_dir missing: {traj_dir}"]
    task_ids, task_data = [], {}
    scanned = 0
    for tj in sorted(glob.glob(os.path.join(traj_dir, layout["traj_glob"]))):
        scanned += 1
        tid = os.path.basename(tj).rsplit(".traj.json", 1)[0]
        stdout_path = os.path.join(traj_dir, tid, f"{tid}.codex_stdout.jsonl")
        if not os.path.exists(stdout_path):
            continue
        events = read_jsonl(stdout_path)
        # AgentSysBench codex_stdout: turn.completed events carry usage.
        # A task can legitimately complete in 1 turn, so >=1 (not 2) is the
        # validity bar; the previous 401/error runs produce 0 turn.completed.
        n_turns = sum(1 for e in events if e.get("type") == "turn.completed")
        if n_turns < 1:
            continue
        task_ids.append(tid)
        task_data[tid] = {"traj_path": tj, "stdout_path": stdout_path,
                          "n_steps": n_turns, "e2e_s": 0.0,
                          **_status_from_tb_traj(read_json(tj) or {})}
    diags.append(f"scanned={scanned} kept={len(task_ids)}")
    return task_ids, task_data, diags


@_register("webagent")
def _load_webagent(data_dir: str):
    layout = C.AGENT_LAYOUT["webagent"]
    traj_dir = os.path.join(data_dir, layout["traj_dir"])
    diags: List[str] = []
    if not os.path.isdir(traj_dir):
        return [], [], [f"traj_dir missing: {traj_dir}"]
    task_ids, task_data = [], {}
    scanned = 0
    for sp in sorted(glob.glob(os.path.join(traj_dir, "*/*.session.jsonl"))):
        scanned += 1
        events = read_jsonl(sp)
        end_ev = [e for e in events if e.get("event") == "session_end"]
        n_steps = end_ev[0].get("n_steps", 0) if end_ev else 0
        if n_steps < 1:
            continue
        tid = os.path.basename(sp).rsplit(".session.jsonl", 1)[0]
        start_ts = events[0].get("timestamp") if events else None
        end_ts = end_ev[0].get("timestamp") if end_ev else None
        s0, s1 = parse_iso_ts(start_ts), parse_iso_ts(end_ts)
        e2e_s = (s1 - s0) if (s0 and s1) else 0.0
        task_ids.append(tid)
        task_data[tid] = {"session_path": sp, "n_steps": n_steps, "e2e_s": e2e_s}
    diags.append(f"scanned={scanned} kept={len(task_ids)}")
    return task_ids, task_data, diags


@_register("guiagent")
def _load_guiagent(data_dir: str):
    # same session.jsonl shape as webagent
    layout = C.AGENT_LAYOUT["guiagent"]
    traj_dir = os.path.join(data_dir, layout["traj_dir"])
    diags: List[str] = []
    if not os.path.isdir(traj_dir):
        return [], [], [f"traj_dir missing: {traj_dir}"]
    task_ids, task_data = [], {}
    scanned = 0
    for sp in sorted(glob.glob(os.path.join(traj_dir, "*/*.session.jsonl"))):
        scanned += 1
        events = read_jsonl(sp)
        end_ev = [e for e in events if e.get("event") == "session_end"]
        n_steps = end_ev[0].get("n_steps", 0) if end_ev else 0
        # gui-agent counts llm_call events for the "n_llm_calls" validity knob
        n_llm = sum(1 for e in events if e.get("event") == "llm_call")
        if n_llm < 2:
            continue
        tid = os.path.basename(sp).rsplit(".session.jsonl", 1)[0]
        start_ts = events[0].get("timestamp") if events else None
        end_ts = end_ev[0].get("timestamp") if end_ev else None
        s0, s1 = parse_iso_ts(start_ts), parse_iso_ts(end_ts)
        e2e_s = (s1 - s0) if (s0 and s1) else 0.0
        task_ids.append(tid)
        task_data[tid] = {"session_path": sp, "n_steps": n_steps,
                          "n_llm_calls": n_llm, "e2e_s": e2e_s}
    diags.append(f"scanned={scanned} kept={len(task_ids)}")
    return task_ids, task_data, diags


@_register("claude")
def _load_claude(data_dir: str):
    layout = C.AGENT_LAYOUT["claude"]
    tasks_dir = os.path.join(data_dir, layout["traj_dir"])
    diags: List[str] = []
    if not os.path.isdir(tasks_dir):
        return [], [], [f"tasks_dir missing: {tasks_dir}"]
    task_ids, task_data = [], {}
    scanned = 0
    for sp in sorted(glob.glob(os.path.join(tasks_dir, layout["traj_glob"]))):
        scanned += 1
        events = read_jsonl(sp)
        n_asst = sum(1 for e in events if e.get("type") == "assistant")
        if n_asst < 2:
            continue
        run_dir = os.path.dirname(sp)
        tid = os.path.basename(os.path.dirname(sp))  # <run_id>
        summary = read_json(os.path.join(run_dir, "run_summary.json")) or {}
        wall_s = float(summary.get("wall_s") or 0)
        task_ids.append(tid)
        task_data[tid] = {"stream_path": sp, "run_dir": run_dir,
                          "summary": summary, "n_steps": n_asst, "e2e_s": wall_s,
                          **_status_from_summary(summary)}
    diags.append(f"scanned={scanned} kept={len(task_ids)}")
    return task_ids, task_data, diags


@_register("openclaw")
def _load_openclaw(data_dir: str):
    layout = C.AGENT_LAYOUT["openclaw"]
    base = os.path.join(data_dir, layout["traj_dir"])
    diags: List[str] = []
    if not os.path.isdir(base):
        return [], [], [f"output dir missing: {base}"]
    task_ids, task_data = [], {}
    scanned = 0
    for sp in sorted(glob.glob(os.path.join(base, layout["session_glob"]))):
        scanned += 1
        n_asst = _count_assistant_messages_jsonl(sp)
        if n_asst < 2:
            continue
        run_dir = os.path.dirname(sp)
        summary = read_json(os.path.join(run_dir, "run_summary.json")) or {}
        wall_s = float(summary.get("wall_s") or 0)
        if wall_s <= 0:
            continue
        tid = f"{os.path.basename(os.path.dirname(run_dir))}/{os.path.basename(run_dir)}"
        task_ids.append(tid)
        task_data[tid] = {"chat_path": sp, "run_dir": run_dir,
                          "summary": summary, "n_steps": n_asst, "e2e_s": wall_s,
                          **_status_from_summary(summary)}
    diags.append(f"scanned={scanned} kept={len(task_ids)}")
    return task_ids, task_data, diags


@_register("pi_autoresearch")
def _load_pi_autoresearch(data_dir: str):
    layout = C.AGENT_LAYOUT["pi_autoresearch"]
    traj_dir = os.path.join(data_dir, layout["traj_dir"])
    diags: List[str] = []
    if not os.path.isdir(traj_dir):
        return [], [], [f"traj_dir missing: {traj_dir}"]
    task_ids, task_data = [], {}
    scanned = 0
    # session files live under <task>/pi-sessions/**/*.jsonl
    for sp in sorted(glob.glob(os.path.join(traj_dir, layout["session_glob"]))):
        scanned += 1
        n_asst = _count_assistant_messages_jsonl(sp)
        if n_asst < 2:
            continue
        events = read_jsonl(sp)
        ts = [parse_iso_ts(e["timestamp"]) for e in events if e.get("timestamp")]
        ts = [t for t in ts if t]
        e2e_s = (ts[-1] - ts[0]) if len(ts) >= 2 else 0.0
        tid = os.path.basename(os.path.dirname(os.path.dirname(sp)))
        traj = read_json(os.path.join(traj_dir, tid, f"{tid}.traj.json")) or {}
        task_ids.append(tid)
        task_data[tid] = {"session_path": sp, "n_steps": n_asst, "e2e_s": e2e_s,
                          **_status_from_pi_traj(traj)}
    diags.append(f"scanned={scanned} kept={len(task_ids)}")
    return task_ids, task_data, diags


# ── Pipeline agents (trace-based validity) ───────────────────────────

def _pipeline_valid_from_trace(data_dir: str, layout, diags: List[str]):
    """For rag/deepresearch/hugginggpt: a request is valid if it has a
    top-level (parent_service_name == 'None') e2e row with delay > 0.
    Returns (task_ids, task_data, diags)."""
    e2e_csv = os.path.join(data_dir, layout["e2e_csv"]) if layout.get("e2e_csv") else None
    if not e2e_csv or not os.path.exists(e2e_csv):
        diags.append(f"e2e_csv missing: {e2e_csv}")
        return [], {}, diags
    rows = parse_e2e_csv(e2e_csv)
    top = [r for r in rows if r["parent_service_name"] == "None" and r["delay"] > 0]
    diags.append(f"e2e rows={len(rows)} top-level={len(top)}")
    task_ids, task_data = [], {}
    for r in top:
        rid = r["request_id"]
        task_ids.append(rid)
        task_data[rid] = {"request_id": rid, "e2e_s": r["delay"]}
    return task_ids, task_data, diags


@_register("rag")
def _load_rag(data_dir: str):
    return _pipeline_valid_from_trace(data_dir, C.AGENT_LAYOUT["rag"], [])


@_register("deepresearch")
def _load_deepresearch(data_dir: str):
    return _pipeline_valid_from_trace(data_dir, C.AGENT_LAYOUT["deepresearch"], [])


@_register("hugginggpt")
def _load_hugginggpt(data_dir: str):
    return _pipeline_valid_from_trace(data_dir, C.AGENT_LAYOUT["hugginggpt"], [])
