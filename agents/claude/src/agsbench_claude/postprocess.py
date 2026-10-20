"""Postprocess claude session JSONL into per-LLM-call and per-tool-call CSVs.

Reads the session JSONL that claude writes to
~/.claude/projects/<cwd>/<session-id>.jsonl and produces:
- llm_calls.csv: one row per LLM turn with latency and token usage
- tool_calls.csv: one row per tool invocation with duration

Optionally joins with mcp_tool_calls.jsonl from mcp_bridge.py for
direct MCP tool execution timing.

Output format is consistent with openclaw/postprocess.py.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("claude.postprocess")


def _parse_ts(ts_str: str) -> float:
    """Parse ISO 8601 timestamp to epoch seconds."""
    ts_str = ts_str.rstrip("Z")
    if "." in ts_str:
        dt = datetime.fromisoformat(ts_str)
    else:
        dt = datetime.fromisoformat(ts_str)
    return dt.timestamp()


def _load_session_events(session_jsonl_path: Path) -> list:
    """Load and filter relevant events from session JSONL."""
    events = []
    with open(session_jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = evt.get("type")
            if t in ("assistant", "user"):
                ts = evt.get("timestamp")
                if ts:
                    evt["_ts"] = _parse_ts(ts)
                    events.append(evt)
    return events


def _extract_llm_calls(events: list) -> list:
    """Extract per-LLM-turn timing from session events.

    A turn starts at the first assistant event after a user event (or
    at the first assistant event of the session). The LLM latency is
    the gap between the previous user event's timestamp and the first
    assistant event's timestamp.
    """
    turns = []
    current_turn = None
    prev_user_ts = None

    for evt in events:
        t = evt.get("type")
        ts = evt.get("_ts")
        if ts is None:
            continue

        if t == "user":
            msg = evt.get("message", {})
            content = msg.get("content", "")
            # Check if this is a tool_result or a user prompt
            is_tool_result = False
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        is_tool_result = True
                        break

            if is_tool_result and current_turn is not None:
                current_turn["tool_results"].append({"ts": ts, "event": evt})

            prev_user_ts = ts

        elif t == "assistant":
            msg = evt.get("message", {})
            content = msg.get("content", [])
            usage = msg.get("usage", {})

            content_types = []
            tool_names = []
            thinking_chars = 0
            text_chars = 0
            for c in content:
                if not isinstance(c, dict):
                    continue
                ct = c.get("type", "")
                content_types.append(ct)
                if ct == "tool_use":
                    tool_names.append(c.get("name", ""))
                elif ct == "thinking":
                    thinking_chars += len(c.get("thinking", ""))
                elif ct == "text":
                    text_chars += len(c.get("text", ""))

            if "tool_use" in content_types or "text" in content_types:
                if current_turn is None or (
                    prev_user_ts
                    and prev_user_ts > current_turn.get("first_assistant_ts", 0)
                ):
                    # New turn
                    if current_turn is not None:
                        turns.append(current_turn)
                    llm_latency = ts - prev_user_ts if prev_user_ts else 0
                    current_turn = {
                        "first_assistant_ts": ts,
                        "llm_latency_s": llm_latency,
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                        "tool_names": tool_names,
                        "thinking_chars": thinking_chars,
                        "text_chars": text_chars,
                        "model_id": msg.get("model", ""),
                        "tool_results": [],
                    }
                else:
                    # Same turn, additional content block
                    current_turn["tool_names"].extend(tool_names)
                    current_turn["thinking_chars"] += thinking_chars
                    current_turn["text_chars"] += text_chars
            elif "thinking" in content_types:
                # Thinking-only block — part of the upcoming assistant response
                if current_turn is None or (
                    prev_user_ts
                    and prev_user_ts > current_turn.get("first_assistant_ts", 0)
                ):
                    llm_latency = ts - prev_user_ts if prev_user_ts else 0
                    if current_turn is not None:
                        turns.append(current_turn)
                    current_turn = {
                        "first_assistant_ts": ts,
                        "llm_latency_s": llm_latency,
                        "input_tokens": usage.get("input_tokens", 0),
                        "output_tokens": usage.get("output_tokens", 0),
                        "cache_read_tokens": usage.get("cache_read_input_tokens", 0),
                        "tool_names": [],
                        "thinking_chars": thinking_chars,
                        "text_chars": 0,
                        "model_id": msg.get("model", ""),
                        "tool_results": [],
                    }
                else:
                    current_turn["thinking_chars"] += thinking_chars

    if current_turn is not None:
        turns.append(current_turn)

    return turns


def _extract_tool_calls(events: list) -> list:
    """Extract per-tool-call timing from session events.

    For each tool_use assistant event, find the matching tool_result
    user event and compute the duration from timestamps.
    """
    tool_calls = []
    pending_tools = {}  # tool_use_id -> {name, ts, args_bytes, step_idx}
    step_idx = 0
    call_idx = 0

    for evt in events:
        t = evt.get("type")
        ts = evt.get("_ts")
        if ts is None:
            continue

        if t == "assistant":
            content = evt.get("message", {}).get("content", [])
            for c in content:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    tool_id = c.get("id", "")
                    pending_tools[tool_id] = {
                        "name": c.get("name", ""),
                        "ts": ts,
                        "args_bytes": len(json.dumps(c.get("input", {}))),
                        "step_idx": step_idx,
                    }
                elif isinstance(c, dict) and c.get("type") in ("text",):
                    step_idx += 1

        elif t == "user":
            content = evt.get("message", {}).get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "tool_result":
                        tool_id = c.get("tool_use_id", "")
                        if tool_id in pending_tools:
                            info = pending_tools.pop(tool_id)
                            result_content = c.get("content", "")
                            if isinstance(result_content, list):
                                result_bytes = sum(
                                    len(json.dumps(x)) for x in result_content
                                )
                            else:
                                result_bytes = len(str(result_content))
                            tool_calls.append(
                                {
                                    "call_idx": call_idx,
                                    "step_idx": info["step_idx"],
                                    "tool": info["name"],
                                    "duration_s": round(ts - info["ts"], 4),
                                    "args_bytes": info["args_bytes"],
                                    "result_bytes": result_bytes,
                                    "is_error": bool(c.get("is_error")),
                                }
                            )
                            call_idx += 1
                step_idx += 1

    return tool_calls


def _load_mcp_bridge_log(mcp_tool_log_path: Path) -> list:
    """Load mcp_bridge per-tool-call timing log."""
    records = []
    if not mcp_tool_log_path.exists():
        return records
    with open(mcp_tool_log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def postprocess(
    session_jsonl_path: Path,
    output_dir: Path,
    task_id: str = "",
    run_id: str = "",
    mcp_tool_log_path: Optional[Path] = None,
):
    """Parse session JSONL into llm_calls.csv and tool_calls.csv."""
    events = _load_session_events(session_jsonl_path)
    if not events:
        log.warning("No events found in %s", session_jsonl_path)
        return

    llm_turns = _extract_llm_calls(events)
    tool_calls = _extract_tool_calls(events)

    # Load MCP bridge log if available
    bridge_records = []
    if mcp_tool_log_path and mcp_tool_log_path.exists():
        bridge_records = _load_mcp_bridge_log(mcp_tool_log_path)

    # Join bridge records with tool_calls by sequential order for MCP tools
    mcp_idx = 0
    for tc in tool_calls:
        tc["bridge_elapsed_ms"] = ""
        if mcp_idx < len(bridge_records):
            br = bridge_records[mcp_idx]
            if br.get("tool_name", "").split("_", 1)[0] in tc.get("tool", ""):
                tc["bridge_elapsed_ms"] = br.get("elapsed_ms", "")
                mcp_idx += 1

    # Write llm_calls.csv
    llm_csv_path = output_dir / "llm_calls.csv"
    llm_fields = [
        "step_idx",
        "latency_s",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "n_tool_calls",
        "tool_names",
        "thinking_chars",
        "text_chars",
        "model_id",
        "task_id",
        "run_id",
    ]
    with open(llm_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=llm_fields)
        w.writeheader()
        for i, turn in enumerate(llm_turns):
            w.writerow(
                {
                    "step_idx": i + 1,
                    "latency_s": round(turn["llm_latency_s"], 4),
                    "input_tokens": turn["input_tokens"],
                    "output_tokens": turn["output_tokens"],
                    "cache_read_tokens": turn["cache_read_tokens"],
                    "n_tool_calls": len(turn["tool_names"]),
                    "tool_names": "|".join(turn["tool_names"]),
                    "thinking_chars": turn["thinking_chars"],
                    "text_chars": turn["text_chars"],
                    "model_id": turn["model_id"],
                    "task_id": task_id,
                    "run_id": run_id,
                }
            )

    # Write tool_calls.csv
    tool_csv_path = output_dir / "tool_calls.csv"
    tool_fields = [
        "call_idx",
        "step_idx",
        "tool",
        "duration_s",
        "bridge_elapsed_ms",
        "args_bytes",
        "result_bytes",
        "is_error",
        "task_id",
        "run_id",
    ]
    with open(tool_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=tool_fields)
        w.writeheader()
        for tc in tool_calls:
            w.writerow(
                {
                    "call_idx": tc["call_idx"],
                    "step_idx": tc["step_idx"],
                    "tool": tc["tool"],
                    "duration_s": tc["duration_s"],
                    "bridge_elapsed_ms": tc.get("bridge_elapsed_ms", ""),
                    "args_bytes": tc["args_bytes"],
                    "result_bytes": tc["result_bytes"],
                    "is_error": tc["is_error"],
                    "task_id": task_id,
                    "run_id": run_id,
                }
            )

    log.info(
        "Postprocessed %s: %d LLM turns, %d tool calls -> %s, %s",
        session_jsonl_path.name,
        len(llm_turns),
        len(tool_calls),
        llm_csv_path,
        tool_csv_path,
    )

    return {
        "llm_turns": len(llm_turns),
        "tool_calls": len(tool_calls),
        "total_llm_latency_s": round(sum(t["llm_latency_s"] for t in llm_turns), 3),
        "total_tool_duration_s": round(sum(t["duration_s"] for t in tool_calls), 3),
    }
