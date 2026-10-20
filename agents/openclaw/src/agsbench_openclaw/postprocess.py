"""Telemetry post-processor: combine chat.jsonl + LLM-proxy audit log into
unified per-call CSVs in the schema used by the WildClawBench analysis pipeline.

Reuses the extraction logic from
  /data/agent-sys-bench
  /data/agent-sys-bench
adapted to a single trajectory and a single chat.jsonl, then joins with the
LLM proxy's `llm_calls.jsonl` audit log to add server-side timing (`upstream_ms`,
`parse_ms`, `total_ms`) to each LLM-step row.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Optional


def parse_ts(s) -> float:
    if isinstance(s, (int, float)):
        return float(s) / 1000.0 if s > 1e12 else float(s)
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def extract_llm_calls(chat_jsonl_path: Path) -> list[dict]:
    msgs = []
    with chat_jsonl_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "message":
                msgs.append(d)
    if not msgs:
        return []

    rows = []
    prev_ts = parse_ts(msgs[0]["timestamp"])
    step = 0
    for m in msgs[1:]:
        ts = parse_ts(m["timestamp"])
        gap = ts - prev_ts
        msg = m["message"]
        if msg.get("role") == "assistant":
            step += 1
            usage = msg.get("usage") or {}
            text_chars = 0
            thinking_chars = 0
            n_tc = 0
            tool_names = []
            for c in msg.get("content", []) or []:
                t = c.get("type")
                if t == "text":
                    text_chars += len(c.get("text") or "")
                elif t == "thinking":
                    thinking_chars += len(c.get("thinking") or "")
                elif t == "toolCall":
                    n_tc += 1
                    tool_names.append(c.get("name") or "")
            cost = (usage.get("cost") or {}).get("total", 0.0)
            api = msg.get("api", "")
            # openclaw inserts synthetic transcript-only assistant entries
            # with api="cli" via persistTextTurnTranscript after the run
            # finishes. They are NOT real LLM calls (no network roundtrip,
            # no upstream usage) so they don't appear in the proxy audit
            # log. We label them so downstream analysis can drop or treat
            # them differently.
            is_synthetic = api == "cli"
            input_tokens = usage.get("input", 0)
            cache_read = usage.get("cacheRead", 0)
            # full_prompt_tokens = what the model actually saw as context
            # this step. The chat.jsonl `input` is the *fresh* post-cache
            # prefill; cache_read is the prefix served from cache.
            # For synthetic api:cli rows the components aren't
            # self-consistent (cacheRead is a cumulative snapshot, total
            # is the last real call's total) so full_prompt is unreliable
            # there — we still compute it but is_synthetic flags the row
            # so downstream analysis can drop those.
            full_prompt_tokens = input_tokens + cache_read
            rows.append(
                {
                    "step_idx": step,
                    "is_synthetic": int(is_synthetic),
                    "api": api,
                    "latency_s": gap,
                    "input_tokens": input_tokens,
                    "output_tokens": usage.get("output", 0),
                    "cache_read_tokens": cache_read,
                    "cache_write_tokens": usage.get("cacheWrite", 0),
                    "total_tokens": usage.get("totalTokens", 0),
                    "full_prompt_tokens": full_prompt_tokens,
                    "cost_usd": cost,
                    "text_chars": text_chars,
                    "thinking_chars": thinking_chars,
                    "has_thinking": int(thinking_chars > 0),
                    "n_tool_calls": n_tc,
                    "tool_names": "|".join(tool_names),
                    "stop_reason": msg.get("stopReason", ""),
                    "model_id": msg.get("model", ""),
                }
            )
        prev_ts = ts
    return rows


def extract_tool_calls(chat_jsonl_path: Path) -> list[dict]:
    msgs = []
    with chat_jsonl_path.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("type") == "message":
                msgs.append(d)
    if not msgs:
        return []

    calls = {}
    n_assist_seen = 0
    for pos, m in enumerate(msgs):
        msg = m["message"]
        if msg.get("role") != "assistant":
            continue
        n_assist_seen += 1
        for c in msg.get("content", []) or []:
            if c.get("type") != "toolCall":
                continue
            cid = c.get("id") or c.get("toolUseId")
            calls[cid] = {
                "assistant_msg_idx": n_assist_seen,
                "tool": c.get("name") or "<unknown>",
                "args_bytes": len(json.dumps(c.get("arguments", {}))),
            }

    rows = []
    prev_ts = parse_ts(msgs[0]["timestamp"])
    call_idx = 0
    for m in msgs[1:]:
        ts = parse_ts(m["timestamp"])
        gap = ts - prev_ts
        msg = m["message"]
        if msg.get("role") == "toolResult":
            cid = msg.get("toolCallId")
            info = calls.get(cid, {})
            details = msg.get("details") or {}
            result_bytes = 0
            for c in msg.get("content", []) or []:
                if c.get("type") == "text":
                    result_bytes += len(c.get("text") or "")
            dur_ms = details.get("durationMs")
            took_ms = details.get("tookMs")
            length = details.get("length")
            raw_length = details.get("rawLength")
            rows.append(
                {
                    "call_idx": call_idx,
                    "assistant_msg_idx": info.get("assistant_msg_idx"),
                    "tool": info.get("tool") or msg.get("toolName") or "<unknown>",
                    "is_error": bool(msg.get("isError")),
                    "exit_code": details.get("exitCode"),
                    "status": details.get("status"),
                    "args_bytes": info.get("args_bytes", 0),
                    "result_bytes": result_bytes,
                    "raw_length": raw_length if raw_length is not None else "",
                    "length": length if length is not None else "",
                    "gap_s": gap,
                    "duration_ms": dur_ms if dur_ms is not None else "",
                    "took_ms": took_ms if took_ms is not None else "",
                }
            )
            call_idx += 1
        prev_ts = ts
    return rows


def load_proxy_audit(jsonl_path: Path) -> list[dict]:
    """Read the LLM proxy's per-call audit log."""
    rows = []
    if not jsonl_path.exists():
        return rows
    with jsonl_path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rows.sort(key=lambda r: r.get("timestamp", 0))
    return rows


def join_llm_with_proxy(llm_rows: list[dict], proxy_rows: list[dict]) -> list[dict]:
    """Match openclaw assistant messages to proxy audit rows.

    Both lists are chronological. Synthetic transcript entries (api="cli")
    in llm_rows have no corresponding proxy row, so we pair only the
    non-synthetic llm_rows against the last-N proxy rows.

    We use last-N (not first-N) so the join works even if the proxy was
    serving prior runs and has a longer history; we assume each run is
    bracketed (proxy restart, or proxy_rows file truncated) so the *most
    recent* proxy_rows correspond to this run's LLM calls.
    """
    proxy_fields = [
        ("proxy_total_ms", "total_ms"),
        ("proxy_upstream_ms", "upstream_ms"),
        ("proxy_parse_ms", "parse_ms"),
        ("proxy_request_id", "request_id"),
        ("proxy_upstream_id", "upstream_id"),
        ("proxy_status_code", "status_code"),
        ("proxy_finish_reason", "finish_reason"),
        ("proxy_prompt_tokens", "prompt_tokens"),
        ("proxy_completion_tokens", "completion_tokens"),
        ("proxy_cache_read_tokens", "cache_read_tokens"),
        ("proxy_reasoning_tokens", "reasoning_tokens"),
    ]
    # default all proxy_* fields to None
    for r in llm_rows:
        for k, _ in proxy_fields:
            r[k] = None

    real_rows = [r for r in llm_rows if not r.get("is_synthetic")]
    n_real = len(real_rows)
    n_prox = len(proxy_rows)
    aligned_proxy = proxy_rows[-n_real:] if n_prox >= n_real else proxy_rows
    for i, r in enumerate(real_rows):
        p = aligned_proxy[i] if i < len(aligned_proxy) else {}
        for k, src in proxy_fields:
            r[k] = p.get(src)

    # Derived columns after the join. reasoning_tokens comes from the
    # upstream usage.completion_tokens_details.reasoning_tokens — non-null
    # only when the model is a reasoning model AND `thinking` is enabled
    # (DeepSeek-V4-Flash returns it when `thinking: {type: enabled}` is
    # in the request body; with `thinking: {type: disabled}` it returns
    # nothing on this field and all output is plain text).
    for r in llm_rows:
        out = int(r.get("output_tokens") or 0)
        reasoning = r.get("proxy_reasoning_tokens")
        try:
            reasoning_int = int(reasoning) if reasoning is not None else None
        except (TypeError, ValueError):
            reasoning_int = None
        r["reasoning_tokens"] = reasoning_int if reasoning_int is not None else ""
        # visible_output_tokens = the portion of output_tokens that's
        # user-facing text (everything that isn't internal reasoning).
        # When reasoning_tokens is unknown (proxy didn't capture it, or
        # synthetic row), we leave it blank rather than guessing.
        if reasoning_int is not None:
            r["visible_output_tokens"] = max(0, out - reasoning_int)
        else:
            r["visible_output_tokens"] = ""
    return llm_rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        # write empty file with no header — let the consumer notice.
        out_path.write_text("")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def postprocess(
    chat_jsonl_path: Path,
    proxy_audit_jsonl_path: Path,
    output_dir: Path,
    extra_context: Optional[dict] = None,
) -> dict:
    extra_context = extra_context or {}
    llm_rows = extract_llm_calls(chat_jsonl_path)
    tool_rows = extract_tool_calls(chat_jsonl_path)
    proxy_rows = load_proxy_audit(proxy_audit_jsonl_path)

    llm_rows = join_llm_with_proxy(llm_rows, proxy_rows)

    # Inject per-row context so the CSVs are self-describing
    for r in llm_rows:
        r.update(extra_context)
    for r in tool_rows:
        r.update(extra_context)

    write_csv(llm_rows, output_dir / "llm_calls.csv")
    write_csv(tool_rows, output_dir / "tool_calls.csv")

    return {
        "n_llm_calls": len(llm_rows),
        "n_tool_calls": len(tool_rows),
        "n_proxy_rows": len(proxy_rows),
        "llm_calls_csv": str(output_dir / "llm_calls.csv"),
        "tool_calls_csv": str(output_dir / "tool_calls.csv"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-jsonl", required=True)
    ap.add_argument(
        "--proxy-audit",
        required=True,
        help="Path to the proxy's llm_calls.jsonl audit log.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--task-id", default="")
    ap.add_argument("--category", default="")
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()

    extra = {
        "model": args.model,
        "task_id": args.task_id,
        "category": args.category,
        "run_id": args.run_id,
    }
    summary = postprocess(
        Path(args.chat_jsonl),
        Path(args.proxy_audit),
        Path(args.output_dir),
        extra_context=extra,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
