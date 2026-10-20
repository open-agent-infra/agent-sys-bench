"""Low-level raw-data parsers shared across extractors.

All parsers are tolerant: malformed lines are skipped, missing files return
empty structures. Callers use the diagnostics channel to report what was
found vs missing.
"""

import csv
import glob
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional


def parse_e2e_csv(path: str) -> List[dict]:
    """Parse a fast-service e2e_delay.csv / remote_delay.csv.

    The file has NO header and 7 positional columns:
        end_time, request_id, parent_service_name, service_name,
        delay, call_id, parent_call_id
    (the declared 5-col header omits the trailing call_id/parent_call_id —
    a known sdk bug; we read all 7.)
    """
    rows: List[dict] = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 7:
                continue
            try:
                end_time = float(parts[0])
                delay = float(parts[4])
            except ValueError:
                continue
            rows.append({
                "end_time": end_time,
                "request_id": parts[1],
                "parent_service_name": parts[2],
                "service_name": parts[3],
                "delay": delay,
                "call_id": parts[5],
                "parent_call_id": parts[6],
            })
    return rows


def parse_intermediate(req_dir: str) -> Dict[str, Dict[str, List[dict]]]:
    """Parse a fast-service intermediate ``requests`` directory.

    Layout: <req_dir>/<function_name>/{requests.txt,responses.txt}
    Each file is JSONL with ``{"request_id": ..., "content": <json>}``.
    ``content`` is sometimes a json string (str()-wrapped, a known sdk bug)
    and sometimes a dict; we normalize both.
    Returns {function_name: {"requests": [...], "responses": [...]}}.
    """
    out: Dict[str, Dict[str, List[dict]]] = {}
    if not req_dir or not os.path.isdir(req_dir):
        return out
    for fn in sorted(os.listdir(req_dir)):
        d = os.path.join(req_dir, fn)
        if not os.path.isdir(d):
            continue
        entry = {"requests": [], "responses": []}
        for kind in ("requests", "responses"):
            p = os.path.join(d, f"{kind}.txt")
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    content = obj.get("content")
                    obj["_content_parsed"] = _maybe_parse_content(content)
                    entry[kind].append(obj)
        out[fn] = entry
    return out


def _maybe_parse_content(content):
    """fast-service stores response content as str(dict) for non-BaseModel
    returns; try to recover a real json value."""
    if isinstance(content, str):
        s = content.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # str({...}) form: single-quoted python repr — fall back to eval-free
            # heuristic: if it looks like a python dict/list, leave as string.
            return content
    return content


def content_bytes(content) -> int:
    """UTF-8 byte length of an OpenAI-style message content (str / list of
    parts). Ported from agent-benchmark-bin extract_io_size._content_bytes."""
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content.encode("utf-8"))
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, str):
                total += len(part.encode("utf-8"))
            elif isinstance(part, dict):
                t = part.get("type", "")
                if t in ("text", "input_text", "output_text"):
                    total += len((part.get("text") or "").encode("utf-8"))
                elif t == "thinking":
                    total += len((part.get("thinking") or "").encode("utf-8"))
                elif t in ("tool_result", "toolResult"):
                    total += content_bytes(part.get("content", ""))
                elif t in ("tool_use", "toolCall"):
                    inp = part.get("input") or part.get("arguments", "")
                    if isinstance(inp, str):
                        total += len(inp.encode("utf-8"))
                    else:
                        total += len(json.dumps(inp, default=str).encode("utf-8"))
                elif t == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    total += len(url.encode("utf-8"))
        return total
    return 0


def parse_iso_ts(ts_str: str) -> Optional[float]:
    """Parse an ISO-8601 timestamp (with optional trailing Z) to epoch seconds."""
    if not ts_str:
        return None
    s = ts_str.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def read_jsonl(path: str) -> List[dict]:
    out: List[dict] = []
    if not path or not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def read_json(path: str):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None
