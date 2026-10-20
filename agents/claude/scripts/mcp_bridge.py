#!/usr/bin/env python3
"""MCP stdio bridge to mcp-atlas agent-environment HTTP API.

Acts as a standard MCP stdio server that claude CLI can connect to.
Forwards list_tools and call_tool requests to the agent-environment
HTTP service running on AGENT_ENV_URL (default http://localhost:1984).

Per-tool-call timing is recorded to MCP_BRIDGE_LOG (JSONL) when set.

Usage:
    AGENT_ENV_URL=http://localhost:1984 \
    ENABLED_TOOLS='["wikipedia_search_wikipedia","fetch_fetch"]' \
    MCP_BRIDGE_LOG=/path/to/mcp_tool_calls.jsonl \
    python3 scripts/mcp_bridge.py
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error

AGENT_ENV_URL = os.environ.get("AGENT_ENV_URL", "http://localhost:1984")
ENABLED_TOOLS = json.loads(os.environ.get("ENABLED_TOOLS", "[]"))
MCP_BRIDGE_LOG = os.environ.get("MCP_BRIDGE_LOG", "")


def _log_tool_call(name, arguments, is_error, elapsed_ms):
    if not MCP_BRIDGE_LOG:
        return
    record = {
        "timestamp": time.time(),
        "tool_name": name,
        "elapsed_ms": round(elapsed_ms, 2),
        "args_bytes": len(json.dumps(arguments)),
        "is_error": is_error,
    }
    try:
        with open(MCP_BRIDGE_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def http_post(path, body=None):
    url = f"{AGENT_ENV_URL}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"error": e.read().decode()[:500]}
    except Exception as e:
        return {"error": str(e)}


def list_tools():
    all_tools = http_post("/list-tools")
    if isinstance(all_tools, dict) and "error" in all_tools:
        return []
    if ENABLED_TOOLS:
        return [t for t in all_tools if t.get("name") in ENABLED_TOOLS]
    return all_tools


def call_tool(name, arguments):
    result = http_post("/call-tool", {"tool_name": name, "tool_args": arguments})
    return result


def send(msg):
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def main():
    tools_cache = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = msg.get("method", "")
        msg_id = msg.get("id")
        params = msg.get("params", {})

        if method == "initialize":
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "mcp-atlas-bridge", "version": "1.0.0"},
                    },
                }
            )
        elif method == "notifications/initialized":
            pass
        elif method == "tools/list":
            if tools_cache is None:
                tools_cache = list_tools()
            mcp_tools = []
            for t in tools_cache:
                mcp_tools.append(
                    {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "inputSchema": t.get(
                            "inputSchema", t.get("input_schema", {"type": "object"})
                        ),
                    }
                )
            send({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": mcp_tools}})
        elif method == "tools/call":
            name = params.get("name", "")
            arguments = params.get("arguments", {})
            t0 = time.monotonic()
            result = call_tool(name, arguments)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            is_error = bool(isinstance(result, dict) and result.get("isError"))
            _log_tool_call(name, arguments, is_error, elapsed_ms)
            if isinstance(result, dict) and "content" in result:
                content = result["content"]
            elif isinstance(result, dict) and "error" in result:
                content = [{"type": "text", "text": f"Error: {result['error']}"}]
                is_error = True
            else:
                content = [{"type": "text", "text": json.dumps(result)}]
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {"content": content, "isError": is_error},
                }
            )
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": msg_id, "result": {}})
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            )


if __name__ == "__main__":
    main()
