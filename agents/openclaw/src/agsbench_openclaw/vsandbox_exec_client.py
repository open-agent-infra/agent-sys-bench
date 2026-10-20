#!/usr/bin/env python3
"""vsandbox exec client — spawned by the openclaw vsandbox plugin.

Two modes:
  exec (default):  Run a command inside the sandbox container.
  shell:           Run a script inside the sandbox container.

The client communicates with vsandbox_service RPCs over HTTP.
Uses only stdlib (no third-party deps like requests).

Usage:
  # Exec mode (buildExecSpec)
  vsandbox_exec_client.py --endpoint URL --sandbox-id ID --command "echo hi"

  # Shell mode (runShellCommand)
  vsandbox_exec_client.py --run-shell --endpoint URL --sandbox-id ID \
      --script "#!/bin/bash\necho hello" [-- arg1 arg2]

Timing:
  When VSANDBOX_TIMING_CSV is set, each RPC call appends a row:
    timestamp, request_id, invoker, service_name, delay_s, call_id, parent_call_id
  This matches the fast-service remote_delay.csv format for easy correlation
  with the server-side e2e_delay.csv.
"""

import argparse
import base64
import csv
import json
import os
import sys
import time
import urllib.request
import urllib.error
import uuid


def _write_timing(service_name: str, delay: float, sandbox_id: str):
    """Append a timing row to the CSV file specified by VSANDBOX_TIMING_CSV."""
    csv_path = os.environ.get("VSANDBOX_TIMING_CSV", "")
    if not csv_path:
        return
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [
                    time.time(),  # timestamp
                    sandbox_id,  # request_id (group by task)
                    "vsandbox_exec_client",  # invoker_name
                    service_name,  # remote service name
                    delay,  # HTTP round-trip delay (seconds)
                    str(uuid.uuid4()),  # call_id
                    "None",  # parent_call_id
                ]
            )
    except OSError:
        pass  # best-effort — don't crash on write failure


def _post_rpc(
    endpoint: str,
    module: str,
    func: str,
    payload: dict,
    timeout: int = 300,
    sandbox_id: str = "",
) -> dict:
    """Call a fast-service RPC endpoint using stdlib urllib."""
    url = f"{endpoint}/{module}.{func}"
    service_name = f"{module}.{func}"
    data = json.dumps({"req": payload}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        delay = time.time() - t0
        _write_timing(service_name, delay, sandbox_id)
        return result
    except urllib.error.HTTPError as e:
        delay = time.time() - t0
        _write_timing(service_name, delay, sandbox_id)
        body = e.read().decode("utf-8", errors="replace")[:500]
        print(f"vsandbox RPC HTTP {e.code} error: {body}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        delay = time.time() - t0
        _write_timing(service_name, delay, sandbox_id)
        print(f"vsandbox RPC error: {e}", file=sys.stderr)
        sys.exit(1)


def exec_mode(args):
    """Execute a command (buildExecSpec path)."""
    # Read stdin if available and not a tty.
    # Use select() to avoid blocking on pipes that stay open (stdinMode: "pipe-open").
    stdin_b64 = ""
    if not sys.stdin.isatty():
        import select

        # Check if stdin has data ready within 50ms
        ready, _, _ = select.select([sys.stdin.buffer], [], [], 0.05)
        if ready:
            stdin_data = sys.stdin.buffer.read()
            if stdin_data:
                stdin_b64 = base64.b64encode(stdin_data).decode()

    # Build env dict from --env KEY=VAL pairs
    env = {}
    if args.env:
        for e in args.env:
            if "=" in e:
                k, v = e.split("=", 1)
                env[k] = v

    payload = {
        "sandbox_id": args.sandbox_id,
        "command": args.command,
        "workdir": args.workdir or "/workspace",
        "env": env,
        "stdin_b64": stdin_b64,
        "timeout": args.timeout,
    }

    result = _post_rpc(
        args.endpoint,
        "agsbench_openclaw.vsandbox_service",
        "vsandbox_exec",
        payload,
        timeout=args.timeout,
        sandbox_id=args.sandbox_id,
    )

    if result.get("error"):
        print(f"vsandbox_exec error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    # Write stdout/stderr
    if result.get("stdout_b64"):
        sys.stdout.buffer.write(base64.b64decode(result["stdout_b64"]))
        sys.stdout.buffer.flush()
    if result.get("stderr_b64"):
        sys.stderr.buffer.write(base64.b64decode(result["stderr_b64"]))
        sys.stderr.buffer.flush()

    sys.exit(result.get("exit_code", 0))


def shell_mode(args):
    """Execute a script (runShellCommand path)."""
    # Read stdin if available and not a tty.
    # Use select() to avoid blocking on pipes that stay open.
    stdin_b64 = ""
    if not sys.stdin.isatty():
        import select

        ready, _, _ = select.select([sys.stdin.buffer], [], [], 0.05)
        if ready:
            stdin_data = sys.stdin.buffer.read()
            if stdin_data:
                stdin_b64 = base64.b64encode(stdin_data).decode()

    # Script content: from --script arg
    script = args.script
    if not script:
        print("--script is required in shell mode", file=sys.stderr)
        sys.exit(1)

    payload = {
        "sandbox_id": args.sandbox_id,
        "script": script,
        "args": args.script_args or [],
        "stdin_b64": stdin_b64,
        "timeout": args.timeout,
    }

    result = _post_rpc(
        args.endpoint,
        "agsbench_openclaw.vsandbox_service",
        "vsandbox_shell",
        payload,
        timeout=args.timeout,
        sandbox_id=args.sandbox_id,
    )

    if result.get("error"):
        print(f"vsandbox_shell error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    # Write stdout/stderr
    if result.get("stdout_b64"):
        sys.stdout.buffer.write(base64.b64decode(result["stdout_b64"]))
        sys.stdout.buffer.flush()
    if result.get("stderr_b64"):
        sys.stderr.buffer.write(base64.b64decode(result["stderr_b64"]))
        sys.stderr.buffer.flush()

    sys.exit(result.get("code", 0))


def main():
    parser = argparse.ArgumentParser(description="vsandbox exec client")
    parser.add_argument("--endpoint", required=True, help="vsandbox service URL")
    parser.add_argument("--sandbox-id", required=True, help="Sandbox container ID")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")

    # Mode selection
    parser.add_argument("--run-shell", action="store_true", help="Shell mode")

    # Exec mode args
    parser.add_argument("--command", help="Command to execute (exec mode)")
    parser.add_argument("--workdir", default="/workspace", help="Working directory")
    parser.add_argument("--env", action="append", help="Environment variable KEY=VAL")

    # Shell mode args
    parser.add_argument("--script", help="Script content (shell mode)")
    parser.add_argument("script_args", nargs="*", help="Script arguments")

    args = parser.parse_args()

    if args.run_shell:
        shell_mode(args)
    else:
        if not args.command:
            parser.error("--command is required in exec mode")
        exec_mode(args)


if __name__ == "__main__":
    main()
