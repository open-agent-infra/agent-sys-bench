#!/usr/bin/env python3
"""Run mcp-atlas tasks through Claude Code CLI.

Reads compatible_tasks.json (tasks that only need no-API-key MCP servers),
starts the agent-environment Docker container, and runs each task through
claude CLI with the MCP bridge server providing tool access.

Usage:
    python3 scripts/run_mcp_atlas.py \
        --tasks ./data/mcp-atlas/compatible_tasks.json \
        --num-tasks 5 \
        --output-dir ./output/mcp-atlas

Prerequisites:
    - agent-environment Docker container running on port 1984
    - ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL set (for DeepSeek)
    - claude CLI installed
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level="INFO", format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s"
)
log = logging.getLogger("mcp-atlas-runner")

SCRIPT_DIR = Path(__file__).parent.resolve()


def ensure_agent_env_running(port=1984):
    """Check if agent-environment is running, start if not."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=5) as r:
            if r.status == 200:
                log.info("agent-environment already running on port %d", port)
                return True
    except Exception:
        pass

    log.info("Starting agent-environment container on port %d...", port)
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            "mcp-atlas-agent-env",
            "-p",
            f"{port}:1984",
            "agent-environment:latest",
        ],
        check=True,
        capture_output=True,
    )
    # Wait for it to be ready
    for i in range(60):
        time.sleep(2)
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=5
            ) as r:
                if r.status == 200:
                    log.info("agent-environment ready after %ds", (i + 1) * 2)
                    return True
        except Exception:
            pass
    raise RuntimeError("agent-environment failed to start within 120s")


def _cwd_to_project_dir(cwd):
    encoded = cwd.replace("/", "-")
    home = os.environ.get("HOME", os.path.expanduser("~"))
    return Path(home) / ".claude" / "projects" / encoded


def run_task(
    task, model, output_dir, agent_env_port=1984, timeout=300, agent_env_url=None
):
    """Run a single mcp-atlas task through claude CLI with telemetry."""
    task_id = task["task_id"]
    prompt = task["prompt"]
    enabled_tools = task["enabled_tools"]

    # Resolve to absolute path before changing cwd (compose runs from /workspace/claude)
    output_dir = str(Path(output_dir).resolve())
    task_dir = Path(output_dir) / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    mcp_config = {
        "mcpServers": {
            "mcp-atlas": {
                "command": sys.executable,
                "args": [str(SCRIPT_DIR / "mcp_bridge.py")],
                "env": {
                    "AGENT_ENV_URL": agent_env_url
                    or f"http://localhost:{agent_env_port}",
                    "ENABLED_TOOLS": json.dumps(enabled_tools),
                    "MCP_BRIDGE_LOG": str(task_dir / "mcp_tool_calls.jsonl"),
                },
            }
        }
    }
    mcp_config_path = task_dir / "mcp_config.json"
    mcp_config_path.write_text(json.dumps(mcp_config, indent=2))

    cmd = [
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "bypassPermissions",
        "--model",
        model,
        "--mcp-config",
        str(mcp_config_path),
    ]

    prompt_path = task_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    stream_path = task_dir / "agent.stream.jsonl"
    stdout_path = task_dir / "agent.stdout.log"
    stderr_path = task_dir / "agent.stderr.log"

    log.info(
        "[%s] launching claude (tools=%d, timeout=%ds)",
        task_id,
        len(enabled_tools),
        timeout,
    )
    t0 = time.perf_counter()
    timed_out = False
    exit_code = None
    error = None
    session_id = None
    result_event = None

    try:
        task_cwd = os.environ.get("HOME", os.path.expanduser("~"))
        with open(prompt_path) as fi, open(stderr_path, "w") as fe:
            proc = subprocess.Popen(
                cmd,
                stdin=fi,
                stdout=subprocess.PIPE,
                stderr=fe,
                cwd=task_cwd,
            )

        deadline = time.monotonic() + timeout + 60
        with open(stream_path, "w") as fs:
            for raw_line in proc.stdout:
                if time.monotonic() > deadline:
                    proc.kill()
                    timed_out = True
                    break
                line_str = raw_line.decode("utf-8", errors="replace")
                fs.write(line_str)
                line_str = line_str.strip()
                if not line_str:
                    continue
                try:
                    evt = json.loads(line_str)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "system" and evt.get("subtype") == "init":
                    session_id = evt.get("session_id")
                elif evt.get("type") == "result":
                    result_event = evt

        if not timed_out:
            proc.wait(timeout=30)
            exit_code = proc.returncode

    except subprocess.TimeoutExpired:
        proc.kill()
        timed_out = True
        log.warning("[%s] timed out after %ds", task_id, timeout)
    except Exception as e:
        error = str(e)
        log.exception("[%s] error: %s", task_id, e)

    wall = time.perf_counter() - t0

    # Write plain text output for backward compatibility
    final_text = result_event.get("result", "") if result_event else ""
    stdout_path.write_text(final_text, encoding="utf-8")

    # Copy session JSONL to task output dir
    session_jsonl_path = None
    if session_id:
        proj_dir = _cwd_to_project_dir(task_cwd)
        src = proj_dir / f"{session_id}.jsonl"
        if src.exists():
            dst = task_dir / "session.jsonl"
            shutil.copy2(src, dst)
            session_jsonl_path = str(dst)
            log.info("[%s] session JSONL: %d bytes", task_id, dst.stat().st_size)

    # Extract telemetry from result event
    duration_api_ms = None
    num_turns = None
    if result_event:
        duration_api_ms = result_event.get("duration_api_ms")
        num_turns = result_event.get("num_turns")

    result = {
        "task_id": task_id,
        "exit_code": exit_code,
        "wall_s": round(wall, 2),
        "timed_out": timed_out,
        "error": error,
        "servers": task["servers"],
        "num_tools": len(enabled_tools),
        "session_id": session_id,
        "session_jsonl_path": session_jsonl_path,
        "duration_api_ms": duration_api_ms,
        "num_turns": num_turns,
    }
    (task_dir / "result.json").write_text(json.dumps(result, indent=2))

    # Postprocess session JSONL into latency breakdown CSVs
    if session_jsonl_path:
        try:
            import importlib.util

            _spec = importlib.util.spec_from_file_location(
                "postprocess",
                str(SCRIPT_DIR.parent / "src" / "agsbench_claude" / "postprocess.py"),
            )
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _mod.postprocess(
                session_jsonl_path=Path(session_jsonl_path),
                output_dir=task_dir,
                task_id=task_id,
                mcp_tool_log_path=task_dir / "mcp_tool_calls.jsonl",
            )
        except Exception as e:
            log.warning("[%s] postprocess failed: %s", task_id, e)

    log.info(
        "[%s] exit=%s wall=%.1fs timed_out=%s turns=%s api_ms=%s",
        task_id,
        exit_code,
        wall,
        timed_out,
        num_turns,
        duration_api_ms,
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Run mcp-atlas tasks through claude")
    parser.add_argument("--tasks", required=True, help="Path to compatible_tasks.json")
    parser.add_argument(
        "--num-tasks", type=int, default=5, help="Number of tasks to run"
    )
    parser.add_argument(
        "--output-dir", default="./output/mcp-atlas", help="Output directory"
    )
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model name")
    parser.add_argument(
        "--timeout", type=int, default=300, help="Per-task timeout (seconds)"
    )
    parser.add_argument(
        "--agent-env-port", type=int, default=1984, help="agent-environment port"
    )
    parser.add_argument(
        "--agent-env-url", default=None, help="agent-environment URL (overrides port)"
    )
    args = parser.parse_args()

    with open(args.tasks) as f:
        all_tasks = json.load(f)

    tasks = all_tasks[: args.num_tasks]
    log.info("Running %d/%d compatible tasks", len(tasks), len(all_tasks))

    ensure_agent_env_running(args.agent_env_port)

    results = []
    for i, task in enumerate(tasks):
        log.info("=== Task %d/%d: %s ===", i + 1, len(tasks), task["task_id"])
        result = run_task(
            task,
            args.model,
            args.output_dir,
            args.agent_env_port,
            args.timeout,
            agent_env_url=args.agent_env_url,
        )
        results.append(result)

    # Summary
    success = sum(1 for r in results if r["exit_code"] == 0)
    timeout_count = sum(1 for r in results if r["timed_out"])
    failed = len(results) - success - timeout_count
    avg_wall = sum(r["wall_s"] for r in results) / len(results) if results else 0

    summary = {
        "total": len(results),
        "success": success,
        "timeout": timeout_count,
        "failed": failed,
        "avg_wall_s": round(avg_wall, 2),
        "results": results,
    }
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info(
        "=== Summary: %d/%d success, %d timeout, %d failed, avg %.1fs ===",
        success,
        len(results),
        timeout_count,
        failed,
        avg_wall,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
