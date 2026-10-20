"""Workload driver: run one WildClawBench task through openclaw, with
fast-service instrumentation on the load/launch/collect boundaries.

Mirrors WildClawBench's `eval/run_batch.py:run_single_task` but uses
openclaw's `--local` embedded mode. Supports both host-direct execution
(no sandbox) and Docker-sandboxed tool execution via the `sandbox` flag.
"""

# DO NOT add `from __future__ import annotations` -- fast-service needs class
# objects in __annotations__, not strings, to detect RequestContext params.

import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel

from fast_service import FastServiceManager, RequestContext

OCLAW_RUN_FSM = FastServiceManager()
log = logging.getLogger("agsbench_openclaw.run_task")


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class TaskSpec(BaseModel):
    task_id: str
    name: str
    category: str
    timeout_seconds: int
    modality: str
    prompt: str
    task_md_path: str
    sandbox_files: list = []

    class Config:
        arbitrary_types_allowed = True


class RunResult(BaseModel):
    task_id: str
    run_id: str
    session_id: Optional[str] = None
    chat_jsonl_path: Optional[str] = None
    results_md_path: Optional[str] = None
    agent_stdout_path: str
    agent_stderr_path: str
    exit_code: Optional[int] = None
    wall_s: float
    timed_out: bool
    error: Optional[str] = None
    output_dir: str

    class Config:
        arbitrary_types_allowed = True


class RunTaskRequest(BaseModel):
    task_md_path: str
    model: str = "deepseek-instrumented/deepseek-v4-flash"
    output_root: str = "./output"
    timeout_override_s: int = 0
    thinking_level: str = ""
    sandbox: bool = False
    sandbox_mode: str = ""
    sandbox_workspace: str = ""
    sessions_root: str = ""
    proxy_audit_jsonl: str = ""

    class Config:
        arbitrary_types_allowed = True


class RunTaskResponse(BaseModel):
    task_id: str
    run_id: str
    exit_code: Optional[int] = None
    wall_s: float
    total_wall_s: float = 0.0
    timed_out: bool
    error: Optional[str] = None
    results_md_produced: bool = False
    sandbox: bool = False
    chat_jsonl_path: Optional[str] = None
    output_dir: str

    class Config:
        arbitrary_types_allowed = True


# ----------------------------------------------------------------------------
# Task format adapters
# ----------------------------------------------------------------------------
def _load_task_yaml(task_path: Path, text: str) -> "TaskSpec":
    """Parse a claw-eval task.yaml file."""
    data = yaml.safe_load(text)
    prompt = data.get("prompt", {})
    prompt_text = prompt.get("text", "") if isinstance(prompt, dict) else str(prompt)
    env_cfg = data.get("environment", {})

    sandbox_files = [
        {"src": str(task_path.parent / f), "dest": f"/workspace/{f}"}
        for f in data.get("sandbox_files", [])
    ]

    return TaskSpec(
        task_id=str(data.get("task_id") or task_path.stem),
        name=str(data.get("task_name") or data.get("name", "")),
        category=str(data.get("category") or ""),
        timeout_seconds=int(env_cfg.get("timeout_seconds", 600)),
        modality="pure-text",
        prompt=prompt_text.strip(),
        task_md_path=str(task_path),
        sandbox_files=sandbox_files,
    )


def _normalize_workspace_files(workspace_files: list, task_path: Path) -> list:
    """Normalize pinchbench workspace_files to unified sandbox_files format."""
    if not workspace_files:
        return []
    assets_dir = task_path.parent.parent / "assets"
    result = []
    for fspec in workspace_files:
        if isinstance(fspec, dict) and "content" in fspec:
            result.append(
                {
                    "content": fspec["content"],
                    "dest": f"/workspace/{fspec['path']}",
                }
            )
        elif isinstance(fspec, dict) and "source" in fspec:
            result.append(
                {
                    "src": str(assets_dir / fspec["source"]),
                    "dest": f"/workspace/{fspec['dest']}",
                }
            )
    return result


# ----------------------------------------------------------------------------
# Instrumented functions
# ----------------------------------------------------------------------------
@OCLAW_RUN_FSM.fast_service
def load_task(task_md_path: str, fsm_context: RequestContext = None) -> TaskSpec:
    """Parse a task file. Supports WildClawBench .md, pinchbench .md, and claw-eval .yaml."""
    task_path = Path(task_md_path)
    text = task_path.read_text(encoding="utf-8")

    if task_path.suffix in (".yaml", ".yml"):
        return _load_task_yaml(task_path, text)

    # WildClawBench / pinchbench .md format: YAML frontmatter + ## Prompt body
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        raise ValueError(f"Task file missing YAML frontmatter: {task_md_path}")
    meta = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)

    pm = re.search(r"##\s+Prompt\s*\n(.*?)(?=^##\s|\Z)", body, re.DOTALL | re.MULTILINE)
    prompt = pm.group(1).strip() if pm else body.strip()

    sandbox_files = _normalize_workspace_files(
        meta.get("workspace_files", []), task_path
    )

    return TaskSpec(
        task_id=str(meta.get("id") or task_path.stem),
        name=str(meta.get("name") or ""),
        category=str(meta.get("category") or ""),
        timeout_seconds=int(meta.get("timeout_seconds") or 600),
        modality=str(meta.get("modality") or "pure-text"),
        prompt=prompt,
        task_md_path=str(task_md_path),
        sandbox_files=sandbox_files,
    )


@OCLAW_RUN_FSM.fast_service
def prepare_workspace(
    output_root: str,
    task_id: str,
    sandbox: bool = False,
    sandbox_workspace: str = "",
    fsm_context: RequestContext = None,
) -> str:
    """Create the per-run output dir and prepare the results directory.

    When sandbox=False (default): creates <output_dir>/task_workspace/results/
    as a host-local substitute for /tmp_workspace/results.

    When sandbox=True: clears and recreates <sandbox_workspace>/results/ so
    the Docker container starts with a clean workspace. The output_dir is
    still created for logs and collected artifacts.
    """
    run_id = uuid.uuid4().hex[:6]
    out = Path(output_root) / task_id / run_id
    out.mkdir(parents=True, exist_ok=True)

    if sandbox and sandbox_workspace:
        results_dir = Path(sandbox_workspace) / "results"
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
    else:
        (out / "task_workspace" / "results").mkdir(parents=True, exist_ok=True)

    return str(out)


@OCLAW_RUN_FSM.fast_service
def launch_openclaw(
    task: TaskSpec,
    model: str,
    output_dir: str,
    timeout_override_s: int = 0,
    thinking_level: str = "",
    sandbox: bool = False,
    agent_id: str = "",
    fsm_context: RequestContext = None,
) -> RunResult:
    """Spawn `openclaw agent --local --agent <agent_id> --model ... --message ...`.

    When sandbox=False (default): uses --agent main, rewrites /tmp_workspace/results
    to a local host path.

    When sandbox=True: uses --agent main-sandbox, rewrites /tmp_workspace/results
    to /workspace/results (the container-internal mount path).
    """
    out = Path(output_dir)
    run_id = out.name

    system_prompt = (
        "You are an expert in a restricted, non-interactive environment. "
        f"Solve the task efficiently before the timeout ({task.timeout_seconds}s). "
        "Run all processes in the foreground without user input or background services. "
        "Provide a complete, functional solution in a single pass with no placeholders.\n"
    )

    if sandbox:
        sandbox_results_path = "/workspace/results"
    else:
        sandbox_results_path = str(out / "task_workspace" / "results")

    rewritten_prompt = task.prompt.replace(
        "/tmp_workspace/results", sandbox_results_path
    )
    results_suffix = ""
    if "results/results.md" not in rewritten_prompt:
        results_suffix = (
            "\n\nWhen you are done, write a concise summary of your work "
            f"and final answer to {sandbox_results_path}/results.md."
        )
    full_prompt = system_prompt + rewritten_prompt + results_suffix

    timeout = timeout_override_s if timeout_override_s > 0 else task.timeout_seconds

    session_id = f"oclaw-poc-{run_id}"
    if not agent_id:
        agent_id = "main-sandbox" if sandbox else "main"

    cmd = [
        "openclaw",
        "agent",
        "--local",
        "--agent",
        agent_id,
        "--session-id",
        session_id,
        "--model",
        model,
        "--timeout",
        str(timeout),
        "--message",
        full_prompt,
    ]
    if thinking_level:
        cmd[-2:-2] = ["--thinking", thinking_level]

    stdout_path = out / "agent.stdout.log"
    stderr_path = out / "agent.stderr.log"
    log.info(
        "[%s] launching openclaw (agent=%s, timeout=%ds)",
        task.task_id,
        agent_id,
        timeout,
    )
    t0 = time.perf_counter()
    timed_out = False
    err = None
    try:
        with open(stdout_path, "w") as fo, open(stderr_path, "w") as fe:
            proc = subprocess.run(
                cmd,
                stdout=fo,
                stderr=fe,
                timeout=timeout + 60,
            )
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        exit_code = None
        timed_out = True
        log.warning("[%s] subprocess timed out", task.task_id)
    except Exception as exc:
        exit_code = None
        err = str(exc)
        log.exception("[%s] launch error: %s", task.task_id, exc)
    wall = time.perf_counter() - t0

    return RunResult(
        task_id=task.task_id,
        run_id=run_id,
        session_id=session_id,
        agent_stdout_path=str(stdout_path),
        agent_stderr_path=str(stderr_path),
        exit_code=exit_code,
        wall_s=wall,
        timed_out=timed_out,
        error=err,
        output_dir=str(out),
    )


def _export_sqlite_transcript(sessions_root: str, session_id: str, target: Path) -> bool:
    """openclaw >= 2026.9 keeps transcripts in <agent>/agent/openclaw-agent.sqlite
    (table transcript_events) instead of <agent>/sessions/<id>.jsonl. Each
    event_json row is the same object the old jsonl had per line, so dump them
    in seq order to reproduce chat.jsonl."""
    db = Path(sessions_root).parent / "agent" / "openclaw-agent.sqlite"
    if not db.exists():
        return False
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT event_json FROM transcript_events WHERE session_id = ? ORDER BY seq",
                (session_id,),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error as e:
        log.warning("transcript export from %s failed: %s", db, e)
        return False
    if not rows:
        return False
    target.write_text("".join(r[0] + "\n" for r in rows))
    return True


@OCLAW_RUN_FSM.fast_service
def collect_outputs(
    run: RunResult,
    sessions_root: str,
    results_md_path: str,
    fsm_context: RequestContext = None,
) -> RunResult:
    """Copy the chat.jsonl from agsbench_openclaw's sessions dir + the produced
    results.md into the run's output_dir.
    """
    out = Path(run.output_dir)
    src_jsonl = None
    if run.session_id:
        candidate = Path(sessions_root) / f"{run.session_id}.jsonl"
        if candidate.exists():
            src_jsonl = candidate
    if src_jsonl is None and run.session_id:
        target = out / "chat.jsonl"
        if _export_sqlite_transcript(sessions_root, run.session_id, target):
            run.chat_jsonl_path = str(target)
    if src_jsonl is None and run.chat_jsonl_path is None:
        candidates = sorted(
            (
                c
                for c in Path(sessions_root).glob("*.jsonl")
                if "trajectory" not in c.name
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            src_jsonl = candidates[0]
    if src_jsonl is not None:
        target = out / "chat.jsonl"
        shutil.copy2(src_jsonl, target)
        run.chat_jsonl_path = str(target)
    # results.md (if produced by the task)
    src = Path(results_md_path)
    if src.exists():
        target = out / "results.md"
        if src.resolve() != target.resolve():
            shutil.copy2(src, target)
        run.results_md_path = str(target)
    # write a summary
    (out / "run_summary.json").write_text(
        json.dumps(
            {
                "task_id": run.task_id,
                "run_id": run.run_id,
                "session_id": run.session_id,
                "exit_code": run.exit_code,
                "wall_s": run.wall_s,
                "timed_out": run.timed_out,
                "error": run.error,
                "chat_jsonl_path": run.chat_jsonl_path,
                "results_md_path": run.results_md_path,
            },
            indent=2,
        )
    )
    return run


# ----------------------------------------------------------------------------
# Top-level RPC endpoint (used by docker-compose loadgen)
# ----------------------------------------------------------------------------
def _configure_ssh_sandbox(sandbox_resp) -> str:
    """Write the SSH private key to a temp file and patch openclaw config
    with the ssh-sandbox agent definition. Returns the temp key path."""
    import stat

    key_fd, key_path = tempfile.mkstemp(prefix="oclaw-ssh-", suffix=".key")
    with os.fdopen(key_fd, "w") as f:
        f.write(sandbox_resp.private_key_data)
    os.chmod(key_path, stat.S_IRUSR)

    proxy_host = os.environ.get("OPENCLAW_SSH_PROXY_HOST", "")

    ssh_wrapper_path = ""
    if proxy_host:
        # Route via ProxyJump: connect through proxy_host, then to localhost:port.
        # Create a wrapper script that injects -J into the ssh command, since
        # openclaw's SSH config generation doesn't support ProxyJump.
        ssh_target = f"{sandbox_resp.user}@localhost:{sandbox_resp.port}"
        proxy_key = os.environ.get("OPENCLAW_SSH_PROXY_KEY", "/root/.ssh/id_rsa")
        ssh_fd, ssh_wrapper_path = tempfile.mkstemp(
            prefix="oclaw-ssh-wrapper-", suffix=".sh"
        )
        with os.fdopen(ssh_fd, "w") as f:
            f.write("#!/bin/sh\n")
            f.write(
                f'exec ssh -o "ProxyCommand=ssh -F /dev/null -i {proxy_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -W %h:%p {proxy_host}" "$@"\n'
            )
        os.chmod(ssh_wrapper_path, stat.S_IRWXU)
    else:
        ssh_target = f"{sandbox_resp.user}@{sandbox_resp.host}:{sandbox_resp.port}"

    ssh_opts = {
        "target": ssh_target,
        "identityFile": key_path,
        "strictHostKeyChecking": False,
        "workspaceRoot": "/workspace",
    }
    if ssh_wrapper_path:
        ssh_opts["command"] = ssh_wrapper_path

    patch = {
        "agents": {
            "list": [
                {"id": "main"},
                {
                    "id": "main-ssh-sandbox",
                    "name": "SSH Sandboxed Agent",
                    "sandbox": {
                        "mode": "all",
                        "scope": "session",
                        "backend": "ssh",
                        "workspaceAccess": "rw",
                        "ssh": ssh_opts,
                    },
                },
            ]
        }
    }
    proc = subprocess.run(
        ["openclaw", "config", "patch", "--stdin"],
        input=json.dumps(patch),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        log.error("openclaw config patch (ssh-sandbox) failed: %s", proc.stderr)
        raise RuntimeError(f"openclaw config patch failed: {proc.stderr}")

    log.info(
        "SSH sandbox configured: target=%s (key=%s, proxy=%s)",
        ssh_target,
        key_path,
        proxy_host or "none",
    )
    return key_path, ssh_wrapper_path


def _scp_results_from_sandbox(
    sandbox_resp, key_path: str, ssh_wrapper_path: str, output_dir: str
) -> Optional[str]:
    """SCP results.md from the SSH sandbox container back to the local output dir."""
    proxy_host = os.environ.get("OPENCLAW_SSH_PROXY_HOST", "")
    if proxy_host:
        remote_path = f"{sandbox_resp.user}@localhost:/workspace/results/results.md"
    else:
        remote_path = (
            f"{sandbox_resp.user}@{sandbox_resp.host}:/workspace/results/results.md"
        )
    local_path = str(Path(output_dir) / "results.md")
    try:
        cmd = [
            "scp",
            "-P",
            str(sandbox_resp.port),
            "-i",
            key_path,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
        ]
        if proxy_host:
            proxy_key = os.environ.get("OPENCLAW_SSH_PROXY_KEY", "/root/.ssh/id_rsa")
            cmd.extend(
                [
                    "-o",
                    f"ProxyCommand=ssh -F /dev/null -i {proxy_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -W %h:%p {proxy_host}",
                ]
            )
        cmd.extend([remote_path, local_path])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode == 0 and Path(local_path).exists():
            log.info("SCP results.md from sandbox: %s", local_path)
            return local_path
        else:
            log.warning("SCP exit=%d stderr=%s", proc.returncode, proc.stderr[:200])
    except Exception as e:
        log.warning("SCP results.md failed: %s", e)
    return None


def _build_ssh_cmd(sandbox_resp, key_path: str, ssh_wrapper_path: str):
    """Build common SSH/SCP options for sandbox communication."""
    proxy_host = os.environ.get("OPENCLAW_SSH_PROXY_HOST", "")
    opts = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"]
    if proxy_host:
        proxy_key = os.environ.get("OPENCLAW_SSH_PROXY_KEY", "/root/.ssh/id_rsa")
        opts.extend(
            [
                "-o",
                f"ProxyCommand=ssh -F /dev/null -i {proxy_key} "
                f"-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
                f"-W %h:%p {proxy_host}",
            ]
        )
        host = "localhost"
    else:
        host = sandbox_resp.host
    return opts, host


def _ssh_exec(sandbox_resp, key_path: str, ssh_wrapper_path: str, command: str):
    """Run a command inside the SSH sandbox via ssh."""
    opts, host = _build_ssh_cmd(sandbox_resp, key_path, ssh_wrapper_path)
    cmd = (
        ["ssh", "-p", str(sandbox_resp.port), "-i", key_path]
        + opts
        + [f"{sandbox_resp.user}@{host}", command]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        log.warning("ssh exec failed (rc=%d): %s", proc.returncode, proc.stderr[:200])
    return proc


def _scp_files_to_sandbox(
    sandbox_resp, key_path: str, ssh_wrapper_path: str, sandbox_files: list
):
    """SCP fixture files into the SSH sandbox before agent launch."""
    if not sandbox_files:
        return
    opts, host = _build_ssh_cmd(sandbox_resp, key_path, ssh_wrapper_path)
    remote_prefix = f"{sandbox_resp.user}@{host}"

    for fspec in sandbox_files:
        dest = fspec["dest"]
        parent_dir = str(Path(dest).parent)
        _ssh_exec(sandbox_resp, key_path, ssh_wrapper_path, f"mkdir -p {parent_dir}")

        tmp_path = None
        if "content" in fspec:
            fd, tmp_path = tempfile.mkstemp(prefix="oclaw-fixture-")
            with os.fdopen(fd, "w") as f:
                f.write(fspec["content"])
            src_path = tmp_path
        else:
            src_path = fspec["src"]

        try:
            cmd = (
                ["scp", "-P", str(sandbox_resp.port), "-i", key_path]
                + opts
                + [src_path, f"{remote_prefix}:{dest}"]
            )
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                log.error(
                    "SCP fixture %s → %s failed: %s", src_path, dest, proc.stderr[:200]
                )
                raise RuntimeError(
                    f"Failed to stage fixture {dest}: {proc.stderr[:200]}"
                )
            log.info("Staged fixture → %s", dest)
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass


# Module-level handle for the daemon process so the finally block can kill it.
_vsandbox_daemon_proc: subprocess.Popen | None = None
_VSANDBOX_DAEMON_PORT = 18040


def _start_vsandbox_daemon(
    vsandbox_endpoint: str, timing_csv: str = ""
) -> subprocess.Popen:
    """Start the vsandbox client daemon for HTTP keep-alive connection reuse."""
    global _vsandbox_daemon_proc
    daemon_script = os.path.join(os.path.dirname(__file__), "vsandbox_client_daemon.py")
    if not os.path.exists(daemon_script):
        log.warning(
            "vsandbox_client_daemon.py not found at %s, skipping daemon", daemon_script
        )
        return None
    env = os.environ.copy()
    if timing_csv:
        env["VSANDBOX_TIMING_CSV"] = timing_csv
    proc = subprocess.Popen(
        [
            "python3",
            daemon_script,
            "--endpoint",
            vsandbox_endpoint,
            "--port",
            str(_VSANDBOX_DAEMON_PORT),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    # Wait briefly for the daemon to start listening
    import time as _time

    _time.sleep(0.3)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode(errors="replace")
        log.error("vsandbox daemon failed to start: %s", stderr)
        return None
    os.environ["VSANDBOX_DAEMON_PORT"] = str(_VSANDBOX_DAEMON_PORT)
    _vsandbox_daemon_proc = proc
    log.info(
        "vsandbox daemon started on port %d (pid=%d)", _VSANDBOX_DAEMON_PORT, proc.pid
    )
    return proc


def _stop_vsandbox_daemon():
    """Stop the vsandbox client daemon."""
    global _vsandbox_daemon_proc
    if _vsandbox_daemon_proc is not None:
        try:
            _vsandbox_daemon_proc.terminate()
            _vsandbox_daemon_proc.wait(timeout=5)
        except Exception:
            _vsandbox_daemon_proc.kill()
        log.info("vsandbox daemon stopped (pid=%d)", _vsandbox_daemon_proc.pid)
        _vsandbox_daemon_proc = None
    os.environ.pop("VSANDBOX_DAEMON_PORT", None)


def _configure_vsandbox(sandbox_resp, vsandbox_endpoint: str):
    """Patch openclaw config with the vsandbox agent definition.

    Sets environment variables for the vsandbox plugin to read, and patches
    the openclaw agent list to include a vsandbox-backed agent.
    Also starts the vsandbox client daemon for connection reuse.
    """
    os.environ["VSANDBOX_ENDPOINT"] = vsandbox_endpoint
    os.environ["VSANDBOX_SANDBOX_ID"] = sandbox_resp.sandbox_id

    patch = {
        "agents": {
            "list": [
                {"id": "main"},
                {
                    "id": "main-vsandbox",
                    "name": "VSandbox Agent",
                    "sandbox": {
                        "mode": "all",
                        "scope": "session",
                        "backend": "vsandbox",
                        "workspaceAccess": "rw",
                        "workspaceRoot": "/workspace",
                    },
                },
            ]
        }
    }
    proc = subprocess.run(
        ["openclaw", "config", "patch", "--stdin"],
        input=json.dumps(patch),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        log.error("openclaw config patch (vsandbox) failed: %s", proc.stderr)
        raise RuntimeError(f"openclaw config patch failed: {proc.stderr}")

    log.info(
        "VSandbox configured: sandbox_id=%s, endpoint=%s",
        sandbox_resp.sandbox_id,
        vsandbox_endpoint,
    )


def _vsandbox_put_files(vsandbox_endpoint: str, sandbox_id: str, sandbox_files: list):
    """Stage fixture files into the sandbox container via vsandbox_put_file RPC."""
    import base64
    import requests

    for fspec in sandbox_files:
        dest = fspec["dest"]
        if "content" in fspec:
            content_bytes = fspec["content"].encode("utf-8")
        else:
            with open(fspec["src"], "rb") as f:
                content_bytes = f.read()

        payload = {
            "sandbox_id": sandbox_id,
            "dest_path": dest,
            "content_b64": base64.b64encode(content_bytes).decode(),
        }
        url = f"{vsandbox_endpoint}/agsbench_openclaw.vsandbox_service.vsandbox_put_file"
        resp = requests.post(url, json={"req": payload}, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if not result.get("ok", False):
            raise RuntimeError(
                f"vsandbox_put_file failed for {dest}: {result.get('error', 'unknown')}"
            )
        log.info("Staged fixture via vsandbox → %s", dest)


def _vsandbox_get_file(
    vsandbox_endpoint: str, sandbox_id: str, src_path: str, local_path: str
) -> bool:
    """Download a file from sandbox container via vsandbox_get_file RPC."""
    import base64
    import requests

    payload = {
        "sandbox_id": sandbox_id,
        "src_path": src_path,
    }
    url = f"{vsandbox_endpoint}/agsbench_openclaw.vsandbox_service.vsandbox_get_file"
    try:
        resp = requests.post(url, json={"req": payload}, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        if result.get("error"):
            log.warning("vsandbox_get_file %s: %s", src_path, result["error"])
            return False
        content = base64.b64decode(result["content_b64"])
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)
        log.info("Downloaded via vsandbox: %s → %s", src_path, local_path)
        return True
    except Exception as e:
        log.warning("vsandbox_get_file %s failed: %s", src_path, e)
        return False


@OCLAW_RUN_FSM.fast_service
def run_task(
    request: RunTaskRequest, fsm_context: RequestContext = None
) -> RunTaskResponse:
    """End-to-end single-task execution, exposed as a fast-service RPC.

    The loadgen calls this once per task. It wraps load_task -> prepare_workspace
    -> launch_openclaw -> collect_outputs into one atomic call.

    Supports four modes:
    - No sandbox (sandbox=False, sandbox_mode=""): host-direct execution
    - Docker sandbox (sandbox=True): local Docker container
    - SSH sandbox (sandbox_mode="ssh"): remote Docker+sshd via env-manager RPC
    - VSandbox (sandbox_mode="vsandbox"): remote Docker via HTTP RPC (no SSH)
    """
    t0_total = time.perf_counter()
    task = load_task(task_md_path=request.task_md_path)

    use_ssh_sandbox = request.sandbox_mode == "ssh"
    use_vsandbox = request.sandbox_mode == "vsandbox"
    use_docker_sandbox = request.sandbox and not use_ssh_sandbox and not use_vsandbox
    sandbox = use_docker_sandbox or use_ssh_sandbox or use_vsandbox

    ssh_sandbox = None
    ssh_key_path = None
    ssh_wrapper_path = None
    vsandbox_resp = None
    vsandbox_endpoint = os.environ.get("VSANDBOX_ENDPOINT", "")

    if use_vsandbox:
        if not vsandbox_endpoint:
            raise RuntimeError(
                "sandbox_mode=vsandbox requested but VSANDBOX_ENDPOINT env var is not set."
            )
        from agsbench_openclaw.env_service import (
            SandboxCreateRequest,
            SandboxCreateResponse,
            SandboxDestroyRequest,
            sandbox_create,
            sandbox_destroy,
        )

        log.info("[%s] Creating vsandbox via env-manager...", task.task_id)
        vsandbox_resp = sandbox_create(
            SandboxCreateRequest(
                image="openclaw-sandbox-ssh:bookworm-slim",
                mode="vsandbox",
            )
        )
        if vsandbox_resp.error:
            total_wall = time.perf_counter() - t0_total
            return RunTaskResponse(
                task_id=task.task_id,
                run_id="",
                exit_code=None,
                wall_s=0,
                total_wall_s=total_wall,
                timed_out=False,
                error=f"sandbox_create (vsandbox) failed: {vsandbox_resp.error}",
                sandbox=True,
                output_dir="",
            )
        _configure_vsandbox(vsandbox_resp, vsandbox_endpoint)

    elif use_ssh_sandbox:
        if os.environ.get("OPENCLAW_SANDBOX_MODE") != "ssh":
            raise RuntimeError(
                "sandbox_mode=ssh requested but OPENCLAW_SANDBOX_MODE env var is not set. "
                "Set OPENCLAW_SANDBOX_MODE=ssh before starting the service."
            )
        from agsbench_openclaw.env_service import (
            SandboxCreateRequest,
            SandboxCreateResponse,
            SandboxDestroyRequest,
            sandbox_create,
            sandbox_destroy,
        )

        log.info("[%s] Creating SSH sandbox via env-manager...", task.task_id)
        ssh_sandbox = sandbox_create(
            SandboxCreateRequest(image="openclaw-sandbox-ssh:bookworm-slim")
        )
        if ssh_sandbox.error:
            total_wall = time.perf_counter() - t0_total
            return RunTaskResponse(
                task_id=task.task_id,
                run_id="",
                exit_code=None,
                wall_s=0,
                total_wall_s=total_wall,
                timed_out=False,
                error=f"sandbox_create failed: {ssh_sandbox.error}",
                sandbox=True,
                output_dir="",
            )
        ssh_key_path, ssh_wrapper_path = _configure_ssh_sandbox(ssh_sandbox)

    try:
        output_dir = prepare_workspace(
            output_root=request.output_root,
            task_id=task.task_id,
            sandbox=use_docker_sandbox,
            sandbox_workspace=request.sandbox_workspace if use_docker_sandbox else "",
        )

        agent_id = ""
        if use_vsandbox:
            agent_id = "main-vsandbox"
        elif use_ssh_sandbox:
            agent_id = "main-ssh-sandbox"
        elif use_docker_sandbox:
            agent_id = "main-sandbox"

        # Set timing CSV for vsandbox client-side latency measurement
        if use_vsandbox:
            timing_csv = str(Path(output_dir) / "vsandbox_client_timing.csv")
            os.environ["VSANDBOX_TIMING_CSV"] = timing_csv
            # Start the daemon now (after timing CSV path is known)
            _start_vsandbox_daemon(vsandbox_endpoint, timing_csv)

        if use_vsandbox and task.sandbox_files:
            _vsandbox_put_files(
                vsandbox_endpoint, vsandbox_resp.sandbox_id, task.sandbox_files
            )
        elif use_ssh_sandbox and task.sandbox_files:
            _scp_files_to_sandbox(
                ssh_sandbox, ssh_key_path, ssh_wrapper_path, task.sandbox_files
            )

        run = launch_openclaw(
            task=task,
            model=request.model,
            output_dir=output_dir,
            timeout_override_s=request.timeout_override_s,
            thinking_level=request.thinking_level,
            sandbox=sandbox,
            agent_id=agent_id,
        )

        if use_vsandbox:
            sessions_root = request.sessions_root or str(
                Path.home() / ".openclaw" / "agents" / "main-vsandbox" / "sessions"
            )
        elif use_ssh_sandbox:
            sessions_root = request.sessions_root or str(
                Path.home() / ".openclaw" / "agents" / "main-ssh-sandbox" / "sessions"
            )
        else:
            sessions_root = request.sessions_root or str(
                Path.home()
                / ".openclaw"
                / "agents"
                / ("main-sandbox" if use_docker_sandbox else "main")
                / "sessions"
            )

        if use_vsandbox and vsandbox_resp:
            local_results = str(Path(output_dir) / "results.md")
            got = _vsandbox_get_file(
                vsandbox_endpoint,
                vsandbox_resp.sandbox_id,
                "/workspace/results/results.md",
                local_results,
            )
            results_md_path = (
                local_results if got else str(Path(output_dir) / "results.md")
            )
        elif use_ssh_sandbox and ssh_sandbox and ssh_key_path:
            scp_result = _scp_results_from_sandbox(
                ssh_sandbox, ssh_key_path, ssh_wrapper_path, output_dir
            )
            results_md_path = scp_result or str(Path(output_dir) / "results.md")
        elif use_docker_sandbox and request.sandbox_workspace:
            results_md_path = str(
                Path(request.sandbox_workspace) / "results" / "results.md"
            )
        else:
            results_md_path = str(
                Path(output_dir) / "task_workspace" / "results" / "results.md"
            )

        run = collect_outputs(
            run=run,
            sessions_root=sessions_root,
            results_md_path=results_md_path,
        )
    finally:
        if use_vsandbox and vsandbox_resp:
            _stop_vsandbox_daemon()
            log.info(
                "[%s] Destroying vsandbox %s...",
                task.task_id,
                vsandbox_resp.sandbox_id,
            )
            from agsbench_openclaw.env_service import SandboxDestroyRequest, sandbox_destroy

            sandbox_destroy(SandboxDestroyRequest(sandbox_id=vsandbox_resp.sandbox_id))
        elif use_ssh_sandbox and ssh_sandbox:
            log.info(
                "[%s] Destroying SSH sandbox %s...",
                task.task_id,
                ssh_sandbox.sandbox_id,
            )
            from agsbench_openclaw.env_service import SandboxDestroyRequest, sandbox_destroy

            sandbox_destroy(SandboxDestroyRequest(sandbox_id=ssh_sandbox.sandbox_id))
            for tmp_path in (ssh_key_path, ssh_wrapper_path):
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    total_wall = time.perf_counter() - t0_total
    return RunTaskResponse(
        task_id=run.task_id,
        run_id=run.run_id,
        exit_code=run.exit_code,
        wall_s=run.wall_s,
        total_wall_s=total_wall,
        timed_out=run.timed_out,
        error=run.error,
        results_md_produced=bool(run.results_md_path),
        sandbox=sandbox,
        chat_jsonl_path=run.chat_jsonl_path,
        output_dir=run.output_dir,
    )
