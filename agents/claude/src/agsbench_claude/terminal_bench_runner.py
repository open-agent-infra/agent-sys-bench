"""Runner for Claude Code CLI on TerminalBench tasks.

Each TerminalBench task has its own docker-compose.yaml that brings up a
sandbox container. We:
1. Download / resolve the task directory via the terminal-bench registry.
2. Bring up the compose project with ``docker compose up --build -d``.
3. Install Claude Code CLI into the task container via npm.
4. ``docker exec`` claude inside the running container.
5. Run ``run-tests.sh`` inside the container to evaluate success.
6. Tear the compose project down.

The approach mirrors codex's terminal_bench_runner.py, but invokes
claude code CLI instead of codex, using ANTHROPIC_API_KEY / ANTHROPIC_BASE_URL
environment variables for authentication.

NB: no ``from __future__ import annotations`` (see fast-service-gotchas).
"""

import json
import os
import shutil
import subprocess
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional

from fast_service import RequestContext

from agsbench_claude.run_task import CLAUDE_RUN_FSM


_RESULTS_LOCK = Lock()


@dataclass
class TerminalBenchRunResult:
    success: bool
    task_id: str
    model_name: str
    tests_passed: int
    tests_total: int
    exit_status: Optional[str]
    error: Optional[str]
    claude_exec_secs: Optional[float] = None
    test_runner_secs: Optional[float] = None


# === Resource pool (mirrors codex runner) ====================================

_CPUSET_POOL: Optional[Queue] = None
_CPUSET_POOL_LOCK = Lock()


def _get_cpuset_pool() -> Optional[Queue]:
    global _CPUSET_POOL
    if _CPUSET_POOL is not None:
        return _CPUSET_POOL
    with _CPUSET_POOL_LOCK:
        if _CPUSET_POOL is not None:
            return _CPUSET_POOL
        spec = os.getenv("CLAUDE_CPUSETS", "").strip()
        if not spec:
            return None
        pool: Queue = Queue()
        for cpuset in spec.split(","):
            cpuset = cpuset.strip()
            if cpuset:
                pool.put(cpuset)
        _CPUSET_POOL = pool
        return _CPUSET_POOL


def _acquire_cpuset() -> Optional[str]:
    pool = _get_cpuset_pool()
    return pool.get() if pool is not None else None


def _release_cpuset(cpuset: Optional[str]) -> None:
    if cpuset is None:
        return
    pool = _get_cpuset_pool()
    if pool is not None:
        pool.put(cpuset)


# === Task directory resolution ===============================================


def _resolve_task_dir(task: Dict[str, Any], task_cache_dir: str) -> Path:
    """Return the local path to the task directory."""
    task_id = task["task_id"]
    if "task_dir" in task:
        p = Path(task["task_dir"])
        if p.is_dir():
            return p
    cache_path = Path(task_cache_dir) / task_id
    if cache_path.is_dir():
        return cache_path
    raise FileNotFoundError(
        f"Task directory not found for '{task_id}'. "
        f"Run 2b.prepare_terminal_bench.sh first, or set task_dir in the task dict."
    )


# === Docker Compose helpers ===================================================


def _compose_project_name(trial_id: str) -> str:
    return f"claude-tb-{trial_id[:12]}"


def _compose_up(
    task_dir: Path,
    project: str,
    logs_dir: Path,
    container_name: str,
    image_name: str,
    cpuset: Optional[str] = None,
    claude_home_host_dir: Optional[str] = None,
    claude_home_container_path: Optional[str] = None,
) -> None:
    """Bring up the compose project and wait until the client container is running."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": image_name,
        "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": container_name,
        "T_BENCH_TEST_DIR": "/tests",
        "T_BENCH_TASK_LOGS_PATH": str(logs_dir / "task.log"),
        "T_BENCH_TASK_AGENT_LOGS_PATH": str(logs_dir / "agent.log"),
        "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
        "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/agent_logs",
        "T_BENCH_DISABLE_ASCIINEMA": "1",
    }
    cpuset_line = f'    cpuset: "{cpuset}"\n' if cpuset else ""
    if claude_home_host_dir and claude_home_container_path:
        volume_lines = (
            "    volumes:\n"
            f'      - "{claude_home_host_dir}:{claude_home_container_path}"\n'
        )
    else:
        volume_lines = ""
    override_content = (
        "services:\n"
        "  client:\n"
        "    extra_hosts:\n"
        '      - "host.docker.internal:host-gateway"\n' + cpuset_line + volume_lines
    )
    import tempfile as _tmpfile

    with _tmpfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="tb-override-"
    ) as tf:
        tf.write(override_content)
        override_path = tf.name
    try:
        r = subprocess.run(
            [
                "docker",
                "compose",
                "-p",
                project,
                "-f",
                str(task_dir / "docker-compose.yaml"),
                "-f",
                override_path,
                "up",
                "--build",
                "-d",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode != 0:
            raise subprocess.CalledProcessError(
                r.returncode,
                r.args,
                output=r.stdout,
                stderr=r.stderr,
            )
    finally:
        os.unlink(override_path)
    for _ in range(30):
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True,
        )
        if r.stdout.strip() == "true":
            break
        time.sleep(1)


def _compose_down(task_dir: Path, project: str) -> None:
    try:
        subprocess.run(
            [
                "docker",
                "compose",
                "-p",
                project,
                "down",
                "--volumes",
                "--remove-orphans",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception:
        pass


# === Timed helpers (monitor_only) ============================================


@CLAUDE_RUN_FSM.fast_service(monitor_only=True)
def tb_exec_call(
    container_name: str,
    cmd_str: str,
    timeout: int,
    env_overrides: Optional[Dict[str, str]] = None,
    fsm_context: RequestContext = None,
) -> dict:
    """Run a shell command inside the TerminalBench task container."""
    docker_cmd = ["docker", "exec"]
    for k, v in (env_overrides or {}).items():
        docker_cmd += ["-e", f"{k}={v}"]
    docker_cmd += [container_name, "bash", "-lc", cmd_str]
    result = subprocess.run(
        docker_cmd,
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[-4096:],
        "stderr": result.stderr[-4096:],
    }


# === Claude CLI installation into task container ==============================


def _install_claude_in_container(container_name: str, timeout: int = 300) -> None:
    """Install Node.js and Claude Code CLI into the task container."""
    install_cmd = (
        "apt-get update -qq && "
        "apt-get install -y -qq curl && "
        "curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && "
        "apt-get install -y -qq nodejs && "
        "npm install -g @anthropic-ai/claude-code"
    )
    result = subprocess.run(
        ["docker", "exec", container_name, "bash", "-c", install_cmd],
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Try alternative: maybe node is already installed
        check = subprocess.run(
            ["docker", "exec", container_name, "which", "claude"],
            capture_output=True,
            text=True,
        )
        if check.returncode != 0:
            raise RuntimeError(
                f"Failed to install claude in container: {result.stderr[-2048:]}"
            )


# === Test evaluation ==========================================================


def _parse_pytest_output(output: str) -> tuple:
    """Parse pytest -rA output -> (tests_passed, tests_total)."""
    passed = 0
    total = 0
    in_summary = False
    for line in output.splitlines():
        if "short test summary info" in line:
            in_summary = True
            continue
        if in_summary:
            if not line.startswith(
                ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")
            ):
                continue
            total += 1
            status = line.split()[0]
            if status in ("PASSED", "XFAIL", "SKIPPED"):
                passed += 1
    if total == 0:
        import re

        m = re.search(r"(\d+) passed", output)
        m2 = re.search(r"(\d+) failed", output)
        m3 = re.search(r"(\d+) error", output)
        p = int(m.group(1)) if m else 0
        f = int(m2.group(1)) if m2 else 0
        e = int(m3.group(1)) if m3 else 0
        passed = p
        total = p + f + e
    return passed, total


def _run_tests(container_name: str, task_dir: Path, timeout: int) -> tuple:
    """Run run-tests.sh inside the container, return (passed, total, raw_output)."""
    tests_dir = task_dir / "tests"
    run_tests_sh = task_dir / "run-tests.sh"
    if tests_dir.is_dir():
        subprocess.run(
            ["docker", "cp", str(tests_dir), f"{container_name}:/tests"],
            check=True,
            timeout=60,
            capture_output=True,
        )
    if run_tests_sh.is_file():
        subprocess.run(
            ["docker", "cp", str(run_tests_sh), f"{container_name}:/run-tests.sh"],
            check=True,
            timeout=30,
            capture_output=True,
        )
    result = subprocess.run(
        ["docker", "exec", container_name, "bash", "/run-tests.sh"],
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    raw = result.stdout + result.stderr
    passed, total = _parse_pytest_output(raw)
    return passed, total, raw


# === Results persistence ======================================================


def _update_results(results_dir: Path, entry: dict) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.json"
    with _RESULTS_LOCK:
        data = {}
        if results_path.exists():
            try:
                data = json.loads(results_path.read_text())
            except json.JSONDecodeError:
                data = {}
        data[entry["task_id"]] = entry
        results_path.write_text(json.dumps(data, indent=2))


# === Session JSONL helpers ====================================================


def _find_session_jsonl_in_container(
    container_name: str,
) -> Optional[str]:
    """Find the most recent session JSONL inside the container."""
    result = subprocess.run(
        [
            "docker",
            "exec",
            container_name,
            "bash",
            "-c",
            "find /root/.claude -name '*.jsonl' -path '*/projects/*' 2>/dev/null "
            "| head -5",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    paths = [p.strip() for p in result.stdout.strip().splitlines() if p.strip()]
    if not paths:
        return None
    return paths[-1]


def _copy_session_from_container(
    container_name: str, container_path: str, dest: Path
) -> Optional[Path]:
    """Copy a session JSONL from the container to the host."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["docker", "cp", f"{container_name}:{container_path}", str(dest)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if r.returncode == 0 and dest.exists():
        return dest
    return None


# === Per-task runner ==========================================================


def run_one_terminal_bench(
    task: Dict[str, Any],
    output_dir: str,
    task_cache_dir: str,
    fsm_context: Optional[RequestContext],
    model_name_override: Optional[str] = None,
    step_timeout: int = 900,
    test_timeout: int = 120,
) -> TerminalBenchRunResult:
    task_id = task["task_id"]
    instruction = task.get("instruction", task.get("problem_statement", ""))
    out_root = Path(output_dir)
    task_out_dir = out_root / task_id
    task_out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = os.environ.get("CLAUDE_TB_RUN_DIR")
    if run_dir:
        artifact_dir = Path(run_dir) / "tasks" / task_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
    else:
        artifact_dir = task_out_dir

    model_name = model_name_override or os.environ.get(
        "CLAUDE_MODEL", "claude-sonnet-4-6"
    )
    trial_id = uuid.uuid4().hex
    project = _compose_project_name(trial_id)
    container_name = f"claude-tb-client-{trial_id[:8]}"
    image_name = f"claude-tb-img-{trial_id[:8]}"

    exit_status: Optional[str] = None
    error: Optional[str] = None
    tests_passed = 0
    tests_total = 0
    extra: dict = {}

    cpuset = _acquire_cpuset()
    task_dir: Optional[Path] = None
    compose_up = False

    try:
        task_dir = _resolve_task_dir(task, task_cache_dir)
        logs_dir = artifact_dir / "logs"

        claude_home_container_path = f"/tmp/claude-home-{trial_id[:8]}"
        if run_dir:
            claude_home_host_dir = str(artifact_dir / "claude_home")
            Path(claude_home_host_dir).mkdir(parents=True, exist_ok=True)
        else:
            claude_home_host_dir = None

        _compose_up(
            task_dir=task_dir,
            project=project,
            logs_dir=logs_dir,
            container_name=container_name,
            image_name=image_name,
            cpuset=cpuset,
            claude_home_host_dir=claude_home_host_dir,
            claude_home_container_path=claude_home_container_path,
        )
        compose_up = True

        _install_claude_in_container(container_name)

        # Create a non-root user with sudo so claude can escalate when needed.
        # Claude CLI refuses --dangerously-skip-permissions as root, so we
        # must run as a regular user. Install sudo and grant NOPASSWD access
        # so the agent can write to root-owned paths like /app/.
        subprocess.run(
            [
                "docker",
                "exec",
                container_name,
                "bash",
                "-c",
                "(apt-get update -qq && apt-get install -y -qq sudo 2>/dev/null || "
                "yum install -y sudo 2>/dev/null || "
                "apk add sudo 2>/dev/null || true) && "
                "(id bench 2>/dev/null || useradd -m -s /bin/bash bench) && "
                "echo 'bench ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && "
                "chmod 440 /etc/sudoers && "
                "chown -R bench:bench /tmp 2>/dev/null; true",
            ],
            check=False,
            timeout=60,
            capture_output=True,
        )

        # Prepend environment context so the model knows it has sudo.
        full_instruction = (
            "You are running as user 'bench' in a Docker container. "
            "You have passwordless sudo access — use 'sudo' for any "
            "operation that requires root privileges (e.g. writing to "
            "/app/, installing packages with apt-get, modifying system "
            "files). Always prefer 'sudo' over workarounds when you "
            "encounter permission errors.\n\n" + instruction
        )

        # Write instruction to a temp file inside the container.
        prompt_path = "/tmp/tb_instruction.txt"
        subprocess.run(
            [
                "docker",
                "exec",
                "-i",
                container_name,
                "bash",
                "-c",
                f"cat > {prompt_path}",
            ],
            input=full_instruction,
            text=True,
            check=True,
            timeout=30,
        )

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        base_url = os.environ.get(
            "ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic"
        )
        effort = os.environ.get("CLAUDE_EFFORT", "")
        effort_flag = f"--effort {effort}" if effort else ""

        exec_cmd = (
            f"export ANTHROPIC_API_KEY='{api_key}'; "
            f"export ANTHROPIC_BASE_URL='{base_url}'; "
            f"export CLAUDE_CONFIG_DIR={claude_home_container_path}; "
            f"mkdir -p {claude_home_container_path}; "
            f'echo "[diag] claude $(claude --version 2>&1 | head -1)" >&2; '
            f"claude --print --output-format stream-json --verbose "
            f"--dangerously-skip-permissions "
            f"--model {model_name} {effort_flag} "
            f'"$(cat {prompt_path})"'
        )
        _claude_exec_start = time.monotonic()
        exec_result = subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "bench",
                "-e",
                f"ANTHROPIC_API_KEY={api_key}",
                "-e",
                f"ANTHROPIC_BASE_URL={base_url}",
                "-e",
                f"HOME=/home/bench",
                "-e",
                f"CLAUDE_TASK_ID={task_id}",
                container_name,
                "bash",
                "-lc",
                exec_cmd,
            ],
            timeout=step_timeout,
            capture_output=True,
            text=True,
        )
        extra["claude_exec_secs"] = round(time.monotonic() - _claude_exec_start, 2)
        extra["claude_returncode"] = exec_result.returncode
        extra["claude_stdout_tail"] = exec_result.stdout[-2048:]
        extra["claude_stderr_tail"] = exec_result.stderr[-2048:]

        # Parse stream-json output for session_id and result event
        session_id = None
        result_event = None
        for line in exec_result.stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                evt = json.loads(line)
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "system" and evt.get("subtype") == "init":
                session_id = evt.get("session_id")
            elif evt.get("type") == "result":
                result_event = evt

        if result_event:
            extra["duration_api_ms"] = result_event.get("duration_api_ms")
            extra["num_turns"] = result_event.get("num_turns")
            usage = result_event.get("usage", {})
            extra["input_tokens"] = usage.get("input_tokens")
            extra["output_tokens"] = usage.get("output_tokens")

        # Save stream-json stdout
        for dest in (
            [task_out_dir, artifact_dir]
            if artifact_dir != task_out_dir
            else [task_out_dir]
        ):
            try:
                (dest / f"{task_id}.claude_stdout.jsonl").write_text(exec_result.stdout)
            except Exception:
                pass

        # Copy session JSONL from container
        session_jsonl_path = None
        if session_id or True:
            container_session_path = _find_session_jsonl_in_container(container_name)
            if container_session_path:
                for dest in (
                    [task_out_dir, artifact_dir]
                    if artifact_dir != task_out_dir
                    else [task_out_dir]
                ):
                    dst = dest / "session.jsonl"
                    copied = _copy_session_from_container(
                        container_name, container_session_path, dst
                    )
                    if copied and session_jsonl_path is None:
                        session_jsonl_path = str(copied)

        # Postprocess session JSONL
        if session_jsonl_path:
            try:
                from agsbench_claude.postprocess import postprocess

                postprocess(
                    session_jsonl_path=Path(session_jsonl_path),
                    output_dir=task_out_dir,
                    task_id=task_id,
                    run_id=trial_id[:6],
                )
            except Exception:
                pass

        if exec_result.returncode != 0:
            exit_status = f"claude_returncode_{exec_result.returncode}"

        # Evaluate using run-tests.sh.
        _tests_start = time.monotonic()
        tests_passed, tests_total, raw_test_output = _run_tests(
            container_name=container_name,
            task_dir=task_dir,
            timeout=test_timeout,
        )
        extra["test_runner_secs"] = round(time.monotonic() - _tests_start, 2)
        extra["test_output_tail"] = raw_test_output[-2048:]
        for dest in (
            [task_out_dir, artifact_dir]
            if artifact_dir != task_out_dir
            else [task_out_dir]
        ):
            try:
                (dest / f"{task_id}.test_output.txt").write_text(raw_test_output)
            except Exception:
                pass

        if tests_total > 0 and tests_passed == tests_total:
            exit_status = exit_status or "Resolved"
        elif tests_total > 0:
            exit_status = exit_status or f"Failed({tests_passed}/{tests_total})"
        else:
            exit_status = exit_status or "NoTests"

    except subprocess.TimeoutExpired as e:
        exit_status = "Timeout"
        error = f"timeout after {step_timeout}s: {e}"
        extra["traceback"] = traceback.format_exc()
    except Exception as e:
        exit_status = type(e).__name__
        error = str(e)
        extra["traceback"] = traceback.format_exc()
    finally:
        if compose_up and task_dir is not None:
            _compose_down(task_dir, project)
        _release_cpuset(cpuset)

        result_entry = {
            "task_id": task_id,
            "model_name": model_name,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "success": tests_total > 0 and tests_passed == tests_total,
            "exit_status": exit_status,
            "error": error,
            **extra,
        }
        for dest in (
            [task_out_dir, artifact_dir]
            if artifact_dir != task_out_dir
            else [task_out_dir]
        ):
            try:
                (dest / f"{task_id}.result.json").write_text(
                    json.dumps(result_entry, indent=2)
                )
            except Exception:
                pass
        _update_results(out_root, result_entry)

    return TerminalBenchRunResult(
        success=tests_total > 0 and tests_passed == tests_total,
        task_id=task_id,
        model_name=model_name,
        tests_passed=tests_passed,
        tests_total=tests_total,
        exit_status=exit_status,
        error=error,
        claude_exec_secs=extra.get("claude_exec_secs"),
        test_runner_secs=extra.get("test_runner_secs"),
    )
