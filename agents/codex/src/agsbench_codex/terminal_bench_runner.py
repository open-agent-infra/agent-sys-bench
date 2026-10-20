"""Runner for OpenAI Codex CLI on TerminalBench tasks.

Each TerminalBench task has its own docker-compose.yaml that brings up a
sandbox container. We:
1. Resolve the task directory (staged by the terminal-use prepare_data).
2. Bring up the compose project with ``docker compose up --build -d``.
3. Write ``~/.codex/config.toml`` into the agent container (routes LLM
   calls through codex-proxy).
4. ``docker exec`` codex inside the running container.
5. Run ``run-tests.sh`` inside the container to evaluate success.
6. Tear the compose project down, including the per-trial image.

The approach mirrors terminal-bench's own ``installed_agents/codex``
harness, but wires codex through our in-tree responses-API proxy instead
of talking to OpenAI directly, and uses the fast-service timing wrappers.

NB: no ``from __future__ import annotations`` (see fast-service-gotchas).
"""

import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Dict, List, Optional

from fast_service import RequestContext

from agsbench_codex.fsm import codex_fsm


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
    codex_exec_secs: Optional[float] = None
    test_runner_secs: Optional[float] = None


# === Resource pool ===========================================================

_CPUSET_POOL: Optional[Queue] = None
_CPUSET_POOL_LOCK = Lock()


def _get_cpuset_pool() -> Optional[Queue]:
    global _CPUSET_POOL
    if _CPUSET_POOL is not None:
        return _CPUSET_POOL
    with _CPUSET_POOL_LOCK:
        if _CPUSET_POOL is not None:
            return _CPUSET_POOL
        spec = os.getenv("CODEX_CPUSETS", "").strip()
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
    """Return the local path to the task directory.

    If the task dict has a ``task_dir`` key pointing to an existing path,
    use it directly (local dataset).  Otherwise fall back to the
    terminal-bench registry cache at ``task_cache_dir/<task_id>``.
    """
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
        f"Run scripts/1.prepare_data.sh first, or set task_dir in the task dict."
    )


# === Docker Compose helpers ===================================================

_COMPOSE_ENV_KEYS = (
    "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME",
    "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME",
    "T_BENCH_TEST_DIR",
    "T_BENCH_TASK_LOGS_PATH",
    "T_BENCH_TASK_AGENT_LOGS_PATH",
)


def _compose_project_name(trial_id: str) -> str:
    return f"codex-tb-{trial_id[:12]}"


def _compose_up(
    task_dir: Path,
    project: str,
    logs_dir: Path,
    container_name: str,
    image_name: str,
    cpuset: Optional[str] = None,
    codex_home_host_dir: Optional[str] = None,
    codex_home_container_path: Optional[str] = None,
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
        # Disable asciinema recording in CI-style runs.
        "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
        "T_BENCH_DISABLE_ASCIINEMA": "1",
    }
    # Write a temporary compose override so task containers can reach
    # host.docker.internal (needed for codex → codex-proxy).
    cpuset_line = f'    cpuset: "{cpuset}"\n' if cpuset else ""
    if codex_home_host_dir and codex_home_container_path:
        volume_lines = (
            "    volumes:\n"
            f'      - "{codex_home_host_dir}:{codex_home_container_path}"\n'
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
        import os as _os

        _os.unlink(override_path)
    # Poll until the container is in 'running' state before returning.
    import time as _time

    for _ in range(30):
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", container_name],
            capture_output=True,
            text=True,
        )
        if r.stdout.strip() == "true":
            break
        _time.sleep(1)


def _compose_down(task_dir: Path, project: str) -> None:
    # Tear down by project name only — no -f needed and avoids mismatch
    # with the runtime override file that was already deleted.
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
                # Each trial builds its own uniquely named image; drop it.
                "--rmi",
                "all",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except Exception:
        pass


# === Timed helpers (monitor_only) ============================================


# Two timed call-sites: sandbox setup (compose build + up) and the task's
# test run. The codex exec in between is the agent step itself; it stays
# untraced, so analysis attributes it as the residual. monitor_only because
# the args/returns are not pydantic models.


@codex_fsm.fast_service(monitor_only=True)
def env_execute_compose_up(
    task_dir: str,
    project: str,
    logs_dir: str,
    container_name: str,
    image_name: str,
    cpuset: Optional[str] = None,
    codex_home_host_dir: Optional[str] = None,
    codex_home_container_path: Optional[str] = None,
    fsm_context: RequestContext = None,
) -> dict:
    """Build and start the task's sandbox container."""
    _compose_up(
        task_dir=Path(task_dir),
        project=project,
        logs_dir=Path(logs_dir),
        container_name=container_name,
        image_name=image_name,
        cpuset=cpuset,
        codex_home_host_dir=codex_home_host_dir,
        codex_home_container_path=codex_home_container_path,
    )
    return {"container_name": container_name}


@codex_fsm.fast_service(monitor_only=True)
def env_execute_tests(
    container_name: str,
    task_dir: str,
    timeout: int,
    fsm_context: RequestContext = None,
) -> dict:
    """Run the task's run-tests.sh inside the sandbox."""
    passed, total, raw = _run_tests(
        container_name=container_name, task_dir=Path(task_dir), timeout=timeout
    )
    return {"tests_passed": passed, "tests_total": total, "output": raw}


# === Config injection =========================================================


def _write_codex_config_to_container(container_name: str) -> None:
    """Write ~/.codex/config.toml into the task container."""
    proxy_url = os.environ.get("CODEX_PROXY_URL", "http://host.docker.internal:4001/v1")
    model_name = os.environ.get("CODEX_MODEL", "deepseek-chat")
    config_toml = (
        textwrap.dedent(
            f"""
        model_provider = "litellm"
        model = "{model_name}"
        approval_policy = "never"
        sandbox_mode = "danger-full-access"

        [model_providers.litellm]
        name = "LiteLLM Proxy"
        base_url = "{proxy_url}"
        env_key = "OPENAI_API_KEY"
        wire_api = "responses"
        env_http_headers = {{ "X-Codex-Task-Id" = "CODEX_TASK_ID" }}
        """
        ).strip()
        + "\n"
    )
    subprocess.run(
        [
            "docker",
            "exec",
            "-i",
            container_name,
            "bash",
            "-c",
            "mkdir -p /root/.codex && cat > /root/.codex/config.toml",
        ],
        input=config_toml,
        text=True,
        check=True,
        timeout=30,
    )


# === Test evaluation ==========================================================


def _parse_pytest_output(output: str) -> tuple[int, int]:
    """Parse pytest -rA output → (tests_passed, tests_total).

    Counts PASSED/XFAIL/SKIPPED as passed (matches terminal-bench
    PytestParser semantics).
    """
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
    # If we found no summary block, try the condensed line: "X passed, Y failed in Zs"
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


def _run_tests(
    container_name: str, task_dir: Path, timeout: int
) -> tuple[int, int, str]:
    """Run run-tests.sh inside the container, return (passed, total, raw_output)."""
    # Copy run-tests.sh and tests/ into the container (they may not be baked in).
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

    # If CODEX_TB_RUN_DIR is set, write all per-task artifacts there too.
    # That directory must be mounted into the service container at the same
    # host path so the host Docker daemon can bind-mount subdirs into task
    # containers (for CODEX_HOME).
    run_dir = os.environ.get("CODEX_TB_RUN_DIR")
    if run_dir:
        artifact_dir = Path(run_dir) / "tasks" / task_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
    else:
        artifact_dir = task_out_dir

    model_name = model_name_override or os.environ.get("CODEX_MODEL", "deepseek-chat")
    trial_id = uuid.uuid4().hex
    project = _compose_project_name(trial_id)
    container_name = f"codex-tb-client-{trial_id[:8]}"
    image_name = f"codex-tb-img-{trial_id[:8]}"

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

        # Per-task CODEX_HOME: host dir mounted into the task container so
        # codex session files land directly on the host filesystem.
        codex_home_container_path = f"/tmp/codex-home-{trial_id[:8]}"
        if run_dir:
            codex_home_host_dir = str(artifact_dir / "codex_home")
            Path(codex_home_host_dir).mkdir(parents=True, exist_ok=True)
        else:
            codex_home_host_dir = None

        env_execute_compose_up(
            task_dir=str(task_dir),
            project=project,
            logs_dir=str(logs_dir),
            container_name=container_name,
            image_name=image_name,
            cpuset=cpuset,
            codex_home_host_dir=codex_home_host_dir,
            codex_home_container_path=codex_home_container_path,
            fsm_context=fsm_context,
        )
        compose_up = True

        # Install codex binary into the task container.
        # CODEX_BIN_HOST_PATH is a host-side path used for docker run -v mounts.
        # Inside the service container, the staged binary is accessible at the
        # /workspace/codex/.cache/codex_bin/codex container-internal path
        # (the .cache dir is volume-mounted). Fall back to the baked-in binary.
        codex_bin_container_path = (
            "/workspace/codex/.cache/codex_bin/codex"
            if __import__("os").path.isfile("/workspace/codex/.cache/codex_bin/codex")
            else "/usr/local/bin/codex"
        )
        subprocess.run(
            [
                "docker",
                "cp",
                codex_bin_container_path,
                f"{container_name}:/usr/local/bin/codex",
            ],
            check=True,
            timeout=60,
            capture_output=True,
        )

        _write_codex_config_to_container(container_name)

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
            input=instruction,
            text=True,
            check=True,
            timeout=30,
        )

        api_key = os.environ.get("OPENAI_API_KEY", "")
        # codex_home_container_path is the per-task CODEX_HOME inside the container.
        # If a host dir is mounted there via the compose override, session files land
        # directly on the host; otherwise we fall back to a tmpdir.
        exec_cmd = (
            f'export OPENAI_API_KEY="$OPENAI_API_KEY"; '
            f"export CODEX_HOME={codex_home_container_path}; "
            f"mkdir -p {codex_home_container_path}; "
            f"cp -r /root/.codex/. {codex_home_container_path}/ 2>/dev/null || true; "
            f'echo "[diag] codex $(codex --version 2>&1 | head -1)" >&2; '
            f"codex exec --yolo --skip-git-repo-check "
            f"--json --output-last-message /tmp/codex_last.txt "
            f'"$(cat {prompt_path})"'
        )
        _codex_exec_start = time.monotonic()
        exec_result = subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                f"OPENAI_API_KEY={api_key}",
                "-e",
                "CODEX_HOME=/root/.codex",
                "-e",
                f"CODEX_TASK_ID={task_id}",
                container_name,
                "bash",
                "-c",
                exec_cmd,
            ],
            timeout=step_timeout,
            capture_output=True,
            text=True,
        )
        extra["codex_exec_secs"] = round(time.monotonic() - _codex_exec_start, 2)
        extra["codex_returncode"] = exec_result.returncode
        extra["codex_stdout_tail"] = exec_result.stdout[-2048:]
        extra["codex_stderr_tail"] = exec_result.stderr[-2048:]
        # Write stdout JSONL to the output dir and to the run artifact dir (if set).
        for dest in (
            [task_out_dir, artifact_dir]
            if artifact_dir != task_out_dir
            else [task_out_dir]
        ):
            try:
                (dest / f"{task_id}.codex_stdout.jsonl").write_text(exec_result.stdout)
            except Exception:
                pass
        # Session files land in artifact_dir/codex_home/sessions/ automatically via
        # the bind-mount added to the compose override. No docker cp needed.

        if exec_result.returncode != 0:
            exit_status = f"codex_returncode_{exec_result.returncode}"

        # Evaluate using run-tests.sh.
        _tests_start = time.monotonic()
        tests = env_execute_tests(
            container_name=container_name,
            task_dir=str(task_dir),
            timeout=test_timeout,
            fsm_context=fsm_context,
        )
        tests_passed = tests["tests_passed"]
        tests_total = tests["tests_total"]
        raw_test_output = tests["output"]
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
                # {instance_id, info} wrapper, the shape the analysis
                # toolkit reads (sdk/analysis/task_registry._load_codex).
                (dest / f"{task_id}.traj.json").write_text(
                    json.dumps(
                        {"instance_id": task_id, "info": result_entry}, indent=2
                    )
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
        codex_exec_secs=extra.get("codex_exec_secs"),
        test_runner_secs=extra.get("test_runner_secs"),
    )
