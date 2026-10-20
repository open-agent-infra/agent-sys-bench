"""Workload driver: run one WildClawBench / pinchbench task through Claude Code CLI.

Mirrors openclaw/run_task.py: load_task -> prepare_workspace -> stage_workspace_files
-> launch_claude -> collect_outputs, all wired as fast-service RPCs.

Claude Code CLI is invoked in non-interactive print mode with stream-json
output for per-turn telemetry:

    claude --print --output-format stream-json --verbose \
           --permission-mode bypassPermissions \
           --model <model> [--effort <level>] "<prompt>"

Session persistence is enabled so claude writes a session JSONL to
~/.claude/projects/<cwd>/<session-id>.jsonl with timestamps on every
event, enabling per-turn LLM and tool-call latency breakdown.

ANTHROPIC_API_KEY and ANTHROPIC_BASE_URL are set from config/.env.
For DeepSeek, ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic.

NB: do NOT add ``from __future__ import annotations`` — fast-service needs
class objects in __annotations__, not strings.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

from fast_service import FastServiceManager, RequestContext

CLAUDE_RUN_FSM = FastServiceManager()
log = logging.getLogger("claude.run_task")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
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
    output_dir: str
    agent_stdout_path: str
    agent_stderr_path: str
    exit_code: Optional[int] = None
    wall_s: float
    timed_out: bool
    error: Optional[str] = None
    results_md_path: Optional[str] = None
    session_id: Optional[str] = None
    session_jsonl_path: Optional[str] = None
    duration_api_ms: Optional[float] = None
    num_turns: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class RunTaskRequest(BaseModel):
    task_md_path: str
    model: str = "deepseek-v4-flash"
    output_root: str = "./output"
    timeout_override_s: int = 0
    effort: str = ""
    workspace_root: str = ""

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
    output_dir: str
    session_jsonl_path: Optional[str] = None
    duration_api_ms: Optional[float] = None
    num_turns: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Task format adapters (identical to openclaw's)
# ---------------------------------------------------------------------------
def _load_task_yaml(task_path: Path, text: str) -> TaskSpec:
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
    if not workspace_files:
        return []
    assets_dir = task_path.parent.parent / "assets"
    result = []
    for fspec in workspace_files:
        if isinstance(fspec, dict) and "content" in fspec:
            result.append(
                {"content": fspec["content"], "dest": f"/workspace/{fspec['path']}"}
            )
        elif isinstance(fspec, dict) and "source" in fspec:
            result.append(
                {
                    "src": str(assets_dir / fspec["source"]),
                    "dest": f"/workspace/{fspec['dest']}",
                }
            )
    return result


@CLAUDE_RUN_FSM.fast_service
def load_task(task_md_path: str, fsm_context: RequestContext = None) -> TaskSpec:
    task_path = Path(task_md_path)
    text = task_path.read_text(encoding="utf-8")
    if task_path.suffix in (".yaml", ".yml"):
        return _load_task_yaml(task_path, text)
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


@CLAUDE_RUN_FSM.fast_service
def prepare_workspace(
    output_root: str,
    task_id: str,
    workspace_root: str = "",
    fsm_context: RequestContext = None,
) -> str:
    run_id = uuid.uuid4().hex[:6]
    out = Path(output_root) / task_id / run_id
    out.mkdir(parents=True, exist_ok=True)
    (out / "task_workspace" / "results").mkdir(parents=True, exist_ok=True)
    return str(out)


@CLAUDE_RUN_FSM.fast_service
def stage_workspace_files(
    task: TaskSpec,
    output_dir: str,
    fsm_context: RequestContext = None,
) -> str:
    task_ws = Path(output_dir) / "task_workspace"
    for fspec in task.sandbox_files:
        dest = task_ws / Path(fspec["dest"]).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "content" in fspec:
            dest.write_text(fspec["content"], encoding="utf-8")
        elif "src" in fspec:
            src = Path(fspec["src"])
            if src.exists():
                shutil.copy2(src, dest)
            else:
                log.warning("sandbox file source missing: %s", src)
    return str(task_ws)


# ---------------------------------------------------------------------------
# Session JSONL locator
# ---------------------------------------------------------------------------
def _cwd_to_project_dir(cwd: str) -> Path:
    """Convert a cwd path to claude's project directory name.

    Claude CLI encodes the absolute cwd by replacing both '/' and '_'
    with '-'. We must resolve to absolute first (claude uses os.getcwd()
    which is always absolute) and then apply the same encoding.
    """
    abs_cwd = os.path.abspath(cwd)
    encoded = abs_cwd.replace("/", "-").replace("_", "-")
    home = os.environ.get("HOME", os.path.expanduser("~"))
    return Path(home) / ".claude" / "projects" / encoded


def _find_session_jsonl(cwd: str, session_id: str) -> Optional[Path]:
    """Locate the session JSONL that claude wrote."""
    proj_dir = _cwd_to_project_dir(cwd)
    candidate = proj_dir / f"{session_id}.jsonl"
    if candidate.exists():
        return candidate
    return None


# ---------------------------------------------------------------------------
# Launch with stream-json capture
# ---------------------------------------------------------------------------
@CLAUDE_RUN_FSM.fast_service
def launch_claude(
    task: TaskSpec,
    model: str,
    output_dir: str,
    task_workspace: str,
    timeout_override_s: int = 0,
    effort: str = "",
    fsm_context: RequestContext = None,
) -> RunResult:
    """Launch claude with stream-json output and session persistence."""
    out = Path(output_dir)
    run_id = out.name

    system_prefix = (
        "You are an expert in a restricted, non-interactive environment. "
        f"Solve the task efficiently before the timeout ({task.timeout_seconds}s). "
        "Run all processes in the foreground without user input or background services. "
        "Provide a complete, functional solution in a single pass with no placeholders.\n"
    )

    results_path = str(Path(task_workspace) / "results")
    rewritten_prompt = task.prompt.replace("/tmp_workspace/results", results_path)
    full_prompt = system_prefix + rewritten_prompt

    timeout = timeout_override_s if timeout_override_s > 0 else task.timeout_seconds
    effective_effort = "high" if effort == "thinking" else effort

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
    ]
    if effective_effort:
        cmd += ["--effort", effective_effort]
    cmd.append(full_prompt)

    stream_path = out / "agent.stream.jsonl"
    stdout_path = out / "agent.stdout.log"
    stderr_path = out / "agent.stderr.log"

    env = os.environ.copy()
    env["PWD"] = task_workspace

    log.info(
        "[%s] launching claude (model=%s, effort=%s, timeout=%ds, cwd=%s)",
        task.task_id,
        model,
        effective_effort or "default",
        timeout,
        task_workspace,
    )

    t0 = time.perf_counter()
    timed_out = False
    err = None
    exit_code = None
    session_id = None
    result_event = None

    try:
        with open(stderr_path, "w") as fe:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=fe,
                cwd=task_workspace,
                env=env,
            )

        # Read stream-json lines, extract session_id and result event
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
        log.warning("[%s] subprocess timed out", task.task_id)
    except Exception as exc:
        err = str(exc)
        log.exception("[%s] launch error: %s", task.task_id, exc)

    wall = time.perf_counter() - t0

    # Extract text result for backward compatibility
    final_text = ""
    duration_api_ms = None
    num_turns = None
    input_tokens = None
    output_tokens = None
    if result_event:
        final_text = result_event.get("result", "")
        duration_api_ms = result_event.get("duration_api_ms")
        num_turns = result_event.get("num_turns")
        usage = result_event.get("usage", {})
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
    stdout_path.write_text(final_text, encoding="utf-8")

    # Copy session JSONL to output dir
    session_jsonl_path = None
    if session_id:
        src = _find_session_jsonl(task_workspace, session_id)
        if src:
            dst = out / "session.jsonl"
            shutil.copy2(src, dst)
            session_jsonl_path = str(dst)
            log.info(
                "[%s] session JSONL copied: %s (%d bytes)",
                task.task_id,
                dst,
                dst.stat().st_size,
            )

    return RunResult(
        task_id=task.task_id,
        run_id=run_id,
        output_dir=str(out),
        agent_stdout_path=str(stdout_path),
        agent_stderr_path=str(stderr_path),
        exit_code=exit_code,
        wall_s=wall,
        timed_out=timed_out,
        error=err,
        session_id=session_id,
        session_jsonl_path=session_jsonl_path,
        duration_api_ms=duration_api_ms,
        num_turns=num_turns,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


@CLAUDE_RUN_FSM.fast_service
def collect_outputs(
    run: RunResult,
    results_md_path: str,
    fsm_context: RequestContext = None,
) -> RunResult:
    out = Path(run.output_dir)
    src = Path(results_md_path)
    if src.exists():
        target = out / "results.md"
        if src.resolve() != target.resolve():
            shutil.copy2(src, target)
        run.results_md_path = str(target)

    (out / "run_summary.json").write_text(
        json.dumps(
            {
                "task_id": run.task_id,
                "run_id": run.run_id,
                "exit_code": run.exit_code,
                "wall_s": run.wall_s,
                "timed_out": run.timed_out,
                "error": run.error,
                "results_md_path": run.results_md_path,
                "session_id": run.session_id,
                "session_jsonl_path": run.session_jsonl_path,
                "duration_api_ms": run.duration_api_ms,
                "num_turns": run.num_turns,
                "input_tokens": run.input_tokens,
                "output_tokens": run.output_tokens,
            },
            indent=2,
        )
    )
    return run


# ---------------------------------------------------------------------------
# Top-level RPC endpoint
# ---------------------------------------------------------------------------
@CLAUDE_RUN_FSM.fast_service
def run_task(
    request: RunTaskRequest, fsm_context: RequestContext = None
) -> RunTaskResponse:
    t0_total = time.perf_counter()
    task = load_task(task_md_path=request.task_md_path)
    output_dir = prepare_workspace(
        output_root=request.output_root,
        task_id=task.task_id,
        workspace_root=request.workspace_root,
    )
    task_workspace = stage_workspace_files(task=task, output_dir=output_dir)
    run = launch_claude(
        task=task,
        model=request.model,
        output_dir=output_dir,
        task_workspace=task_workspace,
        timeout_override_s=request.timeout_override_s,
        effort=request.effort,
    )
    results_md_path = str(Path(task_workspace) / "results" / "results.md")
    run = collect_outputs(run=run, results_md_path=results_md_path)

    # Postprocess session JSONL into latency breakdown CSVs
    if run.session_jsonl_path:
        try:
            from agsbench_claude.postprocess import postprocess

            postprocess(
                session_jsonl_path=Path(run.session_jsonl_path),
                output_dir=Path(run.output_dir),
                task_id=run.task_id,
                run_id=run.run_id,
            )
        except Exception as e:
            log.warning("[%s] postprocess failed: %s", run.task_id, e)

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
        output_dir=run.output_dir,
        session_jsonl_path=run.session_jsonl_path,
        duration_api_ms=run.duration_api_ms,
        num_turns=run.num_turns,
    )


# ============================================================================
# TerminalBench endpoint
# ============================================================================

from agsbench_claude import terminal_bench_runner as _tb_runner


class RunTerminalBenchRequest(BaseModel):
    """Request to run claude on a single TerminalBench task."""

    task_id: str
    task: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Task dict with at least 'task_id' and 'instruction'. "
            "Optionally 'task_dir' for a pre-downloaded local task directory."
        ),
    )
    output_dir: str = Field(default="/workspace/claude/trajectories_tb")
    task_cache_dir: str = Field(
        default="/workspace/claude/.cache/terminal_bench_tasks",
        description="Directory where terminal-bench tasks are cached.",
    )
    model_name: Optional[str] = Field(default=None)
    step_timeout: int = Field(default=900)
    test_timeout: int = Field(default=120)

    class Config:
        arbitrary_types_allowed = True


class RunTerminalBenchResponse(BaseModel):
    success: bool
    task_id: str
    model_name: str
    tests_passed: int = 0
    tests_total: int = 0
    exit_status: Optional[str] = None
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


@CLAUDE_RUN_FSM.fast_service
def run_terminal_bench_task(
    req: RunTerminalBenchRequest,
    fsm_context: RequestContext = None,
) -> RunTerminalBenchResponse:
    model_override = req.model_name or os.environ.get("CLAUDE_MODEL")
    result = _tb_runner.run_one_terminal_bench(
        task=req.task,
        output_dir=req.output_dir,
        task_cache_dir=req.task_cache_dir,
        fsm_context=fsm_context,
        model_name_override=model_override,
        step_timeout=req.step_timeout,
        test_timeout=req.test_timeout,
    )
    return RunTerminalBenchResponse(
        success=result.success,
        task_id=result.task_id,
        model_name=result.model_name,
        tests_passed=result.tests_passed,
        tests_total=result.tests_total,
        exit_status=result.exit_status,
        error=result.error,
    )
