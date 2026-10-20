"""Fast-service entrypoint for the codex benchmark service.

Exposes ``solve_terminal_bench``, which runs ``codex exec --json --yolo``
inside a TerminalBench task's own sandbox container (built from the task's
docker-compose.yaml), then scores the result with the task's run-tests.sh.
See terminal_bench_runner.py.

NB: no ``from __future__ import annotations`` (see fast-service-gotchas).
"""

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from fast_service import RequestContext

from agsbench_codex.fsm import codex_fsm
from agsbench_codex import terminal_bench_runner as _tb_runner


class SolveTerminalBenchRequest(BaseModel):
    """Request to run codex on a single TerminalBench task."""

    task_id: str
    task: Dict[str, Any] = Field(
        description=(
            "Task dict with at least 'task_id' and 'instruction'. "
            "Optionally 'task_dir' for a pre-downloaded local task directory."
        )
    )
    output_dir: str = Field(default="/workspace/codex/trajectories")
    task_cache_dir: str = Field(
        default="/workspace/codex/tb-tasks",
        description="Directory where terminal-bench tasks are cached (one subdir per task_id).",
    )
    model_name: Optional[str] = Field(default=None)
    step_timeout: int = Field(default=900)
    test_timeout: int = Field(default=180)


class SolveTerminalBenchResponse(BaseModel):
    success: bool
    task_id: str
    model_name: str
    tests_passed: int = 0
    tests_total: int = 0
    exit_status: Optional[str] = None
    error: Optional[str] = None


@codex_fsm.fast_service
def solve_terminal_bench(
    req: SolveTerminalBenchRequest,
    fsm_context: RequestContext = None,
) -> SolveTerminalBenchResponse:
    model_override = req.model_name or os.getenv("CODEX_MODEL")
    result = _tb_runner.run_one_terminal_bench(
        task=req.task,
        output_dir=req.output_dir,
        task_cache_dir=req.task_cache_dir,
        fsm_context=fsm_context,
        model_name_override=model_override,
        step_timeout=req.step_timeout,
        test_timeout=req.test_timeout,
    )
    return SolveTerminalBenchResponse(
        success=result.success,
        task_id=result.task_id,
        model_name=result.model_name,
        tests_passed=result.tests_passed,
        tests_total=result.tests_total,
        exit_status=result.exit_status,
        error=result.error,
    )
