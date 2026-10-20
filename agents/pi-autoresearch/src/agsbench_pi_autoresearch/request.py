"""Fast-service entrypoint for the pi-autoresearch benchmark service.

Exposes a single RPC, solve_task, that drives one MLE-bench competition
(AutoML research task) end-to-end through the pi coding agent with the
pi-autoresearch extension and returns the grade plus telemetry.

NB: do NOT add ``from __future__ import annotations`` here. fast-service
inspects raw __annotations__ to discover return classes for client-side
deserialization; PEP 563 lazy annotations would break that path.
"""

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from fast_service import RequestContext

from agsbench_pi_autoresearch.fsm import agsbench_pi_autoresearch_fsm

# Import runner to ensure @agsbench_pi_autoresearch_fsm.fast_service decorators
# fire at import time before the service rewriter runs.
from agsbench_pi_autoresearch import runner as _runner


class SolveTaskRequest(BaseModel):
    task_id: str
    competition_id: str = Field(description="MLE-bench competition id.")
    data_root: str = Field(
        default="/workspace/pi_autoresearch/mlebench-public",
        description="Directory holding <competition_id>/ with the public (agent-visible) data.",
    )
    max_iterations: int = Field(
        default=5,
        description="Maximum number of autoresearch iterations before stopping.",
    )
    timeout: int = Field(
        default=3600,
        description="Time limit in seconds for the entire pi agent run.",
    )
    output_dir: str = Field(
        default="/workspace/pi_autoresearch/trajectories",
        description="Directory for preds.json and per-task trajectories.",
    )
    pi_binary: str = Field(
        default="pi",
        description="Path to the pi binary.",
    )
    env_overrides: Optional[Dict[str, str]] = Field(
        default=None,
        description="Extra environment variables for the pi subprocess.",
    )


class SolveTaskResponse(BaseModel):
    success: bool
    task_id: str
    competition_id: Optional[str] = None
    score: Optional[float] = None
    valid_submission: bool = False
    any_medal: bool = False
    medal: Optional[str] = None
    above_median: bool = False
    grade_report: Optional[Dict[str, Any]] = None
    final_metric: Optional[float] = None
    baseline_metric: Optional[float] = None
    improvement_pct: Optional[float] = None
    n_experiments: int = 0
    n_kept: int = 0
    duration_seconds: float = 0.0
    timed_out: bool = False
    error: Optional[str] = None
    token_usage: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-turn token usage parsed from pi session JSONL files.",
    )


@agsbench_pi_autoresearch_fsm.fast_service
def solve_task(
    req: SolveTaskRequest,
    fsm_context: RequestContext = None,
) -> SolveTaskResponse:
    """Run a single MLE-bench competition with pi-autoresearch."""
    result = _runner.run_one(
        task_id=req.task_id,
        competition_id=req.competition_id,
        data_root=req.data_root,
        max_iterations=req.max_iterations,
        timeout=req.timeout,
        output_dir=req.output_dir,
        pi_binary=req.pi_binary,
        env_overrides=req.env_overrides,
        fsm_context=fsm_context,
    )
    return SolveTaskResponse(
        success=result.success,
        task_id=result.task_id,
        competition_id=result.competition_id,
        score=result.score,
        valid_submission=result.valid_submission,
        any_medal=result.any_medal,
        medal=result.medal,
        above_median=result.above_median,
        grade_report=result.grade_report,
        final_metric=result.final_metric,
        baseline_metric=result.baseline_metric,
        improvement_pct=result.improvement_pct,
        n_experiments=result.n_experiments,
        n_kept=result.n_kept,
        duration_seconds=result.duration_seconds,
        timed_out=result.timed_out,
        error=result.error,
        token_usage=result.token_usage,
    )
