"""Auto-research scenario wire protocol — AutoML research on MLE-bench.

Wire contract:
  - POST {"req": AutoResearchTaskRequest(...)} → AutoResearchTaskResult JSON
  - RPC endpoint: "agsbench_pi_autoresearch.request.solve_task"
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from scenario_protocol.interface import ScenarioRequest, ScenarioResult


class AutoResearchTaskRequest(ScenarioRequest):
    task_id: str
    competition_id: str = Field(description="MLE-bench competition id.")
    data_root: str = Field(
        default="/workspace/pi_autoresearch/mlebench-public",
        description="Directory holding <competition_id>/ with the public (agent-visible) data.",
    )
    max_iterations: int = Field(default=5)
    timeout: int = Field(default=3600)
    output_dir: str = Field(default="/workspace/pi_autoresearch/trajectories")
    pi_binary: str = Field(default="pi", description="Path to the pi binary.")


class AutoResearchTaskResult(ScenarioResult):
    success: bool
    task_id: str
    competition_id: Optional[str] = None
    # MLE-bench grade of the final submission/submission.csv.
    score: Optional[float] = None
    valid_submission: bool = False
    any_medal: bool = False
    medal: Optional[str] = None
    above_median: bool = False
    grade_report: Optional[Dict[str, Any]] = None
    # Agent's own (CV) metric as logged by the autoresearch extension.
    final_metric: Optional[float] = None
    baseline_metric: Optional[float] = None
    improvement_pct: Optional[float] = None
    n_experiments: int = 0
    n_kept: int = 0
    duration_seconds: float = 0.0
    timed_out: bool = False
    error: Optional[str] = None
    token_usage: Optional[Dict[str, Any]] = Field(default=None)


RPC_NAME = "agsbench_pi_autoresearch.request.solve_task"
PARAM_NAME = "req"
