"""Terminal-use scenario wire protocol — TerminalBench task solving.

Defines the Request/Result models and RPC endpoint name for the
terminal-use scenario. Both the scenario loadgen and any agent that
serves this scenario import from here — neither depends on the other.

Wire contract:
  - Agent's fast-service endpoint must accept POST with JSON body
    {"req": TerminalTaskRequest(...)} and return TerminalTaskResult JSON.
  - RPC endpoint path: "agsbench_codex.request.solve_terminal_bench"
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from scenario_protocol.interface import ScenarioRequest, ScenarioResult


class TerminalTaskRequest(ScenarioRequest):
    """Request to solve one TerminalBench task."""

    task_id: str = Field(..., description="TerminalBench task id (e.g. 'hello-world')")
    task: Dict[str, Any] = Field(
        ..., description="Manifest record (task_id, instruction, difficulty, category, ...)"
    )
    output_dir: str = Field(
        default="/workspace/codex/trajectories",
        description="Directory (inside the agent service container) for trajectory output.",
    )
    task_cache_dir: str = Field(
        default="/workspace/codex/tb-tasks",
        description="Directory (inside the agent service container) holding one subdir per task_id.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional model name override (e.g. 'deepseek-v4-pro').",
    )
    step_timeout: int = Field(
        default=900,
        description="Hard wall-clock cap (seconds) on the agent run.",
    )
    test_timeout: int = Field(
        default=180,
        description="Hard wall-clock cap (seconds) on the task's test run.",
    )


class TerminalTaskResult(ScenarioResult):
    """Result of solving one TerminalBench task."""

    success: bool
    task_id: str
    model_name: str
    tests_passed: int = 0
    tests_total: int = 0
    exit_status: Optional[str] = None
    error: Optional[str] = None


RPC_NAME = "agsbench_codex.request.solve_terminal_bench"
PARAM_NAME = "req"
