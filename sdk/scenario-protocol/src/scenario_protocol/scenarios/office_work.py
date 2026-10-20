"""Office-work scenario wire protocol.

Wire contract:
  - POST {"request": OfficeTaskRequest(...)} → OfficeTaskResult JSON
  - RPC endpoint: "agsbench_openclaw.run_task.run_task"
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from scenario_protocol.interface import ScenarioRequest, ScenarioResult


class OfficeTaskRequest(ScenarioRequest):
    task_md_path: str = Field(description="Path to the task .md file")
    model: str = Field(default="deepseek-instrumented/deepseek-v4-flash")
    output_root: str = Field(default="./output")
    timeout_override_s: int = Field(default=0)
    thinking_level: str = Field(default="")
    sandbox: bool = Field(default=False)
    sandbox_mode: str = Field(default="")
    sandbox_workspace: str = Field(default="")
    sessions_root: str = Field(default="")
    proxy_audit_jsonl: str = Field(default="")


class OfficeTaskResult(ScenarioResult):
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


RPC_NAME = "agsbench_openclaw.run_task.run_task"
PARAM_NAME = "request"
