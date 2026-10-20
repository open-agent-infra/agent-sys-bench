"""Tool-rich-assistant scenario wire protocol.

Wire contract:
  - POST {"req": ToolRichTaskRequest(...)} → ToolRichTaskResult JSON
  - RPC endpoint: "agsbench_claude.run_task.run_task"
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from scenario_protocol.interface import ScenarioRequest, ScenarioResult


class ToolRichTaskRequest(ScenarioRequest):
    task_md_path: str = Field(description="Path to the task .md file (inside the agent service container)")
    model: str = Field(default="deepseek-v4-pro")
    output_root: str = Field(default="./output")
    timeout_override_s: int = Field(default=0)
    effort: str = Field(default="")
    workspace_root: str = Field(default="")


class ToolRichTaskResult(ScenarioResult):
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


RPC_NAME = "agsbench_claude.run_task.run_task"
PARAM_NAME = "request"
