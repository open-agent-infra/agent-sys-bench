"""Computer-use scenario wire protocol — OSWorld desktop GUI tasks.

Wire contract:
  - POST {"req": ComputerTaskRequest(...)} → ComputerTaskResult JSON
  - RPC: "agsbench_gui_agent.request.solve_task"
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from scenario_protocol.interface import ScenarioRequest, ScenarioResult


class ComputerTaskRequest(ScenarioRequest):
    task_id: str = Field(description="OSWorld task ID")
    task_config: Dict[str, Any] = Field(description="Full OSWorld task JSON (instruction, config, evaluator)")
    output_dir: str = Field(default="/workspace/agsbench_gui_agent/trajectories")
    max_steps: int = Field(default=15)
    observation_type: str = Field(
        default="a11y_tree",
        description="One of: 'a11y_tree', 'screenshot', 'screenshot_a11y_tree'",
    )


class ComputerTaskResult(ScenarioResult):
    success: bool
    task_id: str
    reward: float = 0.0
    n_steps: int = 0
    error: Optional[str] = None


RPC_NAME = "agsbench_gui_agent.request.solve_task"
PARAM_NAME = "req"
