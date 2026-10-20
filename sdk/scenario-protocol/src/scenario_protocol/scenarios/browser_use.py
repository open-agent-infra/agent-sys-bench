"""Browser-use scenario wire protocol.
POST {"req": BrowserTaskRequest(...)} → BrowserTaskResult JSON
RPC: "agsbench_webagent.request.solve_task"
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from scenario_protocol.interface import ScenarioRequest, ScenarioResult

class BrowserTaskRequest(ScenarioRequest):
    task_id: str = Field(description="Gym task id, e.g. 'browsergym/miniwob.click-test'")
    instance_id: str = Field(description="Logical id for output paths")
    seed: int = Field(default=0)
    max_steps: int = Field(default=50)
    output_dir: str = Field(default="/workspace/webagent/trajectories")

class BrowserTaskResult(ScenarioResult):
    success: bool
    instance_id: str
    task_id: str
    reward: float = 0.0
    n_steps: int = 0

RPC_NAME = "agsbench_webagent.request.solve_task"
PARAM_NAME = "req"
