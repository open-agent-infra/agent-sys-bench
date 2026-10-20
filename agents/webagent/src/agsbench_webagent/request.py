"""Fast-service entrypoint for the browsergym benchmark service.

Exposes a single RPC, :func:`solve_task`, that drives one browsergym Gym
task end-to-end with a DeepSeek-backed agent.

NB: do NOT add ``from __future__ import annotations`` here. fast-service's
fastapi engine reads ``f.__annotations__["return"]`` to look up the response
class for client-side deserialization; PEP 563 lazy annotations would make
it a string and break the RPC.
"""

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from fast_service import RequestContext

from agsbench_webagent.fsm import agsbench_webagent_fsm

# Importing runner here ensures that any @agsbench_webagent_fsm.fast_service
# decorators in runner.py fire at import time and the functions are
# registered with the fsm before the service rewriter runs.
from agsbench_webagent import runner as _runner


class SolveTaskRequest(BaseModel):
    task_id: str = Field(
        description="Gym task id, e.g. 'browsergym/miniwob.click-test'. Pass the full registered id including the 'browsergym/' prefix.",
    )
    instance_id: str = Field(
        description="Logical id used for output paths (e.g. 'miniwob-0000'). Independent of task_id so the same task_id can be replayed with different seeds.",
    )
    seed: int = Field(default=0)
    max_steps: int = Field(default=50)
    output_dir: str = Field(
        default="/workspace/webagent/trajectories",
    )


class SolveTaskResponse(BaseModel):
    success: bool
    instance_id: str
    task_id: str
    reward: float = 0.0
    n_steps: int = 0
    terminated: bool = False
    truncated: bool = False
    error: Optional[str] = None
    actions: List[str] = []


@agsbench_webagent_fsm.fast_service
def solve_task(
    req: SolveTaskRequest,
    fsm_context: RequestContext = None,
) -> SolveTaskResponse:
    """Run a single browsergym task with the DeepSeek-backed mini-agent."""
    result = _runner.run_one(
        task_id=req.task_id,
        instance_id=req.instance_id,
        seed=req.seed,
        max_steps=req.max_steps,
        output_dir=req.output_dir,
        fsm_context=fsm_context,
    )
    return SolveTaskResponse(
        success=result.success,
        instance_id=result.instance_id,
        task_id=result.task_id,
        reward=result.reward,
        n_steps=result.n_steps,
        terminated=result.terminated,
        truncated=result.truncated,
        error=result.error,
        actions=result.actions,
    )
