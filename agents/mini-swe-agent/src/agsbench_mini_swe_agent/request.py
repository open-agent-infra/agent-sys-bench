"""Fast-service entrypoint for the mini-swe-agent benchmark service.

Exposes a single RPC method, :func:`solve_instance`, that drives one SWE-bench
task end-to-end. Inside, it dispatches to :func:`agsbench_mini_swe_agent.runner.run_one`,
which is responsible for the per-instance timing instrumentation.

NB: do NOT add ``from __future__ import annotations`` here. fast-service's
fastapi engine reads ``f.__annotations__["return"]`` to look up the response
class for client-side deserialization; PEP 563 lazy annotations would make
it a string and break the RPC. See swe_agent/src/sweagent/request.py for
the same caveat.
"""

import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from fast_service import RequestContext

from agsbench_mini_swe_agent.fsm import mini_swe_fsm

# Importing runner here ensures the model_query_call / env_execute_call
# decorators run at import time and the functions are registered with
# ``mini_swe_fsm`` before the service is rewritten.
from agsbench_mini_swe_agent import runner as _runner


class SolveInstanceRequest(BaseModel):
    instance_id: str
    instance: Dict[str, Any]
    config_path: str = Field(
        default="/workspace/agsbench_mini_swe_agent/config/default.yaml",
        description="Path (inside the service container) to a mini-swe-agent YAML config.",
    )
    output_dir: str = Field(
        default="/workspace/agsbench_mini_swe_agent/trajectories",
        description="Directory where preds.json and per-instance trajectories are written.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional override for the litellm model name (e.g. 'openai/gpt-4o-mini').",
    )


class SolveInstanceResponse(BaseModel):
    success: bool
    instance_id: str
    model_name: str
    patch: str = ""
    exit_status: Optional[str] = None
    error: Optional[str] = None


@mini_swe_fsm.fast_service
def solve_instance(
    req: SolveInstanceRequest,
    fsm_context: RequestContext = None,
) -> SolveInstanceResponse:
    """Solve a single SWE-bench instance with mini-swe-agent."""
    model_override = req.model_name or os.getenv("MINI_SWE_MODEL")
    result = _runner.run_one(
        instance=req.instance,
        config_path=req.config_path,
        output_dir=req.output_dir,
        fsm_context=fsm_context,
        model_name_override=model_override,
    )
    return SolveInstanceResponse(
        success=result.success,
        instance_id=result.instance_id,
        model_name=result.model_name,
        patch=result.patch,
        exit_status=result.exit_status,
        error=result.error,
    )
