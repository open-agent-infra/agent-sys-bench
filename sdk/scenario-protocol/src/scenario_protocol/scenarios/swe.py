"""Software-engineering scenario wire protocol — SWE-bench task solving.

Defines the Request/Result models and RPC endpoint name for the
swe scenario. Both the scenario loadgen and any agent
that serves this scenario import from here — neither depends on the other.

Wire contract:
  - Agent's fast-service endpoint must accept POST with JSON body
    {"req": SWETaskRequest(...)} and return SWETaskResult JSON.
  - RPC endpoint path: "agsbench_mini_swe_agent.request.solve_instance"
    (agents serving this scenario must register a fast-service function
    at this module.function path).

Future: standardize the RPC path to "se.solve_instance" so it's
agent-agnostic; for now we match the existing mini-swe-agent endpoint
to preserve wire compatibility during the migration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from scenario_protocol.interface import ScenarioRequest, ScenarioResult


class SWETaskRequest(ScenarioRequest):
    """Request to solve one SWE-bench instance."""

    instance_id: str = Field(..., description="SWE-bench instance ID (e.g. 'django__django-16379')")
    instance: Dict[str, Any] = Field(
        ...,
        description="Raw SWE-bench instance JSON (problem_statement, repo, base_commit, FAIL_TO_PASS, etc.)",
    )
    config_path: str = Field(
        default="/workspace/agsbench_mini_swe_agent/config/default.yaml",
        description="Path (inside the agent service container) to the agent's YAML config.",
    )
    output_dir: str = Field(
        default="/workspace/agsbench_mini_swe_agent/trajectories",
        description="Directory (inside the agent service container) for trajectory output.",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional override for the litellm model name (e.g. 'openai/deepseek-v4-pro').",
    )


class SWETaskResult(ScenarioResult):
    """Result of solving one SWE-bench instance."""

    success: bool
    instance_id: str
    model_name: str
    patch: str = ""
    exit_status: Optional[str] = None
    error: Optional[str] = None


# The fast-service RPC endpoint path (module.function).
# The agent service must expose a function at this path.
RPC_NAME = "agsbench_mini_swe_agent.request.solve_instance"

# The fast-service function parameter name (for body wrapping).
# Agents should name their request parameter "req" to match this convention.
PARAM_NAME = "req"
