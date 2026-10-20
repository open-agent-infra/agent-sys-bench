"""Scenario interface protocol — base types for scenario-agent decoupling.

Each scenario package defines its own concrete Request/Result models
(subclassing pydantic BaseModel) and an RPC_NAME string identifying the
agent's fast-service endpoint. The loadgen depends only on these scenario
definitions + ScenarioClient — never on any agent package.
"""

from __future__ import annotations

from typing import Type, TypeVar
from pydantic import BaseModel

# The wire protocol is HTTP POST with JSON body, matching fast-service's
# FastAPI engine. The URL path is f"{endpoint}/{rpc_name}" where rpc_name
# is typically "module_name.function_name" (e.g.
# "agsbench_mini_swe_agent.request.solve_instance").

T = TypeVar("T", bound=BaseModel)


class ScenarioRequest(BaseModel):
    """Base for scenario-specific request models.

    Subclass this in ``request/<name>/interface.py`` to define the wire
    contract for the scenario. The agent's fast-service endpoint must accept
    this schema.
    """

    model_config = {"arbitrary_types_allowed": True}


class ScenarioResult(BaseModel):
    """Base for scenario-specific result models."""

    model_config = {"arbitrary_types_allowed": True}


class ScenarioInterface:
    """Describes a scenario's RPC contract.

    A scenario package defines:
      - RPC_NAME: the fast-service endpoint path (module.function)
      - Request: the pydantic model for the request body
      - Result: the pydantic model for the response body

    Example::

        class SEInterface(ScenarioInterface):
            RPC_NAME = "agsbench_mini_swe_agent.request.solve_instance"
            Request = SWETaskRequest
            Result = SWETaskResult
    """

    RPC_NAME: str = ""
    Request: Type[ScenarioRequest]
    Result: Type[ScenarioResult]
