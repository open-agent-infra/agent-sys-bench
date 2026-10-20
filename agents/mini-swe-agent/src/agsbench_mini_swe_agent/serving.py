"""mini-swe-agent serving interface — which request this agent supports.

This file is the central entry point for developers to understand what
request the mini-swe-agent can serve and at which RPC endpoints.

Supported request:
  - swe: SWE-bench task solving

The wire protocol is defined in the SDK:
    sdk/scenario-protocol/src/scenario_protocol.scenarios/swe.py

The agent's fast-service function is registered at:
    agents/mini-swe-agent/src/agsbench_mini_swe_agent/request.py :: solve_instance()

Both the loadgen and this agent import the protocol from the SDK — the
loadgen does NOT import this agent's code, and this agent does NOT import
the loadgen's code.
"""

from scenario_protocol.scenarios.swe import (
    SWETaskRequest,
    SWETaskResult,
    RPC_NAME,
    PARAM_NAME,
)

# Agent metadata
AGENT_NAME = "mini-swe-agent"

# Map of scenario_name -> (rpc_name, param_name, request_type, result_type)
SUPPORTED_SCENARIOS = {
    "swe": (RPC_NAME, PARAM_NAME, SWETaskRequest, SWETaskResult),
}
