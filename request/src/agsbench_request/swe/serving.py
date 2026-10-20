"""Software-engineering scenario serving contract.

This file is the central entry point for developers to understand what the
swe scenario requires from an agent service. It documents:

  - The wire protocol (Request/Result models + RPC endpoint)
  - What the agent must implement to serve this scenario
  - How the scenario evaluates results

The actual wire protocol models live in the SDK:
    sdk/scenario-protocol/src/scenario_protocol/request/swe.py

Both the loadgen (request/swe/load_gen.py) and any agent
that serves this scenario import the protocol from the SDK — neither depends
on the other's code.

Supported agents:
  - mini-swe-agent: serves RPC "agsbench_mini_swe_agent.request.solve_instance"
  - (future) codex: will serve the same RPC name once migrated

To add a new agent that serves this scenario:
  1. Implement a fast-service function at the RPC endpoint path
  2. The function must accept SWETaskRequest and return SWETaskResult
  3. Document the agent in agents/<name>/serving.py
"""

from scenario_protocol.scenarios.swe import (
    SWETaskRequest,
    SWETaskResult,
    RPC_NAME,
    PARAM_NAME,
)

# Re-export for convenience
__all__ = ["SWETaskRequest", "SWETaskResult", "RPC_NAME", "PARAM_NAME"]

# Scenario metadata
SCENARIO_NAME = "swe"
DESCRIPTION = "SWE-bench task solving — agent receives a GitHub issue and produces a git diff patch"
DATASET = "princeton-nlp/SWE-bench"
EVALUATION = "swebench harness — applies the patch and runs FAIL_TO_PASS + PASS_TO_PASS tests"
