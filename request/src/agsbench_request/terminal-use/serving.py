"""Terminal-use scenario serving contract.

Central entry point for developers to understand what the terminal-use
scenario requires from an agent service.

Supported agents:
  - codex: serves RPC "agsbench_codex.request.solve_terminal_bench"
"""

from scenario_protocol.scenarios.terminal_use import (
    TerminalTaskRequest,
    TerminalTaskResult,
    RPC_NAME,
    PARAM_NAME,
)

__all__ = ["TerminalTaskRequest", "TerminalTaskResult", "RPC_NAME", "PARAM_NAME"]

SCENARIO_NAME = "terminal-use"
DESCRIPTION = "TerminalBench task solving — agent works in the task's sandbox container until the task's tests pass"
DATASET = "laude-institute/terminal-bench (original-tasks, pinned commit)"
EVALUATION = "the task's run-tests.sh (pytest) inside the sandbox after the agent finishes"
