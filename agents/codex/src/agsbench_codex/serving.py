"""codex serving interface — which request this agent supports."""

from scenario_protocol.scenarios.terminal_use import (
    TerminalTaskRequest,
    TerminalTaskResult,
    RPC_NAME,
    PARAM_NAME,
)

AGENT_NAME = "codex"
SUPPORTED_SCENARIOS = {
    "terminal-use": (RPC_NAME, PARAM_NAME, TerminalTaskRequest, TerminalTaskResult),
}
