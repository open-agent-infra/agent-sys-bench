"""gui-agent serving interface."""
from scenario_protocol.scenarios.computer_use import (
    ComputerTaskRequest, ComputerTaskResult, RPC_NAME, PARAM_NAME,
)
AGENT_NAME = "gui-agent"
SUPPORTED_SCENARIOS = {
    "computer-use": (RPC_NAME, PARAM_NAME, ComputerTaskRequest, ComputerTaskResult),
}
