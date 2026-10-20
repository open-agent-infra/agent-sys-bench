"""pi-autoresearch serving interface."""
from scenario_protocol.scenarios.auto_research import (
    AutoResearchTaskRequest, AutoResearchTaskResult, RPC_NAME, PARAM_NAME,
)
AGENT_NAME = "pi-autoresearch"
SUPPORTED_SCENARIOS = {
    "auto-research": (RPC_NAME, PARAM_NAME, AutoResearchTaskRequest, AutoResearchTaskResult),
}
