"""rag serving interface."""
from scenario_protocol.scenarios.qa import QARequest, QAResult, RPC_NAME, PARAM_NAME
AGENT_NAME = "rag"
SUPPORTED_SCENARIOS = {
    "qa": (RPC_NAME, PARAM_NAME, QARequest, QAResult),
}
