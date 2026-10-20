"""deep-research serving interface."""
from scenario_protocol.scenarios.ai_search import AISearchRequest, AISearchResult, RPC_NAME, PARAM_NAME
AGENT_NAME = "deep-research"
SUPPORTED_SCENARIOS = {
    "ai-search": (RPC_NAME, PARAM_NAME, AISearchRequest, AISearchResult),
}
