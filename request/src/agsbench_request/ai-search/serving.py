"""AI-search scenario serving contract."""
from scenario_protocol.scenarios.ai_search import AISearchRequest, AISearchResult, RPC_NAME, PARAM_NAME
__all__ = ["AISearchRequest", "AISearchResult", "RPC_NAME", "PARAM_NAME"]
SCENARIO_NAME = "ai-search"
DESCRIPTION = "Iterative deep research — agent searches, summarizes, and synthesizes a report"
DATASET = "HLE / GAIA / FRAMES"
EVALUATION = "LLM judge (report quality + correctness)"
