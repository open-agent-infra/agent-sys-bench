"""AI-search scenario loadgen — scenario-specific logic only."""
import json
from scenario_protocol.scenarios.ai_search import AISearchRequest, AISearchResult, RPC_NAME, PARAM_NAME

Request = AISearchRequest
Result = AISearchResult

def line_to_request(line, args):
    data = json.loads(line.strip())
    return AISearchRequest(topic=data.get("prompt", data.get("topic", "")))

def is_success(result):
    return bool(result.final_report)
