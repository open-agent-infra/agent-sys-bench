"""Multimodal scenario loadgen — scenario-specific logic only."""
from scenario_protocol.scenarios.multimodal import (
    MultimodalRequest, MultimodalResult, RPC_NAME, PARAM_NAME,
)
Request = MultimodalRequest
Result = MultimodalResult

def line_to_request(line, args):
    # Manifest format: "tasks_to_plan, question text"
    # e.g. "12, Please talk about the figure /path/to/image.jpg"
    # The whole line is the question (the number prefix is parsed by the task planner)
    return MultimodalRequest(collection_name="benchmark", question=line.strip())

def is_success(result):
    return bool(result.answer)
