"""Office-work scenario serving contract."""
from scenario_protocol.scenarios.office_work import (
    OfficeTaskRequest, OfficeTaskResult, RPC_NAME, PARAM_NAME,
)
__all__ = ["OfficeTaskRequest", "OfficeTaskResult", "RPC_NAME", "PARAM_NAME"]
SCENARIO_NAME = "office-work"
DESCRIPTION = "Office task automation — file editing, bash, web search, code execution"
DATASET = "WildClawBench / PinchBench / claw-eval"
EVALUATION = "External grader"
