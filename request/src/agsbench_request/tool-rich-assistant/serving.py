"""Tool-rich-assistant scenario serving contract."""
from scenario_protocol.scenarios.tool_rich_assistant import (
    ToolRichTaskRequest, ToolRichTaskResult, RPC_NAME, PARAM_NAME,
)
__all__ = ["ToolRichTaskRequest", "ToolRichTaskResult", "RPC_NAME", "PARAM_NAME"]
SCENARIO_NAME = "tool-rich-assistant"
DESCRIPTION = "Multi-step natural-language tasks requiring file editing, shell commands, and MCP tool use"
DATASET = "WildClawBench / PinchBench"
EVALUATION = "External grader — compares agent output against ground truth"
