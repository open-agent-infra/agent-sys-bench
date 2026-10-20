"""claude serving interface."""
from scenario_protocol.scenarios.tool_rich_assistant import (
    ToolRichTaskRequest, ToolRichTaskResult, RPC_NAME, PARAM_NAME,
)
AGENT_NAME = "claude"
SUPPORTED_SCENARIOS = {
    "tool-rich-assistant": (RPC_NAME, PARAM_NAME, ToolRichTaskRequest, ToolRichTaskResult),
}
