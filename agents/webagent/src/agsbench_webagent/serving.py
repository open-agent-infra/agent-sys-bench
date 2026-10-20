"""webagent serving interface."""
from scenario_protocol.scenarios.browser_use import (
    BrowserTaskRequest, BrowserTaskResult, RPC_NAME, PARAM_NAME,
)
AGENT_NAME = "webagent"
SUPPORTED_SCENARIOS = {
    "browser-use": (RPC_NAME, PARAM_NAME, BrowserTaskRequest, BrowserTaskResult),
}
