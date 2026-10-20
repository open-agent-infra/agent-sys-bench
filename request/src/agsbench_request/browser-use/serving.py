"""Browser-use scenario serving contract."""
from scenario_protocol.scenarios.browser_use import (
    BrowserTaskRequest, BrowserTaskResult, RPC_NAME, PARAM_NAME,
)
__all__ = ["BrowserTaskRequest", "BrowserTaskResult", "RPC_NAME", "PARAM_NAME"]
SCENARIO_NAME = "browser-use"
DESCRIPTION = "Browser automation via BrowserGym + Playwright"
DATASET = "MiniWoB++ / WebArena verified"
EVALUATION = "Environment reward (task success/fail)"
