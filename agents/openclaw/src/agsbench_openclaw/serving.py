"""openclaw serving interface."""
from scenario_protocol.scenarios.office_work import (
    OfficeTaskRequest, OfficeTaskResult, RPC_NAME, PARAM_NAME,
)
AGENT_NAME = "openclaw"
SUPPORTED_SCENARIOS = {
    "office-work": (RPC_NAME, PARAM_NAME, OfficeTaskRequest, OfficeTaskResult),
}
