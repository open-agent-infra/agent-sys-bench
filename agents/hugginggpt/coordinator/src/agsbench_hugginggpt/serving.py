"""hugginggpt serving interface."""
from scenario_protocol.scenarios.multimodal import (
    MultimodalRequest, MultimodalResult, RPC_NAME, PARAM_NAME,
)
AGENT_NAME = "hugginggpt"
SUPPORTED_SCENARIOS = {
    "multimodal": (RPC_NAME, PARAM_NAME, MultimodalRequest, MultimodalResult),
}
