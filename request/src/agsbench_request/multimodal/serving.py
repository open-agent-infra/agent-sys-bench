"""Multimodal scenario serving contract."""
from scenario_protocol.scenarios.multimodal import (
    MultimodalRequest, MultimodalResult, RPC_NAME, PARAM_NAME,
)
__all__ = ["MultimodalRequest", "MultimodalResult", "RPC_NAME", "PARAM_NAME"]
SCENARIO_NAME = "multimodal"
DESCRIPTION = "Multi-model orchestration for vision+language tasks"
DATASET = "HuggingGPT hybrid image-text benchmark"
EVALUATION = "Response quality (LLM judge)"
