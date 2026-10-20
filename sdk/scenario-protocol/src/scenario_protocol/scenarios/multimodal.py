"""Multimodal scenario wire protocol — HuggingGPT multi-model orchestration.

Wire contract:
  - POST {"hugginggpt_e2e_input": MultimodalRequest(...)} → MultimodalResult JSON
  - RPC: "agsbench_hugginggpt.hugginggpt_e2e.hugginggpt_e2e"
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field
from scenario_protocol.interface import ScenarioRequest, ScenarioResult


class MultimodalRequest(ScenarioRequest):
    collection_name: str = Field(default="benchmark")
    question: str = Field(description="Multimodal task instruction (text)")
    context_limit: int = Field(default=3)
    tasks_to_plan: Optional[List[int]] = None


class MultimodalResult(ScenarioResult):
    answer: str = ""


RPC_NAME = "agsbench_hugginggpt.hugginggpt_e2e.hugginggpt_e2e"
PARAM_NAME = "hugginggpt_e2e_input"
