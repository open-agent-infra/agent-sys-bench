"""AI-search scenario wire protocol — iterative deep research.

Wire contract:
  - POST {"request": AISearchRequest(...)} → AISearchResult JSON
  - RPC: "agsbench_deep_research.agents.deep_research.langchain_drs.workflow.deep_research.deep_research_workflow"
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from scenario_protocol.interface import ScenarioRequest, ScenarioResult


class AISearchRequest(ScenarioRequest):
    topic: str = Field(description="The research topic/question")


class AISearchResult(ScenarioResult):
    task_name: str = "deep_research_workflow"
    final_report: str = Field(default="", description="The complete research report")


RPC_NAME = "agsbench_deep_research.agents.deep_research.langchain_drs.workflow.deep_research.deep_research_workflow"
PARAM_NAME = "request"
