"""QA scenario wire protocol — retrieval-augmented generation.

Wire contract:
  - POST {"item": QARequest(...)} → QAResult JSON
  - RPC endpoint: "agsbench_rag.rag_e2e.naive_rag"

The service-side RAGE2EInput has optional settings fields that default
to None — when None, the service uses its own config (settings.yml).
So the loadgen only needs to pass query + uri + type.
"""

from __future__ import annotations
from typing import List, Union
from pydantic import BaseModel, Field
from scenario_protocol.interface import ScenarioRequest, ScenarioResult


class QARequest(ScenarioRequest):
    query: str = Field(description="The question to answer")
    uri: Union[str, List[str]] = Field(description="Path(s) to document(s) to ingest")
    type: str = Field(default="local_html", description="Document type")
    clear_collection: bool = Field(default=True)


class QAResult(ScenarioResult):
    answer: str


RPC_NAME = "agsbench_rag.rag_e2e.naive_rag"
PARAM_NAME = "item"
