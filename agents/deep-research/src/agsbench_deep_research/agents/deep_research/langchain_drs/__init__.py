"""
Deep Research Agent based on LangChain's Deep Research System.

This module provides a complete deep research workflow implementation
following LegoAgent's patterns and architecture.

The code is referenced from https://github.com/langchain-ai/open_deep_research
"""

from .utils import get_drs_config_value
from .shared import (
    agsbench_deep_research_fsm,
    DEFAULT_SEARCH_POST_PROCESSING_CONFIG,
    DEFAULT_VDB_COLLECTION,
)
from .workflow import (
    DeepResearchWorkflowRequest,
    DeepResearchWorkflowResponse,
    deep_research_workflow,
    deep_research_workflow_async,
)

__all__ = [
    "DeepResearchWorkflowRequest",
    "DeepResearchWorkflowResponse",
    "deep_research_workflow",
    "deep_research_workflow_async",
    "get_drs_config_value",
    "agsbench_deep_research_fsm",
    "DEFAULT_SEARCH_POST_PROCESSING_CONFIG",
    "DEFAULT_VDB_COLLECTION",
]
