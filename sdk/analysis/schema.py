"""Unified analysis record schema.

Each agent extractor produces an ``AnalysisData`` holding four pandas
DataFrames with fixed columns. All downstream metric computation is
agent-agnostic and operates only on these schemas.
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd


# Column contracts (documented; not enforced at runtime to keep extractors simple).
LATENCY_COLUMNS = [
    "request_id", "call_id", "parent_call_id", "function_name", "category",
    "start_time", "end_time", "delay", "delay_type", "is_top_level",
]
IO_COLUMNS = [
    "request_id", "call_id", "function_name", "category",
    "direction", "bytes", "tokens",
]
TOKEN_COLUMNS = [
    "request_id", "call_id", "function_name", "category", "model",
    "input_tokens", "output_tokens", "cache_read_tokens",
    "cache_write_tokens", "reasoning_tokens", "total_tokens", "cost_usd",
]
REQUEST_SUMMARY_COLUMNS = [
    "request_id", "e2e_latency", "success", "n_steps", "n_llm_calls",
    "total_input_tokens", "total_output_tokens", "total_cost_usd", "error",
]


@dataclass
class AnalysisData:
    """The normalized output of one agent's extraction."""
    agent: str
    latency: pd.DataFrame = field(default_factory=pd.DataFrame)
    io: pd.DataFrame = field(default_factory=pd.DataFrame)
    tokens: pd.DataFrame = field(default_factory=pd.DataFrame)
    requests: pd.DataFrame = field(default_factory=pd.DataFrame)
    # extraction-time diagnostics (what was found / missed)
    diagnostics: List[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.latency.empty and self.tokens.empty and self.io.empty and self.requests.empty
