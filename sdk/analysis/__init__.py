"""AgentSysBench analysis SDK — data extraction + metric computation layer.

Reads the heterogeneous raw data each agent emits under
``benchmark/data/<setting>/`` and aligns it to a unified record schema so
that the 4 standard metrics (e2e latency, latency breakdown, IO size,
token consumption) can be computed agent-agnostically.

This layer does **only** data preparation + computation. Presentation lives
in ``benchmark/analysis/``.
"""

from .config import CATEGORIES, AGENTS, SETTING_AGENT, AGENT_META  # noqa: F401
from .schema import AnalysisData  # noqa: F401
