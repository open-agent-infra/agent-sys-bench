"""scenario-protocol: wire protocol between scenario loadgens and agent services.

The loadgen is a pure HTTP client — it imports no agent code. It constructs
scenario-defined Request models from a manifest, dispatches them via
ScenarioClient to the agent's fast-service endpoint, and deserializes the
response into scenario-defined Result models.

Agent side documents which RPC endpoint names it serves for each scenario.
"""

from scenario_protocol.interface import ScenarioRequest, ScenarioResult, ScenarioInterface
from scenario_protocol.client import ScenarioClient

__all__ = [
    "ScenarioRequest",
    "ScenarioResult",
    "ScenarioInterface",
    "ScenarioClient",
]
