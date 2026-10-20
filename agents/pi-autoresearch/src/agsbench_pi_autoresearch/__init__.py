"""pi-autoresearch benchmark integration for AgentSysBench.

Exposes the fast-service manager and the ``solve_task`` RPC so both the
service entrypoint and the load-generator can import from a single place.
"""

from agsbench_pi_autoresearch.fsm import agsbench_pi_autoresearch_fsm
from agsbench_pi_autoresearch.request import SolveTaskRequest, SolveTaskResponse, solve_task

__all__ = [
    "agsbench_pi_autoresearch_fsm",
    "SolveTaskRequest",
    "SolveTaskResponse",
    "solve_task",
]
