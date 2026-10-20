"""Benchmark wrapper around browsergym."""

from agsbench_webagent.fsm import agsbench_webagent_fsm
from agsbench_webagent.request import solve_task, SolveTaskRequest, SolveTaskResponse

__all__ = [
    "agsbench_webagent_fsm",
    "solve_task",
    "SolveTaskRequest",
    "SolveTaskResponse",
]
