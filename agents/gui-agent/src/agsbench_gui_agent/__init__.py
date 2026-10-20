"""Benchmark wrapper around OSWorld."""

from agsbench_gui_agent.fsm import agsbench_gui_agent_fsm
from agsbench_gui_agent.request import solve_task, SolveTaskRequest, SolveTaskResponse

__all__ = [
    "agsbench_gui_agent_fsm",
    "solve_task",
    "SolveTaskRequest",
    "SolveTaskResponse",
]
