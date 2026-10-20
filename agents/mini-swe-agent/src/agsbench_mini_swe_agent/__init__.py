"""Benchmark wrapper around mini-swe-agent.

Exports the fast-service manager and the public service entrypoints. Both
single-host and multi-node modes share this package; the env_service
entrypoints are registered with mini_swe_fsm at import time and fast-service
chooses (per config) whether to serve them locally or generate client stubs.
"""

from agsbench_mini_swe_agent.fsm import mini_swe_fsm
from agsbench_mini_swe_agent.request import (
    solve_instance,
    SolveInstanceRequest,
    SolveInstanceResponse,
)
from agsbench_mini_swe_agent.env_service import (
    env_create,
    env_execute,
    env_destroy,
    EnvCreateRequest,
    EnvCreateResponse,
    EnvExecuteRequest,
    EnvExecuteResponse,
    EnvDestroyRequest,
    EnvDestroyResponse,
)

__all__ = [
    "mini_swe_fsm",
    "solve_instance",
    "SolveInstanceRequest",
    "SolveInstanceResponse",
    "env_create",
    "env_execute",
    "env_destroy",
    "EnvCreateRequest",
    "EnvCreateResponse",
    "EnvExecuteRequest",
    "EnvExecuteResponse",
    "EnvDestroyRequest",
    "EnvDestroyResponse",
]
