"""Shared logic for all scenario loadgens.

Each scenario's loadgen.py provides three things:
  - extra_args(parser)  — add scenario-specific CLI args to the argparse parser
  - line_to_request(line, args) -> BaseModel  — parse a manifest line into a Request
  - is_success(result) -> bool  — determine success from a Result

This module provides the shared run_benchmark that wires them together
with ScenarioClient + FastServiceBenchmark.
"""

import os
import sys
from typing import Callable, Type

from pydantic import BaseModel

from fast_service import FastServiceBenchmark, FastServiceBenchmarkConfig

from scenario_protocol import ScenarioClient


def resolve_manifest_path(line: str, args: object) -> str:
    """Resolve a manifest entry; relative paths are relative to the manifest dir."""
    path = line.strip()
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(args.bm_file_path), path)
    return path


def run_benchmark(
    endpoint: str,
    bm_file_path: str,
    bm_mode: str,
    bm_request_num: int,
    bm_rps: float,
    rpc_name: str,
    param_name: str,
    request_type: Type[BaseModel],
    result_type: Type[BaseModel],
    line_to_request: Callable[[str, object], BaseModel],
    is_success: Callable[[BaseModel], bool],
    args: object,
):
    """Shared benchmark runner — used by the unified loadgen entry point.

    Args:
        endpoint: Agent service URL (e.g. "http://mini-sweagent-service:1801")
        bm_file_path: Path to the manifest file
        bm_mode: Scheduling mode (one-by-one, poisson, etc.)
        bm_request_num: Number of requests to dispatch
        bm_rps: Target requests per second (for poisson mode)
        rpc_name: Fast-service RPC endpoint path
        param_name: Fast-service function parameter name for body wrapping
        request_type: Pydantic model class for the request
        result_type: Pydantic model class for the result
        line_to_request: Callable that parses a manifest line into a Request
        is_success: Callable that determines success from a Result
        args: Parsed CLI args (passed to line_to_request for scenario-specific fields)
    """
    client = ScenarioClient(endpoint=endpoint)
    if not client.health():
        print(f"ERROR: agent service at {endpoint} is not reachable", file=sys.stderr)
        sys.exit(1)
    print(f"Agent service at {endpoint} is healthy.", flush=True)

    def _line_to_request(line: str) -> BaseModel:
        return line_to_request(line, args)

    def _invoke_service(request: BaseModel) -> bool:
        result = client.dispatch(
            rpc_name, request, response_type=result_type, param_name=param_name
        )
        return is_success(result) if result else False

    benchmark = FastServiceBenchmark(
        config=FastServiceBenchmarkConfig(
            file_path=bm_file_path,
            mode=bm_mode,
            lambda_rate=bm_rps,
            request_num=bm_request_num,
        ),
        line_to_request=_line_to_request,
        invoke_service=_invoke_service,
    )
    print("start benchmark", flush=True)
    benchmark.execute()
    print("end benchmark", flush=True)
    print("Benchmarking Done !", flush=True)
