#!/usr/bin/env python
"""Unified workload generator entry point.

Usage:
  python -m request.loadgen \\
      --scenario swe \\
      --endpoint http://mini-sweagent-service:1801 \\
      --bm_file_path /data/manifest.txt \\
      --bm_request_num 1 \\
      --bm_mode one-by-one

Each scenario's loadgen.py defines:
  - extra_args(parser)           — add scenario-specific CLI args
  - line_to_request(line, args)  — parse manifest line → Request
  - is_success(result)           — determine success from Result
  - RPC_NAME, PARAM_NAME         — imported from sdk/scenario-protocol
  - Request, Result              — imported from sdk/scenario-protocol
"""

import argparse
import sys

from agsbench_request import load_request_module
from agsbench_request._common import run_benchmark


def main():
    # Phase 1: parse --scenario to know which scenario module to load
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--scenario", required=True, help="Scenario name (e.g. swe)")
    pre_args, remaining = pre_parser.parse_known_args()

    # Load the scenario module
    mod = load_request_module(pre_args.scenario, "loadgen")

    # Verify the module has the required interface
    for attr in ("RPC_NAME", "PARAM_NAME", "Request", "Result",
                 "line_to_request", "is_success"):
        if not hasattr(mod, attr):
            print(f"ERROR: scenario '{pre_args.scenario}' loadgen.py missing required attribute: {attr}", file=sys.stderr)
            sys.exit(1)

    # Phase 2: full parser with common args + scenario-specific args
    parser = argparse.ArgumentParser(
        description=f"Run {pre_args.scenario} scenario against any agent service."
    )
    parser.add_argument("--scenario", required=True, help=argparse.SUPPRESS)  # already parsed
    parser.add_argument("--endpoint", required=True, help="Agent service endpoint URL")
    parser.add_argument("--bm_file_path", default="./.cache/manifest.txt", help="Manifest file path")
    parser.add_argument("--bm_mode", default="one-by-one", help="Scheduling mode")
    parser.add_argument("--bm_request_num", default=1, type=int, help="Number of requests")
    parser.add_argument("--bm_rps", type=float, default=1.0, help="Target requests/sec (poisson mode)")

    # Let the scenario add its own args
    if hasattr(mod, "extra_args"):
        mod.extra_args(parser)

    args = parser.parse_args()

    # Run the shared benchmark logic
    run_benchmark(
        endpoint=args.endpoint,
        bm_file_path=args.bm_file_path,
        bm_mode=args.bm_mode,
        bm_request_num=args.bm_request_num,
        bm_rps=args.bm_rps,
        rpc_name=mod.RPC_NAME,
        param_name=mod.PARAM_NAME,
        request_type=mod.Request,
        result_type=mod.Result,
        line_to_request=mod.line_to_request,
        is_success=mod.is_success,
        args=args,
    )


if __name__ == "__main__":
    main()
