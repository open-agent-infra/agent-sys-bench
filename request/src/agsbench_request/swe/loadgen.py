"""Software-engineering scenario loadgen — scenario-specific logic only.

The unified entry point (request/loadgen.py) handles CLI parsing,
ScenarioClient setup, and FastServiceBenchmark scheduling. This module
only defines what's scenario-specific:
  - extra_args: scenario-specific CLI args
  - line_to_request: manifest → SWETaskRequest
  - is_success: SWETaskResult → bool
"""

import json
from agsbench_request._common import resolve_manifest_path

from scenario_protocol.scenarios.swe import (
    SWETaskRequest,
    SWETaskResult,
    RPC_NAME,
    PARAM_NAME,
)

Request = SWETaskRequest
Result = SWETaskResult


def extra_args(parser):
    parser.add_argument("--config_path", default="/workspace/agsbench_mini_swe_agent/config/default.yaml",
                        help="Agent config path (inside the agent service container).")
    parser.add_argument("--output_dir", default="/workspace/agsbench_mini_swe_agent/trajectories",
                        help="Trajectory output dir (inside the agent service container).")
    parser.add_argument("--model_name", default=None, help="Optional model name override.")


def line_to_request(line, args):
    path = resolve_manifest_path(line, args)
    with open(path, "r", encoding="utf-8") as f:
        instance = json.load(f)
    return SWETaskRequest(
        instance_id=instance["instance_id"],
        instance=instance,
        config_path=args.config_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )


def is_success(result):
    return bool(result.success)
