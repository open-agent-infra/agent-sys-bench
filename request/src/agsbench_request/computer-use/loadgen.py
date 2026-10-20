"""Computer-use scenario loadgen — scenario-specific logic only."""
import json
from agsbench_request._common import resolve_manifest_path
from scenario_protocol.scenarios.computer_use import (
    ComputerTaskRequest, ComputerTaskResult, RPC_NAME, PARAM_NAME,
)

Request = ComputerTaskRequest
Result = ComputerTaskResult

def extra_args(parser):
    parser.add_argument("--max_steps", default=15, type=int)
    parser.add_argument("--observation_type", default="a11y_tree",
                        choices=["a11y_tree", "screenshot", "screenshot_a11y_tree"])
    parser.add_argument("--output_dir", default="/workspace/agsbench_gui_agent/trajectories")

def line_to_request(line, args):
    path = resolve_manifest_path(line, args)
    with open(path) as f:
        data = json.load(f)
    return ComputerTaskRequest(
        task_id=data["id"],
        task_config=data,
        max_steps=args.max_steps,
        observation_type=args.observation_type,
        output_dir=args.output_dir,
    )

def is_success(result):
    return bool(result.success)
