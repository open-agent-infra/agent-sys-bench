"""Browser-use scenario loadgen — scenario-specific logic only."""
import json
from agsbench_request._common import resolve_manifest_path
from scenario_protocol.scenarios.browser_use import BrowserTaskRequest, BrowserTaskResult, RPC_NAME, PARAM_NAME

Request = BrowserTaskRequest
Result = BrowserTaskResult

def extra_args(parser):
    parser.add_argument("--max_steps", default=30, type=int)
    parser.add_argument("--output_dir", default="/workspace/webagent/trajectories")

def line_to_request(line, args):
    path = resolve_manifest_path(line, args)
    with open(path) as f:
        data = json.load(f)
    return BrowserTaskRequest(
        task_id=data["task_id"], instance_id=data["instance_id"],
        seed=data.get("seed", 0), max_steps=args.max_steps, output_dir=args.output_dir,
    )

def is_success(result):
    return bool(result.reward > 0)
