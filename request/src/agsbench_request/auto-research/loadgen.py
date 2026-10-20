"""Auto-research scenario loadgen — scenario-specific logic only."""
import json
from scenario_protocol.scenarios.auto_research import AutoResearchTaskRequest, AutoResearchTaskResult, RPC_NAME, PARAM_NAME

Request = AutoResearchTaskRequest
Result = AutoResearchTaskResult

def extra_args(parser):
    parser.add_argument("--max_iterations", default=5, type=int)
    parser.add_argument("--timeout", default=3600, type=int)
    parser.add_argument("--output_dir", default="/workspace/pi_autoresearch/trajectories")

def line_to_request(line, args):
    data = json.loads(line)
    return AutoResearchTaskRequest(
        task_id=data["task_id"], competition_id=data.get("competition_id", data["task_id"]),
        max_iterations=args.max_iterations, timeout=args.timeout, output_dir=args.output_dir,
    )

def is_success(result):
    return bool(result.success)
