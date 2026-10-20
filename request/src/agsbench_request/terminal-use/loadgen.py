"""Terminal-use scenario loadgen — scenario-specific logic only."""

import json

from scenario_protocol.scenarios.terminal_use import (
    TerminalTaskRequest,
    TerminalTaskResult,
    RPC_NAME,
    PARAM_NAME,
)

Request = TerminalTaskRequest
Result = TerminalTaskResult


def extra_args(parser):
    parser.add_argument("--output_dir", default="/workspace/codex/trajectories")
    parser.add_argument("--task_cache_dir", default="/workspace/codex/tb-tasks")
    parser.add_argument("--model_name", default=None)
    # 0 = use the task's own max_agent_timeout_sec / max_test_timeout_sec.
    parser.add_argument("--step_timeout", default=0, type=int)
    parser.add_argument("--test_timeout", default=0, type=int)


def line_to_request(line, args):
    task = json.loads(line)
    return TerminalTaskRequest(
        task_id=task["task_id"],
        task=task,
        output_dir=args.output_dir,
        task_cache_dir=args.task_cache_dir,
        model_name=args.model_name,
        step_timeout=args.step_timeout or int(task.get("max_agent_timeout_sec") or 900),
        test_timeout=args.test_timeout or int(task.get("max_test_timeout_sec") or 180),
    )


def is_success(result):
    return bool(result.success)
