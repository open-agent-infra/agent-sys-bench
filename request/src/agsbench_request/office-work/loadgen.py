"""Office-work scenario loadgen — scenario-specific logic only."""
from scenario_protocol.scenarios.office_work import OfficeTaskRequest, OfficeTaskResult, RPC_NAME, PARAM_NAME
from agsbench_request._common import resolve_manifest_path

Request = OfficeTaskRequest
Result = OfficeTaskResult

def extra_args(parser):
    parser.add_argument("--model", default="deepseek-instrumented/deepseek-v4-flash")
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--sandbox", default="0")
    parser.add_argument("--sandbox_mode", default="")

def line_to_request(line, args):
    return OfficeTaskRequest(
        task_md_path=resolve_manifest_path(line, args), model=args.model,
        output_root=args.output_root, sandbox=(args.sandbox == "1"),
        sandbox_mode=args.sandbox_mode,
    )

def is_success(result):
    return bool(result.exit_code == 0) if result.exit_code is not None else False
