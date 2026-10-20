"""Tool-rich-assistant scenario loadgen — scenario-specific logic only."""
from scenario_protocol.scenarios.tool_rich_assistant import ToolRichTaskRequest, ToolRichTaskResult, RPC_NAME, PARAM_NAME
from agsbench_request._common import resolve_manifest_path

Request = ToolRichTaskRequest
Result = ToolRichTaskResult

def extra_args(parser):
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--output_root", default="./output")
    parser.add_argument("--effort", default="")

def line_to_request(line, args):
    return ToolRichTaskRequest(
        task_md_path=resolve_manifest_path(line, args), model=args.model,
        output_root=args.output_root, effort=args.effort,
    )

def is_success(result):
    return bool(result.exit_code == 0) if result.exit_code is not None else False
