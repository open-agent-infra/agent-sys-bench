"""QA scenario loadgen — scenario-specific logic only."""

import os

from scenario_protocol.scenarios.qa import QARequest, QAResult, RPC_NAME, PARAM_NAME

Request = QARequest
Result = QAResult


def extra_args(parser):
    parser.add_argument("--doc_root", default="",
                        help="Directory (as seen by the agent service) that manifest "
                             "doc paths are relative to.")


def line_to_request(line, args):
    # question_id, question, document_id, document_title, doc path
    parts = line.rstrip("\n").split("\t")
    query = parts[1] if len(parts) > 1 else parts[0]
    uri = parts[4] if len(parts) > 4 and parts[4] else parts[0]
    if args.doc_root and not os.path.isabs(uri):
        uri = os.path.join(args.doc_root, uri)
    return QARequest(query=query, uri=uri, type="local_html")


def is_success(result):
    return bool(result.answer)
