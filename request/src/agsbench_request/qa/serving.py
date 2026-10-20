"""QA scenario serving contract."""
from scenario_protocol.scenarios.qa import QARequest, QAResult, RPC_NAME, PARAM_NAME
__all__ = ["QARequest", "QAResult", "RPC_NAME", "PARAM_NAME"]
SCENARIO_NAME = "qa"
DESCRIPTION = "Retrieval-augmented generation — ingest documents, retrieve, answer questions"
DATASET = "microsoft/wiki_qa, natural_questions, hotpotqa, ms_marco"
EVALUATION = "Answer correctness (LLM judge or exact match)"
