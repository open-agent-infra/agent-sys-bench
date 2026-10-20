"""Auto-research scenario serving contract."""
from scenario_protocol.scenarios.auto_research import (
    AutoResearchTaskRequest, AutoResearchTaskResult, RPC_NAME, PARAM_NAME,
)
__all__ = ["AutoResearchTaskRequest", "AutoResearchTaskResult", "RPC_NAME", "PARAM_NAME"]
SCENARIO_NAME = "auto-research"
DESCRIPTION = "AutoML research — agent iteratively trains models on an MLE-bench (Kaggle) competition"
DATASET = "MLE-bench competitions (default: spooky-author-identification; --lite: 22 Lite competitions)"
EVALUATION = "MLE-bench grader on the held-out test split (score, medal, above-median)"
