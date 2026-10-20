"""Computer-use scenario serving contract."""
from scenario_protocol.scenarios.computer_use import (
    ComputerTaskRequest, ComputerTaskResult, RPC_NAME, PARAM_NAME,
)
__all__ = ["ComputerTaskRequest", "ComputerTaskResult", "RPC_NAME", "PARAM_NAME"]
SCENARIO_NAME = "computer-use"
DESCRIPTION = "Desktop GUI automation via OSWorld/QEMU VMs"
DATASET = "OSWorld evaluation_examples (github.com/xlang-ai/OSWorld)"
EVALUATION = "Task-specific evaluator (env reward)"
