"""Entrypoint for the orchestrator-side mini-swe-agent fast-service."""

import os

from agsbench_mini_swe_agent import mini_swe_fsm  # registers all @fast_service funcs
from agsbench_mini_swe_agent.config_loader import load_config_with_envvars


if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../config/agsbench_mini_swe_agent-server.yml",
        )
    )
    config = load_config_with_envvars(config_path)
    mini_swe_fsm.execute(config)
