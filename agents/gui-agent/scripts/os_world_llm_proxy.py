"""Entrypoint for the vLLM proxy fast-service."""

import os
from fast_service import FastServiceConfig
from agsbench_gui_agent import agsbench_gui_agent_fsm

import os_world.llm_request  # noqa: F401  register llm_call endpoint

if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config/os_world-llm-proxy.yml")
    )
    config = FastServiceConfig.load_from_file(config_path)
    agsbench_gui_agent_fsm.execute(config)
