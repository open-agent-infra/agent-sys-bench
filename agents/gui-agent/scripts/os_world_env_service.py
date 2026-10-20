"""Entrypoint for the os-world env-manager fast-service (hfserver)."""

import os
from fast_service import FastServiceConfig
from agsbench_gui_agent import agsbench_gui_agent_fsm

import os_world.env_request  # noqa: F401  register env_* endpoints

if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config/os_world-env.yml")
    )
    config = FastServiceConfig.load_from_file(config_path)
    agsbench_gui_agent_fsm.execute(config)
