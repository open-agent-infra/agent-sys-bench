"""Entrypoint script for the browsergym fast-service container."""

import os

from fast_service import FastServiceConfig

from agsbench_webagent import agsbench_webagent_fsm  # registers solve_task + helpers


if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../config/webagent-server.yml",
        )
    )
    config = FastServiceConfig.load_from_file(config_path)
    agsbench_webagent_fsm.execute(config)
