"""Entrypoint script for the pi-autoresearch fast-service container."""

import os

from fast_service import FastServiceConfig

from agsbench_pi_autoresearch import agsbench_pi_autoresearch_fsm  # registers solve_task + helpers


if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../config/pi-autoresearch-server.yml",
        )
    )
    config = FastServiceConfig.load_from_file(config_path)
    agsbench_pi_autoresearch_fsm.execute(config)
