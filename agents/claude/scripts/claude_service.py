"""Entrypoint for the claude fast-service container."""

import os

from fast_service import FastServiceConfig

from agsbench_claude.run_task import CLAUDE_RUN_FSM

if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../config/claude-server.yml",
        )
    )
    config = FastServiceConfig.load_from_file(config_path)
    CLAUDE_RUN_FSM.execute(config)
