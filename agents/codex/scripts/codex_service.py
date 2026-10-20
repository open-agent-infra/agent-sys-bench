"""Entrypoint for the codex fast-service container."""

import os

from fast_service import FastServiceConfig

from agsbench_codex import codex_fsm


if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../config/codex-server.yml",
        )
    )
    config = FastServiceConfig.load_from_file(config_path)
    codex_fsm.execute(config)
