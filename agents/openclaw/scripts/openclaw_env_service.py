"""Entrypoint for the openclaw env-manager service (multi-node v2).

Runs on the remote host (<REMOTE_HOST>) and manages SSH sandbox containers.
"""

import logging
import os
import sys
from pathlib import Path


from agsbench_openclaw.config_loader import load_config_with_envvars
from agsbench_openclaw.run_task import OCLAW_RUN_FSM
from agsbench_openclaw.env_service import (
    sandbox_create,
    sandbox_destroy,
)  # noqa: F401 — registers RPCs

from agsbench_openclaw.vsandbox_service import (  # noqa: F401 — registers RPCs
    vsandbox_exec,
    vsandbox_shell,
    vsandbox_put_file,
    vsandbox_get_file,
)

logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
log = logging.getLogger("agsbench_openclaw.env_service")

if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config/openclaw-env.yml")
    )
    log.info("Loading env-manager config from %s", config_path)
    cfg = load_config_with_envvars(config_path)
    log.info("Starting env-manager on port %d...", cfg.port)
    OCLAW_RUN_FSM.execute(cfg)
