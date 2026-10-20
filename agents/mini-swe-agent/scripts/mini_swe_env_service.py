"""Entrypoint for the env-side mini-swe-agent fast-service (<REMOTE_HOST>).

Hosts env_create / env_execute / env_destroy. The same package is imported
as on the orchestrator side, but loads a different YAML where env_* live
locally; solve_instance is registered but absent from service_list (so
fast-service treats it as a local-only function — no HTTP endpoint, no
client stub).
"""

import os

from agsbench_mini_swe_agent import mini_swe_fsm  # registers all @fast_service funcs
from agsbench_mini_swe_agent.config_loader import load_config_with_envvars


if __name__ == "__main__":
    config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../config/agsbench_mini_swe_agent-env.yml",
        )
    )
    config = load_config_with_envvars(config_path)
    mini_swe_fsm.execute(config)
