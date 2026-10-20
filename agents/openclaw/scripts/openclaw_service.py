"""Entrypoint for the openclaw fast-service RPC server (docker-compose mode).

Configures the openclaw CLI to point at the LLM proxy and search proxy
containers, then starts the fast-service server exposing the run_task RPC.
"""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path


from agsbench_openclaw.run_task import OCLAW_RUN_FSM, run_task  # noqa: F401 — registers the RPC

# In multi-v2 mode, import env_service to register sandbox_create/sandbox_destroy
# as client stubs on OCLAW_RUN_FSM (they share the same FSM).
sandbox_mode = os.environ.get("OPENCLAW_SANDBOX_MODE", "")
if sandbox_mode in ("ssh", "vsandbox"):
    from agsbench_openclaw.env_service import sandbox_create, sandbox_destroy  # noqa: F401
if sandbox_mode == "vsandbox":
    from agsbench_openclaw.vsandbox_service import (  # noqa: F401
        vsandbox_exec,
        vsandbox_shell,
        vsandbox_put_file,
        vsandbox_get_file,
    )


logging.basicConfig(
    level="INFO",
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
log = logging.getLogger("agsbench_openclaw.service")


def configure_openclaw_for_compose():
    """Patch openclaw config to point at proxy containers via compose DNS."""
    llm_proxy_url = os.environ.get(
        "OPENCLAW_LLM_PROXY_URL", "http://openclaw-llm-proxy:1801"
    )
    search_proxy_url = os.environ.get(
        "OPENCLAW_SEARCH_PROXY_URL", "http://openclaw-search-proxy:1802"
    )
    # Upstream model id, forwarded as-is by the LLM proxy (e.g. DashScope
    # serves deepseek-v4-pro but not deepseek-v4-flash).
    model_id = os.environ.get("OPENCLAW_MODEL_ID", "deepseek-v4-flash")

    patch = {
        "models": {
            "mode": "merge",
            "providers": {
                "deepseek-instrumented": {
                    "baseUrl": f"{llm_proxy_url}/v1",
                    "apiKey": "sk-dummy-not-used-by-proxy",
                    "api": "openai-completions",
                    "contextWindow": 1000000,
                    "maxTokens": 8192,
                    "models": [
                        {
                            "id": model_id,
                            "name": f"{model_id} (instrumented)",
                            "api": "openai-completions",
                            "baseUrl": f"{llm_proxy_url}/v1",
                            "contextWindow": 131000,
                            "maxTokens": 8192,
                            "input": ["text"],
                            "cost": {
                                "input": 0.14,
                                "output": 0.28,
                                "cacheRead": 0,
                                "cacheWrite": 0,
                            },
                        }
                    ],
                }
            },
        },
        "plugins": {
            "entries": {
                "firecrawl": {
                    "enabled": True,
                    "config": {
                        "webSearch": {
                            "baseUrl": search_proxy_url,
                            "apiKey": os.environ.get("FIRECRAWL_API_KEY", ""),
                        }
                    },
                }
            }
        },
        "tools": {"web": {"search": {"enabled": True, "provider": "firecrawl"}}},
    }

    # Also configure sandbox agent if OPENCLAW_SANDBOX=1
    sandbox = os.environ.get("OPENCLAW_SANDBOX", "0") == "1"
    agents_list = [{"id": "main"}]
    if sandbox:
        agents_list.append(
            {
                "id": "main-sandbox",
                "name": "WildClawBench Sandboxed Agent",
                "workspace": "/workspace/openclaw/sandbox_workspace",
                "sandbox": {
                    "mode": "all",
                    "scope": "session",
                    "backend": "docker",
                    "workspaceAccess": "rw",
                    "docker": {
                        "image": "openclaw-sandbox:bookworm-slim",
                        "network": "bridge",
                        "readOnlyRoot": False,
                    },
                },
            }
        )
    patch["agents"] = {"list": agents_list}

    proc = subprocess.run(
        ["openclaw", "config", "patch", "--stdin"],
        input=json.dumps(patch),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        log.error("openclaw config patch failed: %s", proc.stderr)
        raise RuntimeError(f"openclaw config patch failed: {proc.stderr}")
    log.info("openclaw config patched for compose networking.")


if __name__ == "__main__":
    configure_openclaw_for_compose()

    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config/openclaw-server.yml")
    )
    log.info("Loading fast-service config from %s", config_path)

    from agsbench_openclaw.config_loader import load_config_with_envvars

    cfg = load_config_with_envvars(config_path)
    log.info("Starting openclaw RPC server on port 1803...")
    OCLAW_RUN_FSM.execute(cfg)
