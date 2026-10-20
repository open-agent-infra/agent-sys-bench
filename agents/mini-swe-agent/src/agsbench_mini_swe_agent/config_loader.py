"""Env-var-aware loader for fast-service YAML configs.

Lives in the package (not under scripts/) so both ``mini_swe_service.py``
(orchestrator) and ``mini_swe_env_service.py`` (env host) can import it
without scripts/ being a Python package.
"""

import os
from typing import Any

import yaml

from fast_service import FastServiceConfig


def _expand(obj: Any) -> Any:
    """Recursively ``os.path.expandvars`` every string in a YAML tree."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def load_config_with_envvars(path: str) -> FastServiceConfig:
    """Load a fast-service YAML, expanding ``${VAR}`` references first.

    Used so the static YAML can carry placeholders like ``${MINI_SWE_ENV_HOST}``
    that get filled in from the compose file's environment. Fails loudly on
    unresolved placeholders so we never silently try to dial an unset host.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    expanded = _expand(raw)
    for svc in expanded.get("service_list", []) or []:
        ip = svc.get("ip")
        if isinstance(ip, str) and "${" in ip:
            raise RuntimeError(
                f"Unresolved env var in service_list[{svc.get('name')}].ip: {ip!r}"
            )
        port = svc.get("port")
        if isinstance(port, str):
            try:
                svc["port"] = int(port)
            except ValueError as e:
                raise RuntimeError(
                    f"service_list[{svc.get('name')}].port must be int, got {port!r}"
                ) from e
    return FastServiceConfig(**expanded)
