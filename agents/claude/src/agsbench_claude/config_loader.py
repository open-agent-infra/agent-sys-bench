"""Env-var-aware loader for fast-service YAML configs."""

import os
from typing import Any

import yaml

from fast_service import FastServiceConfig


import re

_SHELL_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_shell_vars(s: str) -> str:
    """Expand ${VAR} and ${VAR:-default} patterns."""

    def _repl(m):
        expr = m.group(1)
        if ":-" in expr:
            name, default = expr.split(":-", 1)
            return os.environ.get(name, default)
        return os.environ.get(expr, m.group(0))

    return _SHELL_VAR_RE.sub(_repl, s)


def _expand(obj: Any) -> Any:
    if isinstance(obj, str):
        return _expand_shell_vars(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def load_config_with_envvars(path: str) -> FastServiceConfig:
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
