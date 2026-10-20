"""RemoteDesktopEnv — drop-in replacement for DesktopEnv that proxies all
VM operations to a remote env-manager via HTTP.

Used when OS_WORLD_PROVIDER=remote. The env-manager runs on an x86 host
and handles QEMU VM lifecycle; this class just serializes/deserializes
the calls over JSON.
"""

import base64
import os
import uuid
from typing import Any, Dict, Optional, Tuple

import requests


class RemoteDesktopEnv:
    """Proxies DesktopEnv calls to a remote env-manager HTTP service."""

    def __init__(
        self,
        env_manager_url: str,
        provider_name: str = "docker",
        os_type: str = "Ubuntu",
        action_space: str = "pyautogui",
        require_a11y_tree: bool = False,
        headless: bool = True,
        screen_size: Tuple[int, int] = (1920, 1080),
        enable_proxy: bool = False,
        client_password: str = "",
        **kwargs,
    ):
        self._url = env_manager_url.rstrip("/")
        self._env_id: Optional[str] = None
        self._vm_ip = "remote"
        self._timeout = int(os.environ.get("OS_WORLD_REMOTE_TIMEOUT", "600"))

        resp = self._post(
            "/os_world.env_request.env_create",
            {
                "provider_name": provider_name,
                "os_type": os_type,
                "action_space": action_space,
                "require_a11y_tree": require_a11y_tree,
                "headless": headless,
                "screen_size": list(screen_size),
                "enable_proxy": enable_proxy,
                "client_password": client_password,
            },
        )
        self._env_id = resp["env_id"]
        self._vm_ip = resp.get("vm_ip", "remote")

    @property
    def vm_ip(self) -> str:
        return self._vm_ip

    def reset(self, task_config: Optional[Dict] = None, seed=None, options=None):
        resp = self._post(
            "/os_world.env_request.env_reset",
            {
                "env_id": self._env_id,
                "task_config": task_config or {},
            },
        )
        return self._decode_obs(resp.get("obs", {}))

    def step(self, action: str, pause: float = 2.0):
        resp = self._post(
            "/os_world.env_request.env_step",
            {
                "env_id": self._env_id,
                "action": action,
                "pause": pause,
            },
        )
        obs = self._decode_obs(resp.get("obs", {}))
        return (
            obs,
            resp.get("reward", 0.0),
            resp.get("done", False),
            resp.get("info", {}),
        )

    def _get_obs(self):
        resp = self._post(
            "/os_world.env_request.env_get_obs",
            {
                "env_id": self._env_id,
            },
        )
        return self._decode_obs(resp.get("obs", {}))

    def evaluate(self):
        resp = self._post(
            "/os_world.env_request.env_evaluate",
            {
                "env_id": self._env_id,
            },
        )
        return resp.get("reward", 0.0)

    def close(self):
        if self._env_id:
            try:
                self._post(
                    "/os_world.env_request.env_close",
                    {
                        "env_id": self._env_id,
                    },
                )
            except Exception:
                pass
            self._env_id = None

    @property
    def last_obs(self):
        return getattr(self, "_last_obs", None)

    @last_obs.setter
    def last_obs(self, value):
        self._last_obs = value

    def _post(self, endpoint: str, data: dict) -> dict:
        url = f"{self._url}/{endpoint}"
        resp = requests.post(url, json={"req": data}, timeout=self._timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _decode_obs(obs: dict) -> dict:
        if "screenshot" in obs and isinstance(obs["screenshot"], str):
            obs["screenshot"] = base64.b64decode(obs["screenshot"])
        return obs
