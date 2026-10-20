"""Fast-service endpoints for the OSWorld env-manager.

Runs on the x86 host (hfserver). Manages DesktopEnv instances and exposes
VM lifecycle operations over HTTP so the agent on the ARM64 node can
interact with the QEMU VMs remotely.
"""

import base64
import os
import threading
import traceback
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from fast_service import RequestContext

from agsbench_gui_agent.fsm import agsbench_gui_agent_fsm

_envs: Dict[str, Any] = {}
_envs_lock = threading.Lock()


class EnvCreateRequest(BaseModel):
    provider_name: str = "docker"
    os_type: str = "Ubuntu"
    action_space: str = "pyautogui"
    require_a11y_tree: bool = False
    headless: bool = True
    screen_size: List[int] = [1920, 1080]
    enable_proxy: bool = False
    client_password: str = ""


class EnvCreateResponse(BaseModel):
    env_id: str
    vm_ip: str = ""


class EnvResetRequest(BaseModel):
    env_id: str
    task_config: Dict[str, Any] = {}


class EnvObsResponse(BaseModel):
    obs: Dict[str, Any] = {}


class EnvStepRequest(BaseModel):
    env_id: str
    action: str
    pause: float = 2.0


class EnvStepResponse(BaseModel):
    obs: Dict[str, Any] = {}
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = {}


class EnvIdRequest(BaseModel):
    env_id: str


class EnvEvalResponse(BaseModel):
    reward: float = 0.0


class EnvCloseResponse(BaseModel):
    closed: bool = True


def _encode_obs(obs: dict) -> dict:
    if "screenshot" in obs and isinstance(obs["screenshot"], bytes):
        obs["screenshot"] = base64.b64encode(obs["screenshot"]).decode("ascii")
    return obs


@agsbench_gui_agent_fsm.fast_service
def env_create(
    req: EnvCreateRequest,
    fsm_context: RequestContext = None,
) -> EnvCreateResponse:
    from desktop_env.desktop_env import DesktopEnv

    env_id = str(uuid.uuid4())[:8]
    env = DesktopEnv(
        provider_name=req.provider_name,
        os_type=req.os_type,
        action_space=req.action_space,
        require_a11y_tree=req.require_a11y_tree,
        headless=req.headless,
        screen_size=tuple(req.screen_size),
        enable_proxy=req.enable_proxy,
        client_password=req.client_password,
    )
    with _envs_lock:
        _envs[env_id] = env
    return EnvCreateResponse(env_id=env_id, vm_ip=getattr(env, "vm_ip", ""))


@agsbench_gui_agent_fsm.fast_service
def env_reset(
    req: EnvResetRequest,
    fsm_context: RequestContext = None,
) -> EnvObsResponse:
    env = _envs[req.env_id]
    env.reset(task_config=req.task_config if req.task_config else None)
    obs = env._get_obs()
    return EnvObsResponse(obs=_encode_obs(obs))


@agsbench_gui_agent_fsm.fast_service
def env_step(
    req: EnvStepRequest,
    fsm_context: RequestContext = None,
) -> EnvStepResponse:
    env = _envs[req.env_id]
    obs, reward, done, info = env.step(req.action, req.pause)
    return EnvStepResponse(
        obs=_encode_obs(obs),
        reward=float(reward),
        done=bool(done),
        info=info or {},
    )


@agsbench_gui_agent_fsm.fast_service
def env_get_obs(
    req: EnvIdRequest,
    fsm_context: RequestContext = None,
) -> EnvObsResponse:
    env = _envs[req.env_id]
    obs = env._get_obs()
    return EnvObsResponse(obs=_encode_obs(obs))


@agsbench_gui_agent_fsm.fast_service
def env_evaluate(
    req: EnvIdRequest,
    fsm_context: RequestContext = None,
) -> EnvEvalResponse:
    env = _envs[req.env_id]
    try:
        reward = float(env.evaluate())
    except Exception:
        reward = 0.0
    return EnvEvalResponse(reward=reward)


@agsbench_gui_agent_fsm.fast_service
def env_close(
    req: EnvIdRequest,
    fsm_context: RequestContext = None,
) -> EnvCloseResponse:
    with _envs_lock:
        env = _envs.pop(req.env_id, None)
    if env:
        try:
            env.close()
        except Exception:
            pass
    return EnvCloseResponse(closed=True)
