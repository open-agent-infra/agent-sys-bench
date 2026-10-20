"""Single-instance runner for mini-swe-agent.

Re-implements the inner loop of ``minisweagent.run.benchmarks.swebench.process_instance``
without the batch/Live-progress shell, so each call drives one SWE-bench instance
and we can hook fast-service telemetry around ``model.query`` and ``env.execute``.

Per-LLM-call and per-bash-step timing comes from two fast-service ``local`` (i.e.
not listed in service_list, no RPC) functions: :func:`model_query_call` and
:func:`env_execute_call`. They are registered once at import time. At runtime we
monkey-patch each instance's ``model.query`` and ``env.execute`` to delegate to
these timed wrappers, preserving fast-service's parent_call_id linkage.

NB: do NOT add ``from __future__ import annotations`` here. fast-service
inspects raw ``__annotations__`` to discover return classes for client-side
deserialization; PEP 563 lazy annotations would break that path.
"""

import json
import os
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Optional

from jinja2 import StrictUndefined, Template

from fast_service import RequestContext

from minisweagent.agents.default import DefaultAgent
from minisweagent.config import get_config_from_spec
from minisweagent.environments import get_environment
from minisweagent.models import get_model
from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

from agsbench_mini_swe_agent.fsm import mini_swe_fsm

# Importing env_service registers env_create/env_execute/env_destroy with
# mini_swe_fsm so fast-service can rewrite them (into local impls on the
# env host, into client stubs on the orchestrator). The import is cheap and
# safe on both sides because the function bodies only run when invoked.
from agsbench_mini_swe_agent import env_service as _env_service  # noqa: F401
from agsbench_mini_swe_agent.remote_env import RemoteEnv, create_remote_env


_PREDS_LOCK = Lock()


def _is_remote_env_mode() -> bool:
    """Whether the runner should delegate env ops to a remote env service.

    Toggle via ``MINI_SWE_REMOTE_ENV=1`` (set by the orchestrator's compose
    file in multi-node mode). When false, behavior is identical to the
    original single-host integration.
    """
    return os.getenv("MINI_SWE_REMOTE_ENV", "").strip().lower() in {"1", "true", "yes"}


# === Per-sandbox resource isolation =========================================
#
# When the service handles concurrent solve_instance calls, each one spawns a
# per-task swebench docker container. Without ``--cpuset-cpus`` they share the
# host's full CPU set and contend with each other. We model this with a small
# pool of cpuset strings: each ``run_one`` call acquires one before building
# its env and releases it on completion. The pool size is the upper bound on
# concurrent sandboxes.
#
# Configure via env vars:
#   MINI_SWE_CPUSETS  comma-separated cpuset specs, one per slot. Each spec
#                     is passed verbatim to ``docker run --cpuset-cpus``.
#                     e.g. "0-1,2-3,4-5,6-7" → 4 slots of 2 CPUs each.
#                     Empty/unset → no pinning, no concurrency cap.
#   MINI_SWE_MEMORY   memory limit passed to ``docker run --memory`` for each
#                     sandbox (e.g. "4g"). Empty/unset → no memory cap.
#
# Configuration is read once at first acquire, not at module import, so
# updates to env vars between service restarts are picked up.
_CPUSET_POOL: Optional[Queue] = None
_CPUSET_POOL_LOCK = Lock()


def _get_cpuset_pool() -> Optional[Queue]:
    """Lazy-initialize the cpuset pool from env vars."""
    global _CPUSET_POOL
    if _CPUSET_POOL is not None:
        return _CPUSET_POOL
    with _CPUSET_POOL_LOCK:
        if _CPUSET_POOL is not None:
            return _CPUSET_POOL
        spec = os.getenv("MINI_SWE_CPUSETS", "").strip()
        if not spec:
            return None
        pool: Queue = Queue()
        for cpuset in spec.split(","):
            cpuset = cpuset.strip()
            if cpuset:
                pool.put(cpuset)
        _CPUSET_POOL = pool
        return _CPUSET_POOL


def _acquire_sandbox_resources() -> tuple[Optional[str], Optional[str]]:
    """Reserve a cpuset slot (blocking) and read the memory limit. Returns
    (cpuset, memory_limit). Either may be None if not configured."""
    pool = _get_cpuset_pool()
    cpuset = pool.get() if pool is not None else None
    memory = os.getenv("MINI_SWE_MEMORY", "").strip() or None
    return cpuset, memory


def _release_sandbox_resources(cpuset: Optional[str]) -> None:
    if cpuset is None:
        return
    pool = _get_cpuset_pool()
    if pool is not None:
        pool.put(cpuset)


def _stop_sandbox_sync(env: Any) -> None:
    """Synchronously stop and remove the sandbox container.

    mini-swe-agent's ``DockerEnvironment.cleanup()`` fires a background
    ``docker stop`` (Popen with ``&``) and returns immediately. That is fine
    for ergonomic single-task use, but in our pool we MUST not release the
    cpuset slot until the container is actually gone — otherwise a new task
    can grab the same cpuset while the previous container is still draining.
    """
    container_id = getattr(env, "container_id", None)
    if not container_id:
        return
    executable = (
        getattr(env.config, "executable", "docker")
        if hasattr(env, "config")
        else "docker"
    )
    # Prevent mini-swe-agent's __del__-driven cleanup from racing us.
    try:
        env.container_id = None
    except Exception:
        pass
    try:
        subprocess.run(
            [executable, "rm", "-f", container_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )
    except Exception:
        # Best-effort: even if the rm fails, releasing the cpuset is OK
        # because the container is on its way out anyway.
        pass


@dataclass
class RunResult:
    success: bool
    instance_id: str
    model_name: str
    patch: str
    exit_status: Optional[str]
    error: Optional[str]


@mini_swe_fsm.fast_service(monitor_only=True)
def model_query_call(
    model: Any,
    messages: list,
    fsm_context: RequestContext = None,
    **kwargs,
) -> dict:
    """Timed wrapper around ``Model.query``. Registered as a monitor-only
    fast-service function so each call adds a row to the monitor CSV with
    the proper parent_call_id under the surrounding ``solve_instance``
    call.

    ``monitor_only=True`` is load-bearing: the intermediate-storage
    serializer rejects non-Pydantic returns (and our model object isn't),
    and it would refuse `dict` returns when ``store_intermediate`` is on.
    Monitor-only skips the intermediate layer while still recording
    per-call timing in the monitor CSV.
    """
    return model.query(messages, **kwargs)


@mini_swe_fsm.fast_service(monitor_only=True)
def env_execute_call(
    env: Any,
    action: Any,
    cwd: str = "",
    fsm_context: RequestContext = None,
) -> dict:
    """Timed wrapper around ``Environment.execute``. monitor_only=True for
    the same reason as ``model_query_call`` above."""
    return env.execute(action, cwd=cwd)


def _instrument_model(model, fsm_context: Optional[RequestContext]):
    original_query = model.query

    def wrapped_query(messages, **kwargs):
        return model_query_call(
            model_for_call,  # captured below
            messages,
            fsm_context=fsm_context,
            **kwargs,
        )

    # Capture the *original* model object via closure so the wrapper does not
    # recurse on itself.
    model_for_call = _OriginalModelProxy(model, original_query)
    model.query = wrapped_query
    return model


def _instrument_env(env, fsm_context: Optional[RequestContext]):
    original_execute = env.execute

    def wrapped_execute(action, cwd=""):
        return env_execute_call(
            env_for_call,
            action,
            cwd=cwd,
            fsm_context=fsm_context,
        )

    env_for_call = _OriginalEnvProxy(env, original_execute)
    env.execute = wrapped_execute
    return env


class _OriginalModelProxy:
    """Holds a reference to the original ``model.query`` so the timed wrapper
    invokes the un-instrumented call (avoiding recursion)."""

    __slots__ = ("_model", "_query")

    def __init__(self, model, query):
        self._model = model
        self._query = query

    @property
    def config(self):
        return self._model.config

    def query(self, messages, **kwargs):
        return self._query(messages, **kwargs)


class _OriginalEnvProxy:
    __slots__ = ("_env", "_execute")

    def __init__(self, env, execute):
        self._env = env
        self._execute = execute

    def execute(self, action, cwd=""):
        return self._execute(action, cwd=cwd)


def _build_config(config_path: str, model_name: Optional[str]) -> dict:
    config = get_config_from_spec(config_path)
    if model_name:
        config.setdefault("model", {})["model_name"] = model_name
    return config


def _resolved_env_config(config: dict, instance: dict) -> dict:
    """Build the env config dict the same way both local and remote modes need.

    Image substitution, run_args defaults, and so on are shared logic; the
    only difference between modes is who actually `docker run`s it.
    """
    env_config = dict(config.setdefault("environment", {}))
    env_config["environment_class"] = env_config.get("environment_class", "docker")
    image_name = get_swebench_docker_image_name(instance)
    if env_config["environment_class"] in ("docker", "swerex_modal"):
        env_config["image"] = image_name
    elif env_config["environment_class"] in ("singularity", "contree"):
        env_config["image"] = "docker://" + image_name
    return env_config


def _render_startup_command(config: dict, instance: dict) -> Optional[str]:
    startup_command = config.get("run", {}).get("env_startup_command")
    if not startup_command:
        return None
    return Template(startup_command, undefined=StrictUndefined).render(**instance)


def _build_sb_environment(
    config: dict,
    instance: dict,
    cpuset: Optional[str] = None,
    memory: Optional[str] = None,
):
    """Local-mode env builder. In remote mode the runner takes a different
    path that goes through :func:`create_remote_env` instead; see
    :func:`run_one`."""
    env_config = _resolved_env_config(config, instance)

    # Inject per-sandbox resource limits into the docker run args. We extend
    # the existing run_args (default ["--rm"]) rather than overwriting so any
    # user-provided flags in the YAML config survive.
    if env_config["environment_class"] == "docker" and (cpuset or memory):
        extra: list[str] = list(env_config.get("run_args", ["--rm"]))
        if cpuset:
            extra.append(f"--cpuset-cpus={cpuset}")
        if memory:
            extra.append(f"--memory={memory}")
        env_config["run_args"] = extra

    env = get_environment(env_config)
    rendered = _render_startup_command(config, instance)
    if rendered:
        out = env.execute(rendered)
        if out["returncode"] != 0:
            raise RuntimeError(f"Error executing startup command: {out}")
    return env


def _update_preds(output_dir: Path, instance_id: str, model_name: str, patch: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "preds.json"
    with _PREDS_LOCK:
        data = {}
        if preds_path.exists():
            try:
                data = json.loads(preds_path.read_text())
            except json.JSONDecodeError:
                data = {}
        data[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": patch,
        }
        preds_path.write_text(json.dumps(data, indent=2))


def run_one(
    instance: dict,
    config_path: str,
    output_dir: str,
    fsm_context: Optional[RequestContext],
    model_name_override: Optional[str] = None,
) -> RunResult:
    instance_id = instance["instance_id"]
    out_root = Path(output_dir)
    instance_dir = out_root / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    config = _build_config(config_path, model_name_override)
    model = get_model(config=config.get("model", {}))
    model_name = getattr(model.config, "model_name", "unknown")
    _instrument_model(model, fsm_context)

    agent = None
    exit_status: Optional[str] = None
    patch = ""
    error: Optional[str] = None
    extra_info: dict = {}

    # In remote-env mode the cpuset pool lives on the env host; the
    # orchestrator no longer owns docker. In local mode it's still here.
    remote_mode = _is_remote_env_mode()
    if remote_mode:
        cpuset, memory = None, None
    else:
        # Acquire a sandbox slot (blocks if all are in use → this is what caps
        # concurrent sandboxes when the loadgen dispatches more solve_instance
        # calls than there are cpuset slots).
        cpuset, memory = _acquire_sandbox_resources()
    env = None
    try:
        if remote_mode:
            env_config = _resolved_env_config(config, instance)
            startup = _render_startup_command(config, instance)
            env = create_remote_env(env_config, instance, startup, fsm_context)
            # env_execute is itself a fast-service RPC, so it already emits a
            # monitor row per call under the surrounding solve_instance
            # call_id — wrapping again would double-count.
        else:
            env = _build_sb_environment(config, instance, cpuset=cpuset, memory=memory)
            _instrument_env(env, fsm_context)
        agent = DefaultAgent(model, env, **config.get("agent", {}))
        info = agent.run(instance["problem_statement"])
        exit_status = info.get("exit_status")
        patch = info.get("submission") or ""
    except Exception as e:
        exit_status = type(e).__name__
        error = str(e)
        extra_info = {
            "traceback": traceback.format_exc(),
            "exception_str": str(e),
        }
    finally:
        # IMPORTANT: stop the container synchronously BEFORE returning the
        # cpuset to the pool. Otherwise a new task can claim the same cpuset
        # while the draining container still holds it, breaking the
        # "sandboxes do not share resources" invariant.
        if env is not None:
            if isinstance(env, RemoteEnv):
                env.cleanup()  # env_destroy RPC — synchronous on the env host
            else:
                _stop_sandbox_sync(env)
        if not remote_mode:
            _release_sandbox_resources(cpuset)
        if agent is not None:
            traj_path = instance_dir / f"{instance_id}.traj.json"
            agent.save(
                traj_path,
                {
                    "info": {
                        "exit_status": exit_status,
                        "submission": patch,
                        **extra_info,
                    },
                    "instance_id": instance_id,
                },
            )
        _update_preds(out_root, instance_id, model_name, patch)

    return RunResult(
        success=bool(patch) and error is None,
        instance_id=instance_id,
        model_name=model_name,
        patch=patch,
        exit_status=exit_status,
        error=error,
    )
