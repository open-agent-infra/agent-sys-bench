"""Server-side env/sandbox fast-service.

In single-host mode the orchestrator builds a :class:`DockerEnvironment` in
process and drives it directly. In multi-node mode (this module's purpose),
the env lives on a different host that owns the docker daemon. The
orchestrator calls three pydantic-typed RPCs declared here:

* :func:`env_create`   — start a docker sandbox container for one instance.
                         Returns an ``env_id`` (the container id) plus the
                         template_vars the orchestrator needs for prompt
                         rendering.
* :func:`env_execute`  — run one bash command inside the sandbox.
                         ``agsbench_mini_swe_agent``'s ``DockerEnvironment.execute``
                         raises ``Submitted`` when the agent emits the
                         completion sentinel; we catch it server-side and
                         re-raise on the client. The orchestrator side does
                         not import or depend on ``minisweagent.exceptions``.
* :func:`env_destroy`  — ``docker rm -f`` the container and release the
                         cpuset slot.

Per-sandbox resource isolation (cpuset pool + memory cap) lives here, not
in :mod:`runner`, because this is where the docker daemon and the host CPUs
actually are. ``runner`` still gates concurrency on the orchestrator side via
its own queue, but the hard cap is whatever ``MINI_SWE_CPUSETS`` evaluates to
on this process.

NB: do NOT add ``from __future__ import annotations`` here. fast-service's
fastapi engine reads ``f.__annotations__["return"]`` to look up the response
class for client-side deserialization; PEP 563 lazy annotations would make
it a string and break the RPC.
"""

import os
import subprocess
import uuid
from queue import Queue, Empty
from threading import Lock
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from fast_service import RequestContext

from minisweagent.environments.docker import DockerEnvironment
from minisweagent.exceptions import Submitted

from agsbench_mini_swe_agent.fsm import mini_swe_fsm


# === Pydantic wire types ====================================================


class EnvCreateRequest(BaseModel):
    image: str = Field(..., description="Docker image (already includes registry/tag).")
    cwd: str = Field(default="/", description="Working directory inside the container.")
    env: Dict[str, str] = Field(default_factory=dict)
    forward_env: List[str] = Field(default_factory=list)
    timeout: int = Field(default=30, description="Per-command timeout (seconds).")
    run_args: List[str] = Field(
        default_factory=lambda: ["--rm"],
        description="Extra args forwarded to `docker run`. cpuset/memory limits "
        "are appended by the server from the local pool; do NOT pre-bake them.",
    )
    container_timeout: str = Field(default="2h")
    pull_timeout: int = Field(default=120)
    interpreter: List[str] = Field(default_factory=lambda: ["bash", "-lc"])
    startup_command: Optional[str] = Field(
        default=None,
        description="Already-rendered command to run once after the container "
        "starts (env_startup_command from the agent config). Failure here "
        "destroys the sandbox and surfaces an error to the orchestrator.",
    )


class EnvCreateResponse(BaseModel):
    env_id: str = Field(
        ..., description="Server-side handle (also the docker container id)."
    )
    template_vars: Dict[str, Any] = Field(
        default_factory=dict,
        description="Result of DockerEnvironment.get_template_vars() — the "
        "orchestrator needs this to render prompts containing platform info.",
    )
    cpuset: Optional[str] = None
    memory: Optional[str] = None
    error: Optional[str] = None


class EnvExecuteRequest(BaseModel):
    env_id: str
    action: Dict[str, Any] = Field(
        ...,
        description="mini-swe-agent passes {'command': '<bash>'}; we forward it verbatim.",
    )
    cwd: str = ""
    timeout: Optional[int] = None


class EnvExecuteResponse(BaseModel):
    output: str = ""
    returncode: int = 0
    exception_info: str = ""
    extra: Dict[str, Any] = Field(default_factory=dict)
    # Submitted-signal carry-over. ``DockerEnvironment.execute`` raises
    # ``Submitted`` on the completion sentinel; the orchestrator's RemoteEnv
    # re-raises this exception type so the agent loop terminates the same
    # way it does in single-host mode.
    submitted: bool = False
    submission: str = ""
    error: Optional[str] = None


class EnvDestroyRequest(BaseModel):
    env_id: str


class EnvDestroyResponse(BaseModel):
    ok: bool = True
    error: Optional[str] = None


# === Server-side state ======================================================


_ENV_REGISTRY: Dict[str, DockerEnvironment] = {}
_ENV_CPUSETS: Dict[str, Optional[str]] = {}
_REGISTRY_LOCK = Lock()

_CPUSET_POOL: Optional[Queue] = None
_CPUSET_POOL_LOCK = Lock()


def _get_cpuset_pool() -> Optional[Queue]:
    """Lazy-build the cpuset pool from ``MINI_SWE_CPUSETS``.

    Identical semantics to the pool that used to live in runner.py — moved
    here because this is the process that actually shells out to docker.
    Returning None means "no pinning, no concurrency cap from this side";
    the orchestrator's loadgen still bounds concurrency via FSBENCH_NUM_THREADS.
    """
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


def _stop_sandbox_sync(env: DockerEnvironment) -> None:
    """Synchronously `docker rm -f` the sandbox.

    DockerEnvironment.cleanup() fires a background Popen and returns; we MUST
    not release the cpuset slot until the container is really gone, otherwise
    a new task can claim the same cpuset while the draining container still
    holds it.
    """
    container_id = getattr(env, "container_id", None)
    if not container_id:
        return
    executable = getattr(env.config, "executable", "docker")
    try:
        env.container_id = None  # disarm __del__ cleanup
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
        pass


# === RPC entrypoints ========================================================


@mini_swe_fsm.fast_service
def env_create(
    req: EnvCreateRequest,
    fsm_context: RequestContext = None,
) -> EnvCreateResponse:
    """Start a sandbox container and register it under a new env_id.

    Acquires a cpuset slot (blocking) and a memory limit from the local
    pool, appends `--cpuset-cpus`/`--memory` to ``run_args``, then constructs
    a :class:`DockerEnvironment` (which `docker run`s synchronously). If a
    startup command was requested, runs it; a non-zero exit destroys the
    sandbox and surfaces the error.
    """
    cpuset, memory = _acquire_sandbox_resources()
    run_args = list(req.run_args)
    if cpuset:
        run_args.append(f"--cpuset-cpus={cpuset}")
    if memory:
        run_args.append(f"--memory={memory}")

    env: Optional[DockerEnvironment] = None
    try:
        env = DockerEnvironment(
            image=req.image,
            cwd=req.cwd,
            env=req.env,
            forward_env=req.forward_env,
            timeout=req.timeout,
            run_args=run_args,
            container_timeout=req.container_timeout,
            pull_timeout=req.pull_timeout,
            interpreter=req.interpreter,
        )
    except Exception as e:
        _release_sandbox_resources(cpuset)
        return EnvCreateResponse(
            env_id="",
            cpuset=cpuset,
            memory=memory,
            error=f"env_create failed: {type(e).__name__}: {e}",
        )

    env_id = env.container_id or f"mini-swe-{uuid.uuid4().hex[:8]}"

    if req.startup_command:
        try:
            out = env.execute(req.startup_command)
            if out.get("returncode", -1) != 0:
                _stop_sandbox_sync(env)
                _release_sandbox_resources(cpuset)
                return EnvCreateResponse(
                    env_id="",
                    cpuset=cpuset,
                    memory=memory,
                    error=f"startup_command failed (rc={out.get('returncode')}): "
                    f"{out.get('output', '')[:2000]}",
                )
        except Submitted:
            # The startup command should never trigger the completion
            # sentinel, but if it somehow does, treat it as a setup error.
            _stop_sandbox_sync(env)
            _release_sandbox_resources(cpuset)
            return EnvCreateResponse(
                env_id="",
                cpuset=cpuset,
                memory=memory,
                error="startup_command emitted COMPLETE_TASK sentinel",
            )

    with _REGISTRY_LOCK:
        _ENV_REGISTRY[env_id] = env
        _ENV_CPUSETS[env_id] = cpuset

    return EnvCreateResponse(
        env_id=env_id,
        template_vars=env.get_template_vars(),
        cpuset=cpuset,
        memory=memory,
    )


@mini_swe_fsm.fast_service
def env_execute(
    req: EnvExecuteRequest,
    fsm_context: RequestContext = None,
) -> EnvExecuteResponse:
    """Run one bash command inside the sandbox.

    ``DockerEnvironment.execute`` raises ``Submitted`` on the COMPLETE_TASK
    sentinel; we catch it and reflect it in the response so the orchestrator
    can re-raise client-side. Any other exception is serialized as
    ``error`` (treated as a fatal RPC failure by the client).
    """
    with _REGISTRY_LOCK:
        env = _ENV_REGISTRY.get(req.env_id)
    if env is None:
        return EnvExecuteResponse(
            output="",
            returncode=-1,
            error=f"unknown env_id {req.env_id!r}",
        )

    try:
        if req.timeout is not None:
            out = env.execute(req.action, cwd=req.cwd, timeout=req.timeout)
        else:
            out = env.execute(req.action, cwd=req.cwd)
    except Submitted as s:
        # ``Submitted`` extends ``InterruptAgentFlow``, whose __init__ stores
        # the completion dict in ``self.messages`` (a tuple) and calls
        # ``super().__init__()`` with no args — so ``s.args`` is empty. Read
        # ``messages`` first; fall back to ``args`` for older versions that
        # passed the payload positionally.
        msgs = getattr(s, "messages", None) or s.args
        payload = msgs[0] if msgs else {}
        extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
        submission = (extra.get("submission") if isinstance(extra, dict) else None) or (
            payload.get("content", "") if isinstance(payload, dict) else ""
        )
        return EnvExecuteResponse(
            output=payload.get("content", "") if isinstance(payload, dict) else "",
            returncode=0,
            submitted=True,
            submission=submission,
        )
    except Exception as e:
        return EnvExecuteResponse(
            output="",
            returncode=-1,
            exception_info=f"{type(e).__name__}: {e}",
            error=str(e),
        )

    return EnvExecuteResponse(
        output=out.get("output", ""),
        returncode=int(out.get("returncode", 0)),
        exception_info=out.get("exception_info", "") or "",
        extra=out.get("extra", {}) or {},
    )


@mini_swe_fsm.fast_service
def env_destroy(
    req: EnvDestroyRequest,
    fsm_context: RequestContext = None,
) -> EnvDestroyResponse:
    """Stop the sandbox synchronously and release its cpuset slot.

    Idempotent: destroying a missing env_id is a no-op (the orchestrator
    sometimes retries cleanup).
    """
    with _REGISTRY_LOCK:
        env = _ENV_REGISTRY.pop(req.env_id, None)
        cpuset = _ENV_CPUSETS.pop(req.env_id, None)
    if env is None:
        return EnvDestroyResponse(ok=True)
    try:
        _stop_sandbox_sync(env)
    except Exception as e:
        return EnvDestroyResponse(ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        _release_sandbox_resources(cpuset)
    return EnvDestroyResponse(ok=True)
