"""Client-side proxy that satisfies mini-swe-agent's Environment interface
by forwarding ``execute``/``get_template_vars``/``serialize``/``cleanup``
calls to the :mod:`env_service` RPCs running on the env host.

This is the missing piece for multi-node mode: ``DefaultAgent`` interacts
with ``env`` purely through these four hooks (see
``minisweagent/agents/default.py``), so swapping in a RemoteEnv lets the
agent run unchanged while the actual docker sandbox lives elsewhere.

Why not just use ``_instrument_env`` from runner.py around a local
DockerEnvironment in multi-node mode? Because there is no local
DockerEnvironment to instrument — the orchestrator container does not have
a docker daemon (that's the whole point of the split). The env_* RPCs are
themselves fast-service functions, so per-execute monitor rows and
parent_call_id linkage are automatic; we do not wrap them again.
"""

from __future__ import (
    annotations,
)  # safe — this file declares no @fast_service entrypoints

from typing import Any, Dict, Optional

from fast_service import RequestContext

from minisweagent.exceptions import Submitted

from agsbench_mini_swe_agent.env_service import (
    EnvCreateRequest,
    EnvCreateResponse,
    EnvDestroyRequest,
    EnvExecuteRequest,
    env_create,
    env_destroy,
    env_execute,
)


class RemoteEnv:
    """Drop-in replacement for ``minisweagent.environments.docker.DockerEnvironment``.

    Constructed via :meth:`create`, which performs the ``env_create`` RPC.
    The agent then calls ``execute`` repeatedly; each call is one
    ``env_execute`` RPC. ``cleanup`` issues ``env_destroy``.

    Carries an ``fsm_context`` so cross-host monitor rows are linked under
    the parent ``solve_instance`` call_id, the same way local-mode telemetry
    threads parent_call_id through ``env_execute_call``.
    """

    __slots__ = ("env_id", "_template_vars", "_config_dict", "_fsm_context", "_closed")

    def __init__(
        self,
        env_id: str,
        template_vars: Dict[str, Any],
        config_dict: Dict[str, Any],
        fsm_context: Optional[RequestContext],
    ):
        self.env_id = env_id
        self._template_vars = template_vars
        self._config_dict = config_dict
        self._fsm_context = fsm_context
        self._closed = False

    # === mini-swe-agent Environment interface ==============================

    def execute(
        self, action: Any, cwd: str = "", *, timeout: int | None = None
    ) -> dict:
        # DefaultAgent always passes a dict ({"command": "..."}); be liberal
        # in what we accept since some templates might pass a bare string.
        if isinstance(action, str):
            action_dict: Dict[str, Any] = {"command": action}
        else:
            action_dict = dict(action)

        req = EnvExecuteRequest(
            env_id=self.env_id,
            action=action_dict,
            cwd=cwd,
            timeout=timeout,
        )
        resp = env_execute(req, fsm_context=self._fsm_context)

        if resp.error:
            # Transport / server error — surface as a Python exception so
            # the agent loop can catch and log it the same way it would a
            # local subprocess failure.
            raise RuntimeError(f"env_execute RPC error: {resp.error}")

        if resp.submitted:
            # Reconstruct the same Submitted payload that DockerEnvironment
            # raises locally — DefaultAgent inspects 'extra.exit_status' and
            # 'extra.submission' to terminate the loop with the right
            # bookkeeping.
            raise Submitted(
                {
                    "role": "exit",
                    "content": resp.submission,
                    "extra": {
                        "exit_status": "Submitted",
                        "submission": resp.submission,
                    },
                }
            )

        out: Dict[str, Any] = {
            "output": resp.output,
            "returncode": resp.returncode,
            "exception_info": resp.exception_info,
        }
        if resp.extra:
            out["extra"] = resp.extra
        return out

    def get_template_vars(self, **kwargs) -> dict:
        # The default agent merges this with the model + agent vars when
        # rendering prompts. Most SWE-bench configs don't reference platform
        # info, but we forward whatever the env host returned to keep parity.
        merged = dict(self._template_vars)
        merged.update(kwargs)
        return merged

    def serialize(self) -> dict:
        # Mirror DockerEnvironment.serialize so trajectory files look the
        # same as in single-host mode. The remote-side template_vars
        # include the env config, so re-use it as the recorded config.
        return {
            "info": {
                "config": {
                    "environment": self._config_dict,
                    "environment_type": "agsbench_mini_swe_agent.remote_env.RemoteEnv",
                }
            }
        }

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            env_destroy(
                EnvDestroyRequest(env_id=self.env_id), fsm_context=self._fsm_context
            )
        except Exception:
            # Cleanup is best-effort; the env service also has its own
            # idempotent teardown path.
            pass

    def __del__(self):
        # Match DockerEnvironment's behavior — best-effort destroy on GC.
        try:
            self.cleanup()
        except Exception:
            pass

    # `config` is read by ``_stop_sandbox_sync`` in legacy code paths; we
    # don't expose a real config object since the runner branches on
    # isinstance(env, RemoteEnv) before reaching that helper. A simple
    # attribute placeholder keeps any future accesses from blowing up.
    class _ConfigShim:
        executable = "docker"

    config = _ConfigShim()


def create_remote_env(
    env_config: Dict[str, Any],
    instance: Dict[str, Any],
    startup_command: Optional[str],
    fsm_context: Optional[RequestContext],
) -> RemoteEnv:
    """Wrap an ``env_create`` RPC + result handling in one call.

    ``env_config`` is the same dict the local-mode runner passes to
    ``get_environment`` (with ``--cpuset-cpus`` / ``--memory`` *not* yet
    appended — the env host's own pool injects those, so the orchestrator
    side stays oblivious to remote topology).
    ``startup_command`` is the already-rendered env_startup_command from
    the agent config; we send it across rather than doing a separate
    execute round-trip so the env host can roll it back atomically on
    failure.
    """
    req = EnvCreateRequest(
        image=env_config["image"],
        cwd=env_config.get("cwd", "/"),
        env=env_config.get("env", {}) or {},
        forward_env=env_config.get("forward_env", []) or [],
        timeout=env_config.get("timeout", 30),
        run_args=list(env_config.get("run_args", ["--rm"])),
        container_timeout=env_config.get("container_timeout", "2h"),
        pull_timeout=env_config.get("pull_timeout", 120),
        interpreter=env_config.get("interpreter", ["bash", "-lc"]),
        startup_command=startup_command,
    )
    resp: EnvCreateResponse = env_create(req, fsm_context=fsm_context)
    if resp.error or not resp.env_id:
        raise RuntimeError(f"env_create failed: {resp.error or 'no env_id returned'}")
    return RemoteEnv(
        env_id=resp.env_id,
        template_vars=resp.template_vars,
        config_dict=env_config,
        fsm_context=fsm_context,
    )
