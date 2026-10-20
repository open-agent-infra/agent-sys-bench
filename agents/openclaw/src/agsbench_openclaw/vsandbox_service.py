"""Server-side vsandbox RPCs for openclaw multi-node v2.

Provides HTTP-based command execution inside Docker containers via
``docker exec``, replacing the SSH transport layer.  All RPCs are
co-hosted with the env-manager on the same fast-service process
(port 1804).

RPCs:

* :func:`vsandbox_exec` — run a command inside a sandbox container
* :func:`vsandbox_shell` — run a shell script inside a sandbox container
* :func:`vsandbox_put_file` — write a file into a sandbox container
* :func:`vsandbox_get_file` — read a file from a sandbox container
"""

import base64
import logging
import subprocess
import tempfile
from typing import Dict, List, Optional

from pydantic import BaseModel

from fast_service import RequestContext

from agsbench_openclaw.env_service import _REGISTRY_LOCK, _SANDBOX_REGISTRY
from agsbench_openclaw.run_task import OCLAW_RUN_FSM

log = logging.getLogger("agsbench_openclaw.vsandbox_service")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_container(sandbox_id: str) -> str:
    """Look up the Docker container ID for a sandbox_id."""
    with _REGISTRY_LOCK:
        container_id = _SANDBOX_REGISTRY.get(sandbox_id)
    if container_id is None:
        raise ValueError(f"sandbox {sandbox_id!r} not found in registry")
    return container_id


def _docker_exec(
    container_id: str,
    command: List[str],
    stdin_data: Optional[bytes] = None,
    workdir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    user: str = "sandbox",
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """Run a command inside a Docker container via ``docker exec``."""
    cmd = ["docker", "exec"]
    if workdir:
        cmd.extend(["-w", workdir])
    if user:
        cmd.extend(["-u", user])
    if env:
        for k, v in env.items():
            cmd.extend(["-e", f"{k}={v}"])
    if stdin_data is not None:
        cmd.append("-i")
    cmd.append(container_id)
    cmd.extend(command)

    return subprocess.run(
        cmd,
        input=stdin_data,
        capture_output=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# RPC: vsandbox_exec
# ---------------------------------------------------------------------------


class VsandboxExecRequest(BaseModel):
    sandbox_id: str
    command: str
    workdir: str = "/workspace"
    env: Dict[str, str] = {}
    stdin_b64: str = ""
    timeout: int = 300

    class Config:
        arbitrary_types_allowed = True


class VsandboxExecResponse(BaseModel):
    stdout_b64: str = ""
    stderr_b64: str = ""
    exit_code: int = 0
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


@OCLAW_RUN_FSM.fast_service
def vsandbox_exec(
    req: VsandboxExecRequest,
    fsm_context: RequestContext = None,
) -> VsandboxExecResponse:
    """Execute a command inside a sandbox container."""
    try:
        container_id = _resolve_container(req.sandbox_id)
    except ValueError as e:
        return VsandboxExecResponse(exit_code=-1, error=str(e))

    stdin_data = base64.b64decode(req.stdin_b64) if req.stdin_b64 else None

    try:
        proc = _docker_exec(
            container_id,
            ["bash", "-c", req.command],
            stdin_data=stdin_data,
            workdir=req.workdir,
            env=req.env or None,
            timeout=req.timeout,
        )
        return VsandboxExecResponse(
            stdout_b64=base64.b64encode(proc.stdout).decode(),
            stderr_b64=base64.b64encode(proc.stderr).decode(),
            exit_code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return VsandboxExecResponse(
            exit_code=-1,
            error=f"docker exec timed out after {req.timeout}s",
        )
    except Exception as e:
        log.exception("vsandbox_exec failed: %s", e)
        return VsandboxExecResponse(exit_code=-1, error=str(e))


# ---------------------------------------------------------------------------
# RPC: vsandbox_shell
# ---------------------------------------------------------------------------


class VsandboxShellRequest(BaseModel):
    sandbox_id: str
    script: str
    args: List[str] = []
    stdin_b64: str = ""
    timeout: int = 300

    class Config:
        arbitrary_types_allowed = True


class VsandboxShellResponse(BaseModel):
    stdout_b64: str = ""
    stderr_b64: str = ""
    code: int = 0
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


@OCLAW_RUN_FSM.fast_service
def vsandbox_shell(
    req: VsandboxShellRequest,
    fsm_context: RequestContext = None,
) -> VsandboxShellResponse:
    """Run a shell script inside a sandbox container.

    Writes the script to a temp file inside the container, executes it
    with optional args, then cleans up.
    """
    try:
        container_id = _resolve_container(req.sandbox_id)
    except ValueError as e:
        return VsandboxShellResponse(code=-1, error=str(e))

    # Write script into container via docker exec + stdin pipe
    script_path = f"/tmp/_vsandbox_script_{id(req):x}.sh"
    try:
        # Write script content
        write_proc = _docker_exec(
            container_id,
            ["bash", "-c", f"cat > {script_path} && chmod +x {script_path}"],
            stdin_data=req.script.encode("utf-8"),
        )
        if write_proc.returncode != 0:
            return VsandboxShellResponse(
                code=-1,
                error=f"Failed to write script: {write_proc.stderr.decode(errors='replace')[:200]}",
            )

        # Execute the script via bash
        exec_cmd = ["bash", script_path] + req.args
        stdin_data = base64.b64decode(req.stdin_b64) if req.stdin_b64 else None

        proc = _docker_exec(
            container_id,
            exec_cmd,
            stdin_data=stdin_data,
            workdir="/workspace",
            timeout=req.timeout,
        )
        return VsandboxShellResponse(
            stdout_b64=base64.b64encode(proc.stdout).decode(),
            stderr_b64=base64.b64encode(proc.stderr).decode(),
            code=proc.returncode,
        )
    except subprocess.TimeoutExpired:
        return VsandboxShellResponse(
            code=-1,
            error=f"script execution timed out after {req.timeout}s",
        )
    except Exception as e:
        log.exception("vsandbox_shell failed: %s", e)
        return VsandboxShellResponse(code=-1, error=str(e))
    finally:
        # Clean up the temp script
        try:
            _docker_exec(container_id, ["rm", "-f", script_path], timeout=10)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RPC: vsandbox_put_file
# ---------------------------------------------------------------------------


class VsandboxPutFileRequest(BaseModel):
    sandbox_id: str
    dest_path: str
    content_b64: str
    mode: str = ""  # optional chmod, e.g. "0755"

    class Config:
        arbitrary_types_allowed = True


class VsandboxPutFileResponse(BaseModel):
    ok: bool = True
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


@OCLAW_RUN_FSM.fast_service
def vsandbox_put_file(
    req: VsandboxPutFileRequest,
    fsm_context: RequestContext = None,
) -> VsandboxPutFileResponse:
    """Write a file into a sandbox container."""
    try:
        container_id = _resolve_container(req.sandbox_id)
    except ValueError as e:
        return VsandboxPutFileResponse(ok=False, error=str(e))

    content = base64.b64decode(req.content_b64)

    try:
        # Create parent directory (as root since sandbox user may not have perms)
        parent_dir = "/".join(req.dest_path.rsplit("/", 1)[:-1]) or "/"
        _docker_exec(
            container_id,
            ["mkdir", "-p", parent_dir],
            user="root",
            timeout=10,
        )
        _docker_exec(
            container_id,
            ["chown", "sandbox:sandbox", parent_dir],
            user="root",
            timeout=10,
        )

        # Write file content via stdin pipe
        proc = _docker_exec(
            container_id,
            ["bash", "-c", f"cat > {req.dest_path}"],
            stdin_data=content,
            timeout=60,
        )
        if proc.returncode != 0:
            return VsandboxPutFileResponse(
                ok=False,
                error=f"write failed: {proc.stderr.decode(errors='replace')[:200]}",
            )

        # Optional chmod
        if req.mode:
            _docker_exec(
                container_id,
                ["chmod", req.mode, req.dest_path],
                timeout=10,
            )

        return VsandboxPutFileResponse(ok=True)
    except Exception as e:
        log.exception("vsandbox_put_file failed: %s", e)
        return VsandboxPutFileResponse(ok=False, error=str(e))


# ---------------------------------------------------------------------------
# RPC: vsandbox_get_file
# ---------------------------------------------------------------------------


class VsandboxGetFileRequest(BaseModel):
    sandbox_id: str
    src_path: str

    class Config:
        arbitrary_types_allowed = True


class VsandboxGetFileResponse(BaseModel):
    content_b64: str = ""
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


@OCLAW_RUN_FSM.fast_service
def vsandbox_get_file(
    req: VsandboxGetFileRequest,
    fsm_context: RequestContext = None,
) -> VsandboxGetFileResponse:
    """Read a file from a sandbox container."""
    try:
        container_id = _resolve_container(req.sandbox_id)
    except ValueError as e:
        return VsandboxGetFileResponse(error=str(e))

    try:
        proc = _docker_exec(
            container_id,
            ["cat", req.src_path],
            timeout=60,
        )
        if proc.returncode != 0:
            return VsandboxGetFileResponse(
                error=f"cat failed (rc={proc.returncode}): {proc.stderr.decode(errors='replace')[:200]}",
            )
        return VsandboxGetFileResponse(
            content_b64=base64.b64encode(proc.stdout).decode(),
        )
    except Exception as e:
        log.exception("vsandbox_get_file failed: %s", e)
        return VsandboxGetFileResponse(error=str(e))
