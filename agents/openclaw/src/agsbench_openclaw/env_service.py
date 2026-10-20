"""Server-side env-manager for openclaw multi-node v2.

Creates Docker containers with sshd on the remote host, returning SSH
connection info so the orchestrator's openclaw CLI can use the ``ssh``
sandbox backend for tool execution.

RPCs:

* :func:`sandbox_create` — start a Docker container with sshd, generate
  an SSH key pair, and return connection details.
* :func:`sandbox_destroy` — ``docker rm -f`` the container.
"""

import logging
import os
import subprocess
import tempfile
import uuid
from threading import Lock
from typing import Dict, Optional

from pydantic import BaseModel

from fast_service import RequestContext

from agsbench_openclaw.run_task import OCLAW_RUN_FSM

log = logging.getLogger("agsbench_openclaw.env_service")

_SANDBOX_REGISTRY: Dict[str, str] = {}  # sandbox_id -> container_id
_REGISTRY_LOCK = Lock()


class SandboxCreateRequest(BaseModel):
    image: str = "openclaw-sandbox-ssh:bookworm-slim"
    mode: str = "ssh"  # "ssh" or "vsandbox"
    cpuset: Optional[str] = None
    memory: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class SandboxCreateResponse(BaseModel):
    sandbox_id: str = ""
    host: str = ""
    port: int = 0
    user: str = "sandbox"
    private_key_data: str = ""
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class SandboxDestroyRequest(BaseModel):
    sandbox_id: str

    class Config:
        arbitrary_types_allowed = True


class SandboxDestroyResponse(BaseModel):
    ok: bool = True
    error: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


def _get_host_ip() -> str:
    return os.environ.get("OPENCLAW_ENV_ADVERTISE_HOST", "127.0.0.1")


@OCLAW_RUN_FSM.fast_service
def sandbox_create(
    req: SandboxCreateRequest,
    fsm_context: RequestContext = None,
) -> SandboxCreateResponse:
    use_vsandbox = req.mode == "vsandbox"
    prefix = "openclaw-vsandbox" if use_vsandbox else "openclaw-ssh-sandbox"
    sandbox_id = f"{prefix}-{uuid.uuid4().hex[:8]}"

    # --- vsandbox mode: simple container, no SSH ---
    if use_vsandbox:
        image = req.image  # use ssh image as-is (has sudo, stays running via sshd -D)
        docker_cmd = ["docker", "run", "-d", "--name", sandbox_id]
        if req.cpuset:
            docker_cmd.append(f"--cpuset-cpus={req.cpuset}")
        if req.memory:
            docker_cmd.append(f"--memory={req.memory}")
        docker_cmd.append(image)

        try:
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
            )
            container_id = result.stdout.strip()
        except Exception as e:
            return SandboxCreateResponse(error=f"docker run failed: {e}")

        # Create /workspace owned by sandbox user
        try:
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "-u",
                    "root",
                    container_id,
                    "bash",
                    "-c",
                    "mkdir -p /workspace && chown sandbox:sandbox /workspace",
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
        except Exception as e:
            subprocess.run(
                ["docker", "rm", "-f", container_id], capture_output=True, timeout=30
            )
            return SandboxCreateResponse(error=f"workspace setup failed: {e}")

        with _REGISTRY_LOCK:
            _SANDBOX_REGISTRY[sandbox_id] = container_id

        host = _get_host_ip()
        log.info("Created vsandbox %s (container=%s)", sandbox_id, container_id[:12])

        return SandboxCreateResponse(
            sandbox_id=sandbox_id,
            host=host,
            port=0,
            user="sandbox",
            private_key_data="",
        )

    # --- SSH mode (original) ---

    tmpdir = tempfile.mkdtemp(prefix="oclaw-ssh-")
    key_path = os.path.join(tmpdir, "id_ed25519")
    pub_path = key_path + ".pub"

    try:
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", key_path, "-N", "", "-q"],
            check=True,
            timeout=30,
        )
        private_key_data = open(key_path).read()
        public_key_data = open(pub_path).read().strip()
    except Exception as e:
        return SandboxCreateResponse(error=f"ssh-keygen failed: {e}")
    finally:
        for p in (key_path, pub_path):
            try:
                os.unlink(p)
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass

    docker_cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        sandbox_id,
        "-p",
        "0:22",
    ]
    if req.cpuset:
        docker_cmd.append(f"--cpuset-cpus={req.cpuset}")
    if req.memory:
        docker_cmd.append(f"--memory={req.memory}")
    docker_cmd.append(req.image)

    try:
        result = subprocess.run(
            docker_cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        container_id = result.stdout.strip()
    except Exception as e:
        return SandboxCreateResponse(error=f"docker run failed: {e}")

    try:
        subprocess.run(
            [
                "docker",
                "exec",
                container_id,
                "bash",
                "-c",
                f'mkdir -p /home/sandbox/.ssh && echo "{public_key_data}" >> /home/sandbox/.ssh/authorized_keys && '
                "chmod 600 /home/sandbox/.ssh/authorized_keys && chown -R sandbox:sandbox /home/sandbox/.ssh && "
                "mkdir -p /workspace && chown sandbox:sandbox /workspace",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except Exception as e:
        subprocess.run(
            ["docker", "rm", "-f", container_id], capture_output=True, timeout=30
        )
        return SandboxCreateResponse(error=f"key injection failed: {e}")

    try:
        subprocess.run(
            ["docker", "exec", container_id, "/usr/sbin/sshd"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    except Exception as e:
        subprocess.run(
            ["docker", "rm", "-f", container_id], capture_output=True, timeout=30
        )
        return SandboxCreateResponse(error=f"sshd start failed: {e}")

    try:
        result = subprocess.run(
            ["docker", "port", container_id, "22"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        port_line = result.stdout.strip().split("\n")[0]
        mapped_port = int(port_line.rsplit(":", 1)[-1])
    except Exception as e:
        subprocess.run(
            ["docker", "rm", "-f", container_id], capture_output=True, timeout=30
        )
        return SandboxCreateResponse(error=f"port query failed: {e}")

    with _REGISTRY_LOCK:
        _SANDBOX_REGISTRY[sandbox_id] = container_id

    host = _get_host_ip()
    log.info(
        "Created sandbox %s (container=%s, host=%s, port=%d)",
        sandbox_id,
        container_id[:12],
        host,
        mapped_port,
    )

    return SandboxCreateResponse(
        sandbox_id=sandbox_id,
        host=host,
        port=mapped_port,
        user="sandbox",
        private_key_data=private_key_data,
    )


@OCLAW_RUN_FSM.fast_service
def sandbox_destroy(
    req: SandboxDestroyRequest,
    fsm_context: RequestContext = None,
) -> SandboxDestroyResponse:
    with _REGISTRY_LOCK:
        container_id = _SANDBOX_REGISTRY.pop(req.sandbox_id, None)
    if container_id is None:
        log.info("sandbox_destroy: %s not found (already destroyed?)", req.sandbox_id)
        return SandboxDestroyResponse(ok=True)

    try:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as e:
        return SandboxDestroyResponse(ok=False, error=f"docker rm failed: {e}")

    log.info("Destroyed sandbox %s", req.sandbox_id)
    return SandboxDestroyResponse(ok=True)
