#!/usr/bin/env bash
# Shared helpers for multi-node OSWorld deployment.
# Sourced by run_kimi_bench.sh when OS_WORLD_MODE=multi.

: "${OS_WORLD_REMOTE_HOST:=<REMOTE_HOST>}"
: "${OS_WORLD_REMOTE_REPO:=agent-sys-bench}"
: "${OS_WORLD_REMOTE_ROOT:=${OS_WORLD_REMOTE_REPO}/agents/gui-agent}"
: "${OS_WORLD_ENV_PORT:=1905}"
: "${OS_WORLD_CADVISOR_PORT:=8082}"
: "${OS_WORLD_SSH_CONFIG:=${HOME}/.ssh/config}"
: "${OS_WORLD_TUNNEL_PID_FILE:=/tmp/osw-tunnel.pid}"
: "${OS_WORLD_REMOTE_DATA_DIR:=${OS_WORLD_REMOTE_ROOT}/.cache/data}"

_SSH="ssh -F ${OS_WORLD_SSH_CONFIG} -o BatchMode=yes"

remote_rsync() {
    local rsync_ssh="ssh -F ${OS_WORLD_SSH_CONFIG}"
    rsync -az --delete -e "${rsync_ssh}" \
        --exclude '.cache/' \
        --exclude 'trajectories/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        "${OS_WORLD_ROOT}/" \
        "${OS_WORLD_REMOTE_HOST}:${OS_WORLD_REMOTE_ROOT}/"

    ${_SSH} "${OS_WORLD_REMOTE_HOST}" \
        "mkdir -p ~/\${OS_WORLD_REMOTE_REPO:-${OS_WORLD_REMOTE_REPO}}/benchmarking"
    rsync -az -e "${rsync_ssh}" \
        "${BENCHMARKING_DIR}/os-world-env-compose.yaml" \
        "${OS_WORLD_REMOTE_HOST}:~/${OS_WORLD_REMOTE_REPO}/benchmarking/"
}

remote_compose_up() {
    ${_SSH} "${OS_WORLD_REMOTE_HOST}" bash -lc "'
        set -e
        export OS_WORLD_ROOT=\$(cd ~/\"${OS_WORLD_REMOTE_ROOT}\" && pwd)
        export OS_WORLD_ARTIFACT_DIR=\"${OS_WORLD_REMOTE_DATA_DIR}/runs/${RUN_ID}\"
        export OS_WORLD_DATA_DIR=\"${OS_WORLD_REMOTE_DATA_DIR}\"
        export OS_WORLD_RUN_ID=\"${RUN_ID}\"
        export OS_WORLD_ENV_PORT=${OS_WORLD_ENV_PORT}
        export OS_WORLD_CADVISOR_PORT=${OS_WORLD_CADVISOR_PORT}
        export OS_WORLD_CPUSET_CPUS="${OS_WORLD_CPUSET_CPUS:-32-63}"
        export OS_WORLD_VM_BOOT_TIMEOUT="${OS_WORLD_VM_BOOT_TIMEOUT:-600}"

        mkdir -p \"\${OS_WORLD_ARTIFACT_DIR}/.cache/os-world/server\" \
                 \"\${OS_WORLD_ARTIFACT_DIR}/monitoring/os-world-env\" \
                 \"\${OS_WORLD_ARTIFACT_DIR}/intermediate/os-world-env\"

        cd ~/\"${OS_WORLD_REMOTE_REPO}\"
        docker compose -f benchmarking/os-world-env-compose.yaml \
            --project-name \"osw-env-${RUN_ID}\" up -d
    '"
}

remote_tunnel_up() {
    local env_port="${OS_WORLD_ENV_PORT}"
    local cad_port="${OS_WORLD_CADVISOR_PORT}"

    remote_tunnel_down >/dev/null 2>&1 || true

    ssh -F "${OS_WORLD_SSH_CONFIG}" -fN \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -L "0.0.0.0:${env_port}:127.0.0.1:${env_port}" \
        -L "0.0.0.0:${cad_port}:127.0.0.1:${cad_port}" \
        "${OS_WORLD_REMOTE_HOST}"

    pgrep -f "ssh -F.*-L 0.0.0.0:${env_port}:127.0.0.1:${env_port}.*${OS_WORLD_REMOTE_HOST}" \
        > "${OS_WORLD_TUNNEL_PID_FILE}" || true

    echo "SSH tunnels opened (fwd:${env_port},${cad_port})"
}

remote_tunnel_down() {
    local env_port="${OS_WORLD_ENV_PORT:-1905}"
    if [ -f "${OS_WORLD_TUNNEL_PID_FILE}" ]; then
        while read -r pid; do
            [ -n "${pid}" ] && kill "${pid}" 2>/dev/null || true
        done < "${OS_WORLD_TUNNEL_PID_FILE}"
        rm -f "${OS_WORLD_TUNNEL_PID_FILE}"
    fi
    pkill -f "ssh -F.*-L 0.0.0.0:${env_port}:127.0.0.1:${env_port}.*${OS_WORLD_REMOTE_HOST}" 2>/dev/null || true
}

remote_compose_down() {
    ${_SSH} "${OS_WORLD_REMOTE_HOST}" bash -lc "'
        cd ~/\"${OS_WORLD_REMOTE_REPO}\" 2>/dev/null || true
        docker compose -f benchmarking/os-world-env-compose.yaml \
            --project-name \"osw-env-${RUN_ID}\" down 2>/dev/null || true
    '"
}

remote_wait_env() {
    local port="${OS_WORLD_ENV_PORT}"
    local max_wait=60
    local elapsed=0
    echo -n "Waiting for env-manager tunnel on localhost:${port} "
    while [ $elapsed -lt $max_wait ]; do
        if bash -c "echo >/dev/tcp/127.0.0.1/${port}" 2>/dev/null; then
            echo " ready (${elapsed}s)"
            return 0
        fi
        echo -n "."
        sleep 2
        elapsed=$((elapsed + 2))
    done
    echo " TIMEOUT after ${max_wait}s"
    return 1
}

remote_collect_artifacts() {
    local local_artifact_dir="${1:-${ARTIFACT_DIR}}"
    local remote_artifact_dir="${OS_WORLD_REMOTE_DATA_DIR}/runs/${RUN_ID}"
    local rsync_ssh="ssh -F ${OS_WORLD_SSH_CONFIG}"

    echo "Collecting artifacts from ${OS_WORLD_REMOTE_HOST}..."

    mkdir -p "${local_artifact_dir}/monitoring/os-world-env" \
              "${local_artifact_dir}/intermediate/os-world-env"

    rsync -az -e "${rsync_ssh}" \
        "${OS_WORLD_REMOTE_HOST}:${remote_artifact_dir}/monitoring/os-world-env/" \
        "${local_artifact_dir}/monitoring/os-world-env/" 2>/dev/null || true

    rsync -az -e "${rsync_ssh}" \
        "${OS_WORLD_REMOTE_HOST}:${remote_artifact_dir}/intermediate/os-world-env/" \
        "${local_artifact_dir}/intermediate/os-world-env/" 2>/dev/null || true

    echo "Artifacts collected to ${local_artifact_dir}/{monitoring,intermediate}/os-world-env/"
}
