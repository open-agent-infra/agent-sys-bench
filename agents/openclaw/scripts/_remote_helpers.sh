#!/bin/env sh
# Shared helpers for multi-node openclaw deployment.
#
# Sourced by build_image.sh / 4.run_benchmark.sh / 5.cleanup.sh when
# OPENCLAW_MODE=multi.

: "${OPENCLAW_REMOTE_HOST:=<REMOTE_HOST>}"
: "${OPENCLAW_REMOTE_REPO:=agent-sys-bench}"
: "${OPENCLAW_REMOTE_ROOT:=${OPENCLAW_REMOTE_REPO}/agents/openclaw}"

remote_rsync() {
    rsync -az --delete \
        --exclude '.cache/openclaw/' \
        --exclude 'output/' \
        --exclude 'monitoring/prometheus_data/' \
        --exclude 'monitoring/grafana_data/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        ./ "${OPENCLAW_REMOTE_HOST}:${OPENCLAW_REMOTE_ROOT}/"

    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" \
        "mkdir -p ${OPENCLAW_REMOTE_REPO}/benchmarking"
    rsync -az ../benchmarking/openclaw-env-compose.yaml \
        ../benchmarking/openclaw-env-v2-compose.yaml \
        "${OPENCLAW_REMOTE_HOST}:${OPENCLAW_REMOTE_REPO}/benchmarking/"
}

remote_sync_tasks() {
    local tasks_dir="${OPENCLAW_TASKS_DIR:?set OPENCLAW_TASKS_DIR to the local task data dir}"
    local remote_tasks_dir="${OPENCLAW_REMOTE_ROOT}/.cache/tasks"
    echo "[multi] Syncing task files from ${tasks_dir} to ${OPENCLAW_REMOTE_HOST}:${remote_tasks_dir}..."
    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" "mkdir -p ${remote_tasks_dir}"
    rsync -az "${tasks_dir}/" "${OPENCLAW_REMOTE_HOST}:${remote_tasks_dir}/"
}

remote_build_image() {
    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" \
        "cd ${OPENCLAW_REMOTE_ROOT} && docker build -t openclaw:v1 ."
}

remote_build_sandbox_image() {
    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" \
        "cd ${OPENCLAW_REMOTE_ROOT} && bash scripts/build_sandbox_image.sh"
}

remote_compose_up() {
    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" bash -lc "'
        set -e
        export OPENCLAW_ROOT=\$(cd \"\$HOME/${OPENCLAW_REMOTE_ROOT#~/}\" 2>/dev/null && pwd || echo \"${OPENCLAW_REMOTE_ROOT}\")
        export OPENCLAW_REPO_ABS=\$(cd \"\$HOME/${OPENCLAW_REMOTE_REPO#~/}\" 2>/dev/null && pwd || echo \"${OPENCLAW_REMOTE_REPO}\")
        cd \"\$OPENCLAW_REPO_ABS\"
        mkdir -p \"\$OPENCLAW_ROOT/.cache/openclaw/server\"
        mkdir -p \"\$OPENCLAW_ROOT/output\"
        mkdir -p \"\$OPENCLAW_ROOT/sandbox_workspace\"
        OPENCLAW_SERVICE_PORT=${OPENCLAW_SERVICE_PORT:-1803} \
        OPENCLAW_CADVISOR_PORT=${OPENCLAW_CADVISOR_PORT:-8082} \
        OPENCLAW_SANDBOX=\"${OPENCLAW_SANDBOX:-0}\" \
        MOONSHOT_API_KEY=\"${MOONSHOT_API_KEY:-}\" \
        OPENCLAW_TASKS_DIR=\"${OPENCLAW_TASKS_DIR}\" \
        OPENCLAW_TASKS_HOST_DIR=\"\$OPENCLAW_ROOT/.cache/tasks\" \
        OPENCLAW_LLM_PROXY_PORT=${OPENCLAW_LLM_PROXY_PORT:-1801} \
        OPENCLAW_SEARCH_PROXY_PORT=${OPENCLAW_SEARCH_PROXY_PORT:-1802} \
        docker compose -f benchmarking/openclaw-env-compose.yaml up -d
    '"
}

remote_compose_down() {
    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" bash -lc "'
        export OPENCLAW_ROOT=\$(cd \"\$HOME/${OPENCLAW_REMOTE_ROOT#~/}\" 2>/dev/null && pwd || echo \"${OPENCLAW_REMOTE_ROOT}\")
        export OPENCLAW_REPO_ABS=\$(cd \"\$HOME/${OPENCLAW_REMOTE_REPO#~/}\" 2>/dev/null && pwd || echo \"${OPENCLAW_REMOTE_REPO}\")
        cd \"\$OPENCLAW_REPO_ABS\"
        docker compose -f benchmarking/openclaw-env-compose.yaml down
    '"
}

remote_tunnel_up() {
    local svc_port="${OPENCLAW_SERVICE_PORT:-1803}"
    local cad_port="${OPENCLAW_CADVISOR_PORT:-8082}"
    local llm_port="${OPENCLAW_LLM_PROXY_PORT:-1801}"
    local search_port="${OPENCLAW_SEARCH_PROXY_PORT:-1802}"
    local pid_file="${OPENCLAW_TUNNEL_PID_FILE:-/tmp/openclaw-tunnel.pid}"

    remote_tunnel_down >/dev/null 2>&1 || true

    ssh -fN \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -L "0.0.0.0:${svc_port}:127.0.0.1:${svc_port}" \
        -L "0.0.0.0:${cad_port}:127.0.0.1:${cad_port}" \
        -R "127.0.0.1:${llm_port}:127.0.0.1:${llm_port}" \
        -R "127.0.0.1:${search_port}:127.0.0.1:${search_port}" \
        "${OPENCLAW_REMOTE_HOST}"

    pgrep -f "ssh -fN .*-L 0.0.0.0:${svc_port}:127.0.0.1:${svc_port}.*${OPENCLAW_REMOTE_HOST}" \
        > "${pid_file}" || true
}

remote_tunnel_down() {
    local pid_file="${OPENCLAW_TUNNEL_PID_FILE:-/tmp/openclaw-tunnel.pid}"
    local svc_port="${OPENCLAW_SERVICE_PORT:-1803}"
    if [ -f "${pid_file}" ]; then
        while read -r pid; do
            [ -n "${pid}" ] && kill "${pid}" 2>/dev/null || true
        done < "${pid_file}"
        rm -f "${pid_file}"
    fi
    pkill -f "ssh -fN .*-L 0.0.0.0:${svc_port}:127.0.0.1:${svc_port}.*${OPENCLAW_REMOTE_HOST}" 2>/dev/null || true
}

remote_cleanup_sandbox() {
    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" \
        "docker ps -aq --filter 'name=openclaw-sandbox' | xargs -r docker rm -f 2>/dev/null || true"
}

# ============================================================================
# Multi-node v2 helpers (env-manager mode)
# ============================================================================

remote_build_ssh_sandbox_image() {
    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" \
        "cd ${OPENCLAW_REMOTE_ROOT} && bash scripts/build_ssh_sandbox_image.sh"
}

remote_compose_up_v2() {
    local <REMOTE_HOST>_ip
    <REMOTE_HOST>_ip=$(ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" \
        "hostname -I | awk '{print \$1}'")
    echo "[multi-v2] <REMOTE_HOST> IP: ${<REMOTE_HOST>_ip}"

    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" bash -lc "'
        set -e
        export OPENCLAW_ROOT=\$(cd \"\$HOME/${OPENCLAW_REMOTE_ROOT#~/}\" 2>/dev/null && pwd || echo \"${OPENCLAW_REMOTE_ROOT}\")
        export OPENCLAW_REPO_ABS=\$(cd \"\$HOME/${OPENCLAW_REMOTE_REPO#~/}\" 2>/dev/null && pwd || echo \"${OPENCLAW_REMOTE_REPO}\")
        cd \"\$OPENCLAW_REPO_ABS\"
        mkdir -p \"\$OPENCLAW_ROOT/.cache/openclaw/env\"
        OPENCLAW_ENV_PORT=${OPENCLAW_ENV_PORT:-1804} \
        OPENCLAW_CADVISOR_PORT=${OPENCLAW_CADVISOR_PORT:-8082} \
        OPENCLAW_ENV_ADVERTISE_HOST=\"${<REMOTE_HOST>_ip}\" \
        docker compose -f benchmarking/openclaw-env-v2-compose.yaml up -d
    '"
}

remote_compose_down_v2() {
    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" bash -lc "'
        export OPENCLAW_ROOT=\$(cd \"\$HOME/${OPENCLAW_REMOTE_ROOT#~/}\" 2>/dev/null && pwd || echo \"${OPENCLAW_REMOTE_ROOT}\")
        export OPENCLAW_REPO_ABS=\$(cd \"\$HOME/${OPENCLAW_REMOTE_REPO#~/}\" 2>/dev/null && pwd || echo \"${OPENCLAW_REMOTE_REPO}\")
        cd \"\$OPENCLAW_REPO_ABS\"
        docker compose -f benchmarking/openclaw-env-v2-compose.yaml down
    '"
}

remote_tunnel_up_v2() {
    local env_port="${OPENCLAW_ENV_PORT:-1804}"
    local cad_port="${OPENCLAW_CADVISOR_PORT:-8082}"
    local pid_file="${OPENCLAW_TUNNEL_PID_FILE:-/tmp/openclaw-tunnel.pid}"

    remote_tunnel_down >/dev/null 2>&1 || true

    ssh -fN \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -L "0.0.0.0:${env_port}:127.0.0.1:${env_port}" \
        -L "0.0.0.0:${cad_port}:127.0.0.1:${cad_port}" \
        "${OPENCLAW_REMOTE_HOST}"

    pgrep -f "ssh -fN .*-L 0.0.0.0:${env_port}:127.0.0.1:${env_port}.*${OPENCLAW_REMOTE_HOST}" \
        > "${pid_file}" || true
}

remote_cleanup_ssh_sandbox() {
    ssh -o BatchMode=yes "${OPENCLAW_REMOTE_HOST}" \
        "docker ps -aq --filter 'name=openclaw-ssh-sandbox' | xargs -r docker rm -f 2>/dev/null || true"
}
