#!/bin/env sh
# Shared helpers for the multi-node deployment scripts.
#
# Sourced by build_image.sh / 4.run_benchmark.sh / 5.cleanup.sh when
# MINI_SWEAGENT_MODE=multi. Owns the SSH/rsync details so the higher-level
# scripts only deal with intent (build, start, stop).

# Required env vars in multi mode:
#   MINI_SWEAGENT_REMOTE_HOST   ssh host alias for the env machine (default: <REMOTE_HOST>)
#   MINI_SWEAGENT_REMOTE_ROOT   absolute path to agents/mini-swe-agent/ on the remote
#                               (default: ~/agent-sys-bench/agents/mini-swe-agent)
#   MINI_SWEAGENT_REMOTE_REPO   absolute path to the repo root on the remote
#                               (default: ~/agent-sys-bench). Used to host
#                               the env-side compose file.

: "${MINI_SWEAGENT_REMOTE_HOST:=<REMOTE_HOST>}"
: "${MINI_SWEAGENT_REMOTE_REPO:=agent-sys-bench}"
: "${MINI_SWEAGENT_REMOTE_ROOT:=${MINI_SWEAGENT_REMOTE_REPO}/agents/mini-swe-agent}"

# Resolve the env host's reachable IP. We need this because:
#   1. The orchestrator container has to dial an IP, not the ssh alias.
#   2. fast-service's config_loader substitutes ${MINI_SWE_ENV_HOST} from
#      the orchestrator container's env; that var must hold an IP that is
#      routable from inside docker.
# Strategy: ask the remote for its first non-loopback IPv4. Override via
# MINI_SWE_ENV_HOST in the caller's environment if the default picks wrong.
remote_resolve_env_host() {
    if [ -n "${MINI_SWE_ENV_HOST:-}" ]; then
        echo "${MINI_SWE_ENV_HOST}"
        return
    fi
    ssh -o BatchMode=yes "${MINI_SWEAGENT_REMOTE_HOST}" \
        "hostname -I | awk '{print \$2}'" 2>/dev/null \
        | tr -d '[:space:]'
}

remote_rsync() {
    # rsync agsbench_mini_swe_agent/ + .cache/fast-service to the env host. Excludes
    # everything that is local-only or large (trajectories, monitor csvs,
    # prometheus data).
    rsync -az --delete \
        --exclude '.cache/mini-swe-agent/' \
        --exclude 'trajectories/' \
        --exclude 'analysis/' \
        --exclude 'logs/' \
        --exclude 'monitoring/prometheus_data/' \
        --exclude 'monitoring/grafana_data/' \
        --exclude 'monitoring/snapshots/' \
        --exclude '__pycache__/' \
        --exclude '*.pyc' \
        ./ "${MINI_SWEAGENT_REMOTE_HOST}:${MINI_SWEAGENT_REMOTE_ROOT}/"

    # Also sync the env-side compose file under benchmarking/. The remote
    # repo root may or may not exist; create it.
    ssh -o BatchMode=yes "${MINI_SWEAGENT_REMOTE_HOST}" \
        "mkdir -p ${MINI_SWEAGENT_REMOTE_REPO}/benchmarking"
    rsync -az ../benchmarking/mini-sweagent-env-compose.yaml \
        "${MINI_SWEAGENT_REMOTE_HOST}:${MINI_SWEAGENT_REMOTE_REPO}/benchmarking/"
}

remote_build_image() {
    ssh -o BatchMode=yes "${MINI_SWEAGENT_REMOTE_HOST}" \
        "cd ${MINI_SWEAGENT_REMOTE_ROOT} && docker build -t mini-swe-agent:v1 ."
}

remote_compose_up() {
    # docker compose bind-mounts need absolute paths. The default
    # MINI_SWEAGENT_REMOTE_ROOT is relative to $HOME on the remote, so we
    # resolve it there (inside the SSH heredoc) rather than locally — that
    # way ~/.../agents/mini-swe-agent or "agent-sys-bench/agents/mini-swe-agent" both
    # work and the docker compose call always sees /home/<user>/... .
    ssh -o BatchMode=yes "${MINI_SWEAGENT_REMOTE_HOST}" bash -lc "'
        set -e
        export MINI_SWEAGENT_ROOT=\$(cd \"\$HOME/${MINI_SWEAGENT_REMOTE_ROOT#~/}\" 2>/dev/null && pwd || echo \"${MINI_SWEAGENT_REMOTE_ROOT}\")
        export MINI_SWEAGENT_REPO_ABS=\$(cd \"\$HOME/${MINI_SWEAGENT_REMOTE_REPO#~/}\" 2>/dev/null && pwd || echo \"${MINI_SWEAGENT_REMOTE_REPO}\")
        cd \"\$MINI_SWEAGENT_REPO_ABS\"
        mkdir -p \"\$MINI_SWEAGENT_ROOT/.cache/mini-swe-agent/env\"
        MINI_SWE_ENV_PORT=${MINI_SWE_ENV_PORT:-1802} \
        MINI_SWE_CADVISOR_PORT=${MINI_SWE_CADVISOR_PORT:-8082} \
        MINI_SWE_CPUSETS=\"${MINI_SWE_CPUSETS:-0-1,2-3,4-5,6-7}\" \
        MINI_SWE_MEMORY=\"${MINI_SWE_MEMORY:-4g}\" \
        docker compose -f benchmarking/mini-sweagent-env-compose.yaml up -d
    '"
}

remote_tunnel_up() {
    # Open SSH tunnels for env_service (1802) and env cAdvisor (8082).
    # The remote host's firewall blocks inbound on these ports, so we
    # forward via the always-open SSH port. The tunnels listen on 0.0.0.0
    # of this host so docker containers can reach them via the host-gateway
    # alias (added as `extra_hosts: host.docker.internal:host-gateway` in
    # the orchestrator compose).
    local env_port="${MINI_SWE_ENV_PORT:-1802}"
    local cad_port="${MINI_SWE_CADVISOR_PORT:-8082}"
    local pid_file="${MINI_SWE_TUNNEL_PID_FILE:-/tmp/mini-swe-tunnel.pid}"

    # Tear down any stale tunnel first — leftover from a crashed previous
    # run would silently mask a real connectivity problem.
    remote_tunnel_down >/dev/null 2>&1 || true

    # -f goes to background AFTER successful auth; -N means no remote
    # command; -o ExitOnForwardFailure=yes makes ssh fail immediately if
    # the local port is already bound.
    ssh -fN \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -L "0.0.0.0:${env_port}:127.0.0.1:${env_port}" \
        -L "0.0.0.0:${cad_port}:127.0.0.1:${cad_port}" \
        "${MINI_SWEAGENT_REMOTE_HOST}"

    # Record the PID so cleanup can target it without grep heuristics. We
    # find the freshly-backgrounded ssh by matching the forward spec.
    pgrep -f "ssh -fN .*-L 0.0.0.0:${env_port}:127.0.0.1:${env_port}.*${MINI_SWEAGENT_REMOTE_HOST}" \
        > "${pid_file}" || true
}

remote_tunnel_down() {
    local pid_file="${MINI_SWE_TUNNEL_PID_FILE:-/tmp/mini-swe-tunnel.pid}"
    local env_port="${MINI_SWE_ENV_PORT:-1802}"
    if [ -f "${pid_file}" ]; then
        while read -r pid; do
            [ -n "${pid}" ] && kill "${pid}" 2>/dev/null || true
        done < "${pid_file}"
        rm -f "${pid_file}"
    fi
    # Belt-and-suspenders: also kill any ssh forwarder matching our spec.
    pkill -f "ssh -fN .*-L 0.0.0.0:${env_port}:127.0.0.1:${env_port}.*${MINI_SWEAGENT_REMOTE_HOST}" 2>/dev/null || true
}

remote_compose_down() {
    ssh -o BatchMode=yes "${MINI_SWEAGENT_REMOTE_HOST}" bash -lc "'
        export MINI_SWEAGENT_ROOT=\$(cd \"\$HOME/${MINI_SWEAGENT_REMOTE_ROOT#~/}\" 2>/dev/null && pwd || echo \"${MINI_SWEAGENT_REMOTE_ROOT}\")
        export MINI_SWEAGENT_REPO_ABS=\$(cd \"\$HOME/${MINI_SWEAGENT_REMOTE_REPO#~/}\" 2>/dev/null && pwd || echo \"${MINI_SWEAGENT_REMOTE_REPO}\")
        cd \"\$MINI_SWEAGENT_REPO_ABS\"
        docker compose -f benchmarking/mini-sweagent-env-compose.yaml down
    '"
}
