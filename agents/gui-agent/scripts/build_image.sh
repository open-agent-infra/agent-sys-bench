#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
AGSBENCH_ENV="${AGENT_DIR}/../../.env"
[ -f "$AGSBENCH_ENV" ] && set -a && . "$AGSBENCH_ENV" && set +a

cd "$AGENT_DIR"

# Stage docker-cli from host
mkdir -p .cache
if [ ! -f .cache/docker-cli ]; then
  cp "$(which docker)" .cache/docker-cli
fi

# Stage OSWorld (desktop_env + mm_agents) at a pinned commit. PyPI's
# desktop-env lags the repo and lacks mm_agents.kimi / owl15_agent.
# Keep in sync with PINNED_COMMIT in request/.../computer-use/prepare_data.py.
OSWORLD_COMMIT=091f5ef1d5544bc74953c77875d5feb5bed30108
if [ ! -s ".cache/osworld-${OSWORLD_COMMIT}.tar.gz" ]; then
  curl -fL --retry 5 -o ".cache/osworld-${OSWORLD_COMMIT}.tar.gz.tmp" \
    "${GITHUB_PROXY:-}https://github.com/xlang-ai/OSWorld/archive/${OSWORLD_COMMIT}.tar.gz"
  mv ".cache/osworld-${OSWORLD_COMMIT}.tar.gz.tmp" ".cache/osworld-${OSWORLD_COMMIT}.tar.gz"
fi
cp ".cache/osworld-${OSWORLD_COMMIT}.tar.gz" .cache/osworld.tar.gz

docker build "$@" \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple/}" \
  --build-context fast-service=../../sdk/fast-service \
  --build-context scenario-protocol=../../sdk/scenario-protocol \
  -t gui-agent:v1 .
