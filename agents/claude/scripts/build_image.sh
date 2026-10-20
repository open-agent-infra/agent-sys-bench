#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
AGSBENCH_ENV="${AGENT_DIR}/../../.env"
[ -f "$AGSBENCH_ENV" ] && set -a && . "$AGSBENCH_ENV" && set +a

cd "$AGENT_DIR"

# Stage docker-cli from host (Dockerfile COPYs it into the image)
mkdir -p .cache
if [ ! -f .cache/docker-cli ]; then
  cp "$(which docker)" .cache/docker-cli
fi

docker build "$@" \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple/}" \
  --build-arg NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmjs.org/}" \
  --build-context fast-service=../../sdk/fast-service \
  --build-context scenario-protocol=../../sdk/scenario-protocol \
  -t claude:v1 .
