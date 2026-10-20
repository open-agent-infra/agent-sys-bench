#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
AGSBENCH_ENV="${AGENT_DIR}/../../.env"
[ -f "$AGSBENCH_ENV" ] && set -a && . "$AGSBENCH_ENV" && set +a

cd "$AGENT_DIR"

# Stage .cache dependencies
mkdir -p .cache

# Stage docker-cli from host (Dockerfile COPYs it into the image)
if [ ! -f .cache/docker-cli ]; then
  cp "$(which docker)" .cache/docker-cli
fi

# Stage the docker compose plugin from host: the TerminalBench runner calls
# `docker compose` to build and start each task's sandbox.
if [ ! -f .cache/docker-compose ]; then
  for d in "${DOCKER_CONFIG:-$HOME/.docker}/cli-plugins" /usr/local/lib/docker/cli-plugins \
           /usr/local/libexec/docker/cli-plugins /usr/lib/docker/cli-plugins \
           /usr/libexec/docker/cli-plugins; do
    if [ -x "$d/docker-compose" ]; then cp "$d/docker-compose" .cache/docker-compose; break; fi
  done
  if [ ! -f .cache/docker-compose ]; then
    echo "ERROR: docker compose plugin not found on host (needed for TerminalBench tasks)" >&2
    exit 1
  fi
fi

# Stage codex binary (download if not present)
if [ ! -f .cache/codex.tar.gz ]; then
  echo "Downloading codex binary..."
  curl -fL -o .cache/codex.tar.gz \
    "${GITHUB_PROXY:-}https://github.com/openai/codex/releases/download/rust-v0.133.0/codex-x86_64-unknown-linux-musl.tar.gz"
fi

docker build "$@" \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple/}" \
  --build-arg NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmjs.org/}" \
  --build-context fast-service=../../sdk/fast-service \
  --build-context scenario-protocol=../../sdk/scenario-protocol \
  -t codex:v1 .
