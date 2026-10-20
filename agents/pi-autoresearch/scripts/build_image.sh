#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
AGSBENCH_ENV="${AGENT_DIR}/../../.env"
[ -f "$AGSBENCH_ENV" ] && set -a && . "$AGSBENCH_ENV" && set +a

cd "$AGENT_DIR"

# Stage the pi-autoresearch extension at a pinned commit. Set GITHUB_PROXY
# (e.g. https://ghfast.top/) if github.com is unreachable.
PI_AUTORESEARCH_COMMIT=04b8a50dac7efe80ffc91648bd00ed9b022e24c2
EXT=.cache/pi-autoresearch
if [ "$(git -C "$EXT" rev-parse HEAD 2>/dev/null)" != "$PI_AUTORESEARCH_COMMIT" ]; then
    echo "Fetching pi-autoresearch extension @ ${PI_AUTORESEARCH_COMMIT:0:8}..."
    mkdir -p .cache
    [ -d "$EXT/.git" ] || { rm -rf "$EXT"; git init -q "$EXT"; }
    git -C "$EXT" remote remove origin 2>/dev/null || true
    git -C "$EXT" remote add origin "${GITHUB_PROXY:-}https://github.com/davebcn87/pi-autoresearch.git"
    for i in 1 2 3 4 5 6 7 8; do
        timeout 600 git -C "$EXT" -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 \
            fetch -q --depth 1 origin "$PI_AUTORESEARCH_COMMIT" && break
        echo "  fetch attempt $i failed, retrying..." >&2; sleep $((i * 5))
    done
    git -C "$EXT" -c advice.detachedHead=false checkout -q "$PI_AUTORESEARCH_COMMIT"
fi

docker build "$@" \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple/}" \
  --build-arg NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmjs.org/}" \
  --build-context fast-service=../../sdk/fast-service \
  --build-context scenario-protocol=../../sdk/scenario-protocol \
  -t pi-autoresearch:v1 .

# MLE-bench grader / dataset preparer (holds the private answers).
docker build "$@" \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple/}" \
  --build-arg GITHUB_PROXY="${GITHUB_PROXY:-}" \
  -t pi-autoresearch-grader:v1 grader
