#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
AGSBENCH_ENV="${AGENT_DIR}/../../.env"
[ -f "$AGSBENCH_ENV" ] && set -a && . "$AGSBENCH_ENV" && set +a

cd "$AGENT_DIR"

echo "=== Building hugginggpt coordinator image ==="
bash scripts/build_coordinator.sh "$@"

echo "=== Building hugginggpt models_server image ==="
bash scripts/build_models_server.sh "$@"

echo "=== Both images built ==="
docker images --format "{{.Repository}}:{{.Tag}} {{.Size}}" | grep hugginggpt
