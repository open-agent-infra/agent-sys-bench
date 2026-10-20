#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQUEST_DIR="$(dirname "$SCRIPT_DIR")"
AGSBENCH_ENV="${REQUEST_DIR}/../.env"
[ -f "$AGSBENCH_ENV" ] && set -a && . "$AGSBENCH_ENV" && set +a

cd "$REQUEST_DIR/.."

docker build "$@" \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple/}" \
  -t agsbench-request:v1 -f request/Dockerfile .
