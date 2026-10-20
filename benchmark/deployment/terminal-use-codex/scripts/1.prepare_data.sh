#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

mkdir -p "${DEPLOY_DIR}/.cache"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "${DEPLOY_DIR}/.cache:/output" \
  --entrypoint python \
  agsbench-request:v1 -m agsbench_request.prepare_data \
    --scenario terminal-use \
    --output_dir /output \
    "$@"

echo "Data prepared at ${DEPLOY_DIR}/.cache/"
