#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
AGSBENCH_HOME="$(cd "${DEPLOY_DIR}/../.." && pwd)"

# Prepare SWE-bench data using the unified prepare_data entry point
mkdir -p "${DEPLOY_DIR}/.cache"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "${DEPLOY_DIR}/.cache:/output" \
  -e HF_TOKEN -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  --entrypoint python \
  agsbench-request:v1 -m agsbench_request.prepare_data \
    --scenario swe \
    --output_dir /output \
    "$@"

echo "Data prepared at ${DEPLOY_DIR}/.cache/"
