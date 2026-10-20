#!/bin/bash
set -e
DEPLOY_DIR="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "${DEPLOY_DIR}/.cache"
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v "${DEPLOY_DIR}/.cache:/output" -e HF_TOKEN -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  --entrypoint python agsbench-request:v1 -m agsbench_request.prepare_data --scenario ai-search --output_dir /output "$@"
echo "Data prepared at ${DEPLOY_DIR}/.cache/"
