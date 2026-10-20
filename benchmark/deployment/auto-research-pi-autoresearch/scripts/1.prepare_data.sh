#!/bin/bash
# 1. Write the competition manifest (.cache/pi_autoresearch_data.txt).
#    Default: spooky-author-identification. Options: --competitions a b c,
#    --lite (the 22 MLE-bench Lite competitions, ~158 GB), --force.
# 2. Download + split each competition with MLE-bench (grader image) into
#    .cache/mlebench/data (incl. private answers), and hard-link the
#    agent-visible part into .cache/mlebench/public.
#
# Needs a Kaggle API token in KAGGLE_API_TOKEN (or ~/.kaggle/kaggle.json),
# and the competition rules accepted on kaggle.com for that account.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"
CACHE_DIR="${DEPLOY_DIR}/.cache"
mkdir -p "${CACHE_DIR}/mlebench"

docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "${CACHE_DIR}:/output" \
  --entrypoint python \
  agsbench-request:v1 -m agsbench_request.prepare_data \
    --scenario auto-research \
    --output_dir /output \
    "$@"

KAGGLE_MOUNT=()
[ -f "${HOME}/.kaggle/kaggle.json" ] && KAGGLE_MOUNT=(-v "${HOME}/.kaggle/kaggle.json:/tmp/.kaggle/kaggle.json:ro")
# -w /tmp: mlebench keeps a diskcache in ./cache.
docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -e KAGGLE_API_TOKEN -w /tmp \
  "${KAGGLE_MOUNT[@]}" \
  -v "${CACHE_DIR}:/cache" \
  pi-autoresearch-grader:v1 python /opt/grader/prepare.py \
    --manifest /cache/pi_autoresearch_data.txt \
    --data_dir /cache/mlebench/data \
    --public_dir /cache/mlebench/public

echo "Data prepared at ${CACHE_DIR}/"
