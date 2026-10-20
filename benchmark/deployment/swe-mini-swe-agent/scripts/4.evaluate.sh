#!/bin/env sh

# Score trajectories/preds.json against SWE-bench using the upstream
# swebench harness. Independent of the mini-sweagent service stack, so
# this can run before OR after 5.cleanup.sh. Requires the host docker
# daemon (the harness spawns one eval container per instance).
#
# Env knobs (all optional):
#   DATASET_NAME      HF dataset name (default: princeton-nlp/SWE-bench)
#                     The shipped sample task (django__django-16379) is in
#                     the full set. For the official 500-instance verified
#                     subset, set DATASET_NAME=princeton-nlp/SWE-Bench_Verified
#   SPLIT             dataset split   (default: test)
#   PREDICTIONS_PATH  path inside container (default: trajectories/preds.json)
#   RUN_ID            tag for log/report dirs (default: mini-swe-agent-<utc>)
#   MAX_WORKERS       parallel eval containers (default: 2)
#   HF_ENDPOINT       HuggingFace mirror (default: https://hf-mirror.com)
#                     Set to https://huggingface.co if you have direct access.
# Extra args passed through, e.g.:
#   bash scripts/6.evaluate.sh -i astropy__astropy-12907

set -e

DATASET_NAME=${DATASET_NAME:-princeton-nlp/SWE-bench}
SPLIT=${SPLIT:-test}
PREDICTIONS_PATH=${PREDICTIONS_PATH:-trajectories/preds.json}
RUN_ID=${RUN_ID:-mini-swe-agent-$(date -u +%Y%m%d-%H%M%S)}
MAX_WORKERS=${MAX_WORKERS:-2}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

mkdir -p logs

# swebench's harness writes its top-level summary report to CWD as
# "<model>.<run_id>.json" and ignores --report_dir for that file; the
# per-instance test logs go to <CWD>/logs/run_evaluation/<run_id>/...
# We bind-mount trajectories/ into the container and chdir into it so
# both artifacts land on the host.
#
# HF_ENDPOINT is passed into the container so huggingface_hub downloads
# the dataset metadata through a mirror (default hf-mirror.com). Without
# this, huggingface_hub tries s3.amazonaws.com which is blocked in many
# regions, causing a 16-minute hang followed by failure.
docker run --rm \
    -e HF_ENDPOINT="$HF_ENDPOINT" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v $(pwd)/trajectories:/workspace/agsbench_mini_swe_agent/trajectories \
    -v $(pwd)/.cache:/workspace/agsbench_mini_swe_agent/.cache \
    -v $(pwd)/logs:/workspace/agsbench_mini_swe_agent/trajectories/logs \
    --workdir /workspace/agsbench_mini_swe_agent/trajectories \
    mini-swe-agent:v1 \
    python -m swebench.harness.run_evaluation \
        --dataset_name "$DATASET_NAME" \
        --split "$SPLIT" \
        --predictions_path "/workspace/agsbench_mini_swe_agent/$PREDICTIONS_PATH" \
        --run_id "$RUN_ID" \
        --max_workers "$MAX_WORKERS" \
        "$@"

echo
echo "Summary report:   trajectories/*.${RUN_ID}.json"
echo "Per-instance logs: logs/run_evaluation/${RUN_ID}/"
