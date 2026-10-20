#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

cd "$DEPLOY_DIR"
docker compose -f docker-compose.yaml --env-file ../../../.env $([ -f .env ] && echo --env-file .env) down
# Leftover per-task TerminalBench sandboxes (compose projects codex-tb-*)
# and their per-trial images, e.g. after an interrupted run.
for p in $(docker compose ls -a -q --filter name=codex-tb- 2>/dev/null); do
  docker compose -p "$p" down --volumes --remove-orphans --rmi all >/dev/null 2>&1 || true
done
docker images -q --filter "reference=codex-tb-img-*" | xargs -r docker rmi -f >/dev/null 2>&1 || true
source ../_lib/archive_data.sh && make_data_readable "$(basename "$PWD")"
echo "Cleanup complete."
