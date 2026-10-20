#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

cd "$DEPLOY_DIR"
docker compose -f docker-compose.yaml --env-file ../../../.env $([ -f .env ] && echo --env-file .env) down
# Remove sandbox containers
docker ps -a --filter "name=minisweagent" -q | xargs -r docker rm -f 2>/dev/null || true
source ../_lib/archive_data.sh && make_data_readable "$(basename "$PWD")"
echo "Cleanup complete."
