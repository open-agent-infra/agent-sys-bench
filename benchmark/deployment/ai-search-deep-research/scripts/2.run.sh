#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source ../_lib/archive_data.sh && archive_previous_run "$(basename "$PWD")"
docker compose -f docker-compose.yaml --env-file ../../../.env $([ -f ../../../agents/deep-research/config/.env ] && echo --env-file ../../../agents/deep-research/config/.env) $([ -f .env ] && echo --env-file .env) up -d
echo "Logs: docker logs -f deep-research-benchmarking-loadgen-1"
echo "Cleanup: bash scripts/3.cleanup.sh"
