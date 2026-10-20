#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source ../_lib/archive_data.sh && archive_previous_run "$(basename "$PWD")"
docker compose -f docker-compose.yaml --env-file ../../../.env $([ -f ../../../agents/claude/config/.env ] && echo --env-file ../../../agents/claude/config/.env) $([ -f .env ] && echo --env-file .env) up -d
echo "Logs: docker logs -f claude-loadgen"
echo "Cleanup: bash scripts/3.cleanup.sh"
