#!/bin/bash
set -e
cd "$(dirname "$0")/.."
source ../_lib/archive_data.sh && archive_previous_run "$(basename "$PWD")"
# Shared secret between the runner and the grader's /grade endpoint; the
# agent's environment never gets it, so it can validate but not score.
export GRADER_TOKEN="${GRADER_TOKEN:-$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
docker compose -f docker-compose.yaml --env-file ../../../.env $([ -f ../../../agents/pi-autoresearch/config/.env ] && echo --env-file ../../../agents/pi-autoresearch/config/.env) $([ -f .env ] && echo --env-file .env) up -d
echo "Logs: docker logs -f pi-autoresearch-loadgen"
echo "Cleanup: bash scripts/3.cleanup.sh"
