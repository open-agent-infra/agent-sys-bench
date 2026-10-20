#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY_DIR="$(dirname "$SCRIPT_DIR")"

cd "$DEPLOY_DIR"
source ../_lib/archive_data.sh && archive_previous_run "$(basename "$PWD")"
docker compose -f docker-compose.yaml --env-file ../../../.env $([ -f ../../../agents/codex/config/.env ] && echo --env-file ../../../agents/codex/config/.env) $([ -f .env ] && echo --env-file .env) up -d

echo ""
echo "Monitoring:"
echo "  Grafana:    http://localhost:${CODEX_GRAFANA_PORT:-3000}"
echo "  Prometheus: http://localhost:${CODEX_PROMETHEUS_PORT:-9090}"
echo ""
echo "Logs: docker logs -f codex-loadgen"
echo "Cleanup: bash scripts/3.cleanup.sh"
