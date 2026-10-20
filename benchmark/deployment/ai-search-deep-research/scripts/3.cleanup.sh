#!/bin/bash
set -e
cd "$(dirname "$0")/.."
docker compose -f docker-compose.yaml --env-file ../../../.env $([ -f .env ] && echo --env-file .env) down -v
source ../_lib/archive_data.sh && make_data_readable "$(basename "$PWD")"
echo "Cleanup complete."
