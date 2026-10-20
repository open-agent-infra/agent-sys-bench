#!/bin/bash
set -e
cd "$(dirname "$0")/.."
GRADER_TOKEN=unused docker compose -f docker-compose.yaml --env-file ../../../.env $([ -f .env ] && echo --env-file .env) down
source ../_lib/archive_data.sh && make_data_readable "$(basename "$PWD")"
echo "Cleanup complete."
