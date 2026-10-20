#!/usr/bin/env bash
# Build the openclaw Docker sandbox image (openclaw-sandbox:bookworm-slim).
# Safe to re-run — skips if the image already exists.
set -euo pipefail

IMAGE="${1:-openclaw-sandbox:bookworm-slim}"

if docker image inspect "$IMAGE" > /dev/null 2>&1; then
    echo "Image $IMAGE already exists. Skipping build."
    exit 0
fi

echo "Building $IMAGE ..."
docker build "$@" -t "$IMAGE" - <<'DOCKERFILE'
FROM debian:bookworm-slim
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
  bash ca-certificates curl git jq python3 ripgrep \
  && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /bin/bash sandbox
USER sandbox
WORKDIR /home/sandbox
CMD ["sleep", "infinity"]
DOCKERFILE

echo "Done: $IMAGE"
