#!/usr/bin/env bash
# Build the openclaw SSH sandbox image (openclaw-sandbox-ssh:bookworm-slim).
# Similar to build_sandbox_image.sh but adds openssh-server for the SSH
# sandbox backend used in multi-node v2 mode.
set -euo pipefail

IMAGE="${1:-openclaw-sandbox-ssh:bookworm-slim}"

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
  openssh-server sudo \
  && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /bin/bash sandbox \
  && mkdir -p /home/sandbox/.ssh \
  && chmod 700 /home/sandbox/.ssh \
  && chown sandbox:sandbox /home/sandbox/.ssh \
  && mkdir -p /run/sshd \
  && mkdir -p /workspace \
  && chown sandbox:sandbox /workspace
RUN echo "sandbox ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
WORKDIR /home/sandbox
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
DOCKERFILE

echo "Done: $IMAGE"
