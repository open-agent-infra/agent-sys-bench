FROM python:3.12

# Use aliyun apt mirror for China network (consistent with webagent Dockerfile)
ARG APT_MIRROR=mirrors.aliyun.com
RUN sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/* 2>/dev/null; \
    sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list 2>/dev/null; true

# Pip mirror (override for China: --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/)
ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

# Docker CLI — copy statically linked binary from staged .cache
COPY .cache/docker-cli /usr/local/bin/docker
RUN chmod +x /usr/local/bin/docker

# Node.js 22 LTS (for the openclaw CLI binary)
# npm registry (override for China: --build-arg NPM_REGISTRY=https://registry.npmmirror.com)
ARG NPM_REGISTRY=https://registry.npmjs.org/
# Pinned: newer releases change the on-disk session/transcript layout that
# run_task.collect_outputs reads. Bump deliberately and re-validate.
ARG OPENCLAW_VERSION=2026.9.2
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs && \
    npm config set registry ${NPM_REGISTRY} && \
    npm install -g openclaw@${OPENCLAW_VERSION}

# fast-service (provided via BuildKit --build-context)
WORKDIR /workspace/fast-service
COPY --from=fast-service requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY --from=fast-service src src
COPY --from=fast-service pyproject.toml pyproject.toml
COPY --from=fast-service setup.py setup.py
RUN pip install -e .

# scenario-protocol (wire protocol + scenario interface definitions)
WORKDIR /workspace/scenario-protocol
COPY --from=scenario-protocol pyproject.toml pyproject.toml
COPY --from=scenario-protocol requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY --from=scenario-protocol src src
RUN pip install -e .

# openclaw Python package
WORKDIR /workspace/openclaw
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY pyproject.toml pyproject.toml
COPY src src
RUN pip install -e .
COPY scripts scripts
COPY config config
COPY vsandbox-plugin vsandbox-plugin

# Install vsandbox plugin for HTTP-based sandbox backend
RUN openclaw plugins install --link ./vsandbox-plugin --dangerously-force-unsafe-install 2>&1 || true

WORKDIR /workspace/openclaw
