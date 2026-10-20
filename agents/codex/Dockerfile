FROM python:3.12

# Pip mirror (override for China: --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/)
ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

# Docker CLI — copy statically linked binary from staged .cache
COPY .cache/docker-cli /usr/local/bin/docker
RUN chmod +x /usr/local/bin/docker

# Docker compose plugin — staged from host; builds and starts each
# TerminalBench task's sandbox
COPY .cache/docker-compose /usr/local/lib/docker/cli-plugins/docker-compose
RUN chmod +x /usr/local/lib/docker/cli-plugins/docker-compose && docker compose version

# Codex Rust CLI — pre-downloaded to .cache/codex.tar.gz
# To download manually:
#   curl -fL -o .cache/codex.tar.gz https://github.com/openai/codex/releases/download/rust-v0.133.0/codex-x86_64-unknown-linux-musl.tar.gz
COPY .cache/codex.tar.gz /tmp/codex.tar.gz
RUN tar -xzf /tmp/codex.tar.gz -C /tmp \
    && mv /tmp/codex-x86_64-unknown-linux-musl /usr/local/bin/codex \
    && chmod +x /usr/local/bin/codex \
    && rm /tmp/codex.tar.gz \
    && /usr/local/bin/codex --version

WORKDIR /workspace/fast-service
COPY --from=fast-service requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workspace/codex
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workspace/fast-service
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

WORKDIR /workspace/codex
COPY src src
COPY pyproject.toml pyproject.toml
COPY config config
RUN pip install -e .

WORKDIR /workspace/codex
