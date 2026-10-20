FROM python:3.12

# Pip mirror (override for China: --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/)
ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

# Install Node.js 22 LTS for pi CLI
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && \
    apt-get install -y nodejs

# Install pi coding agent CLI from npm. Pinned: config/openai-provider-patch.js
# is mounted over a file inside this package.
ARG PI_CODING_AGENT_VERSION=1.3.0
# npm registry (override for China: --build-arg NPM_REGISTRY=https://registry.npmmirror.com)
ARG NPM_REGISTRY=https://registry.npmjs.org/
RUN npm config set registry ${NPM_REGISTRY} && \
    npm install -g @earendil-works/pi-coding-agent@${PI_CODING_AGENT_VERSION} && \
    pi --version

# Install pi-autoresearch extension and skills
COPY .cache/pi-autoresearch/extensions /tmp/pi-ext
COPY .cache/pi-autoresearch/skills /tmp/pi-skills
RUN mkdir -p /root/.pi/agent/extensions /root/.pi/agent/skills && \
    cp -r /tmp/pi-ext/pi-autoresearch /root/.pi/agent/extensions/ && \
    cp -r /tmp/pi-skills/* /root/.pi/agent/skills/ && \
    rm -rf /tmp/pi-ext /tmp/pi-skills

# ---- fast-service (provided via BuildKit --build-context) ----
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

# ---- pi-autoresearch Python wrapper ----
WORKDIR /workspace/pi_autoresearch
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src src
COPY pyproject.toml pyproject.toml
COPY config config
COPY scripts scripts
RUN pip install -e .

# ---- Python env for the agent's own ML code (separate from the service) ----
# The runner puts /opt/agent-env/bin first on pi's PATH. Versions follow the
# MLE-bench agent image; torch wheels from PyPI bundle CUDA 12.1.
COPY agent-env-requirements.txt /opt/agent-env-requirements.txt
RUN python -m venv /opt/agent-env && \
    /opt/agent-env/bin/pip install --no-cache-dir -r /opt/agent-env-requirements.txt

# autoresearch commits/reverts experiments with git; projects are bind-mounted
# from the host, so ownership differs from root.
RUN git config --global user.name agsbench && \
    git config --global user.email agsbench@localhost && \
    git config --global --add safe.directory '*'

ENV PI_CODING_AGENT_DIR=/root/.pi/agent
WORKDIR /workspace/pi_autoresearch
