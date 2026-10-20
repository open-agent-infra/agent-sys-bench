FROM python:3.12

# Pip mirror (override for China: --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/)
ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

# Install Docker CLI from host binary (statically linked)
COPY .cache/docker-cli /usr/local/bin/docker
RUN chmod +x /usr/local/bin/docker

# fast-service is provided via BuildKit --build-context (see scripts/build_image.sh)
WORKDIR /workspace/fast-service
COPY --from=fast-service requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Wrapper-package dependencies (mini-swe-agent, swebench, datasets, ...)
WORKDIR /workspace/agsbench_mini_swe_agent
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workspace/fast-service
COPY --from=fast-service src src
COPY --from=fast-service pyproject.toml pyproject.toml
COPY --from=fast-service setup.py setup.py
RUN pip install -e .

# scenario-protocol (wire protocol + scenario interface definitions)
# Provided via BuildKit --build-context (see scripts/build_image.sh)
WORKDIR /workspace/scenario-protocol
COPY --from=scenario-protocol pyproject.toml pyproject.toml
COPY --from=scenario-protocol requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY --from=scenario-protocol src src
RUN pip install -e .

# Install the wrapper package
WORKDIR /workspace/agsbench_mini_swe_agent
COPY src src
COPY pyproject.toml pyproject.toml
COPY config config
RUN pip install -e .

# Quiet mini-swe-agent's startup banner so it doesn't pollute fast-service logs
ENV MSWEA_SILENT_STARTUP=1

WORKDIR /workspace/agsbench_mini_swe_agent
