FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# Base image has Python 3.11 + torch 2.5.1 + CUDA 12.4 + cuDNN 9 pre-installed.
# This avoids downloading 2+ GB of CUDA packages during pip install.

# Pip mirror (override for China: --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/)
ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

# Docker CLI — staged from host (Dockerfile COPYs it into the image)
COPY .cache/docker-cli /usr/local/bin/docker
RUN chmod +x /usr/local/bin/docker

# fast-service (provided via BuildKit --build-context)
WORKDIR /workspace/fast-service
COPY --from=fast-service requirements.txt requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends gcc linux-libc-dev && rm -rf /var/lib/apt/lists/* && pip install -r requirements.txt

# scenario-protocol (wire protocol + scenario interface definitions)
WORKDIR /workspace/scenario-protocol
COPY --from=scenario-protocol pyproject.toml pyproject.toml
COPY --from=scenario-protocol requirements.txt requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends gcc linux-libc-dev && rm -rf /var/lib/apt/lists/* && pip install -r requirements.txt
COPY --from=scenario-protocol src src
RUN pip install -e .

# gui-agent dependencies (desktop-env pulls in easyocr/torch — torch already in base)
WORKDIR /workspace/agsbench_gui_agent
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools wheel
RUN apt-get update && apt-get install -y --no-install-recommends gcc linux-libc-dev && rm -rf /var/lib/apt/lists/* && pip install -r requirements.txt

# Overlay desktop_env + mm_agents from the OSWorld commit staged by
# build_image.sh. PyPI desktop-env 1.0.2 lacks mm_agents.kimi/owl15_agent, and
# the repo's own package (osworld, requires-python>=3.12) can't be pip-installed
# here, so desktop-env 1.0.2 supplies the py3.11 dependency set and we copy the
# sources over it. loguru is needed by mm_agents but not pulled by desktop-env.
COPY .cache/osworld.tar.gz /tmp/osworld.tar.gz
RUN pip install loguru && mkdir /tmp/osworld && tar -xzf /tmp/osworld.tar.gz -C /tmp/osworld --strip-components=1 \
    && SP=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
    && rm -rf "$SP/desktop_env" "$SP/mm_agents" \
    && cp -r /tmp/osworld/desktop_env /tmp/osworld/mm_agents "$SP/" \
    && rm -rf /tmp/osworld /tmp/osworld.tar.gz \
    && python -c "import desktop_env.desktop_env, mm_agents.kimi.kimi_agent, mm_agents.owl15_agent"

# Install fast-service (with 3.11 compat patch)
WORKDIR /workspace/fast-service
COPY --from=fast-service src src
COPY --from=fast-service pyproject.toml pyproject.toml
COPY --from=fast-service setup.py setup.py
COPY patches/fast-service-patch.py /tmp/patch.py
RUN python /tmp/patch.py
RUN pip install -e .

# Install gui-agent package
WORKDIR /workspace/agsbench_gui_agent
COPY src src
COPY pyproject.toml pyproject.toml
COPY config config
RUN pip install -e .

# Patch OSWorld's docker provider for container-to-host communication
COPY patches/osworld-provider-patch.py /tmp/osworld-patch.py
RUN python /tmp/osworld-patch.py

WORKDIR /workspace/agsbench_gui_agent
