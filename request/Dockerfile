##############################################################################
# Unified request image for ALL scenarios.
#
# Contains: python:3.12-slim + fast-service + scenario-protocol + request/
# No agent code. ~370 MB.
#
# Build from repo root:
#   docker build -t agsbench-request:v1 -f request/Dockerfile .
#
# Run:
#   docker run --rm --network <net> agsbench-request:v1 \
#       --scenario swe \
#       --endpoint http://mini-sweagent-service:1801 \
#       --bm_file_path /data/manifest.txt
##############################################################################
FROM python:3.12-slim

# git: prepare_data fetches pinned benchmark repos (pinchbench, WildClawBench)
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

# fast-service (for FastServiceBenchmark scheduling + monitor)
WORKDIR /workspace/sdk/fast-service
COPY sdk/fast-service/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY sdk/fast-service/src src
COPY sdk/fast-service/pyproject.toml pyproject.toml
COPY sdk/fast-service/setup.py setup.py
RUN pip install -e .

# scenario-protocol (wire protocol + scenario interface definitions)
WORKDIR /workspace/sdk/scenario-protocol
COPY sdk/scenario-protocol/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY sdk/scenario-protocol/pyproject.toml pyproject.toml
COPY sdk/scenario-protocol/src src
RUN pip install -e .

# request package (workload logic — manifest parsing, request construction)
WORKDIR /workspace/request
COPY request/pyproject.toml pyproject.toml
COPY request/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY request/src src
RUN pip install -e .

ENV PYTHONPATH=/workspace

WORKDIR /workspace
ENTRYPOINT ["python", "-m", "agsbench_request.loadgen"]
