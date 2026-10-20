# MLE-bench grading sidecar + dataset preparer for the auto-research scenario.
# Holds the private answers; the agent image never sees them.
FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

RUN apt-get update && apt-get install -y --no-install-recommends git git-lfs curl unzip p7zip-full && \
    rm -rf /var/lib/apt/lists/*

# MLE-bench at a pinned commit. Competition metadata (leaderboards used for
# the medal thresholds, checksums) is stored in git LFS. Set GITHUB_PROXY
# (e.g. https://ghfast.top/) if github.com is unreachable.
ARG MLEBENCH_COMMIT=507f92e1138bb6e40dac5c6ee7a6758e6424bf97
ARG GITHUB_PROXY=
RUN git lfs install --system && \
    git init -q /opt/mle-bench && cd /opt/mle-bench && \
    git remote add origin ${GITHUB_PROXY}https://github.com/openai/mle-bench.git && \
    GIT_LFS_SKIP_SMUDGE=1 git fetch -q --depth 1 origin ${MLEBENCH_COMMIT} && \
    GIT_LFS_SKIP_SMUDGE=1 git -c advice.detachedHead=false checkout -q FETCH_HEAD && \
    git lfs pull --include "mlebench/competitions/**" --exclude "mlebench/competitions/*/top_solutions/**" && \
    rm -rf .git/lfs

RUN pip install --no-cache-dir -e /opt/mle-bench
# MLE-bench pins kaggle<1.7, which only reads ~/.kaggle/kaggle.json. Newer
# kaggle also accepts the KGAT_ access tokens via KAGGLE_API_TOKEN.
ARG KAGGLE_VERSION=2.2.4
RUN pip install --no-cache-dir kaggle==${KAGGLE_VERSION} flask==3.0.3

COPY grading_server.py prepare.py /opt/grader/
WORKDIR /opt/grader
ENV MLEBENCH_DATA_DIR=/data
CMD ["python", "/opt/grader/grading_server.py"]
