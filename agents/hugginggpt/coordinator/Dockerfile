FROM python:3.12

ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

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

# agent-commons (shared LLM client library — used by hugginggpt utils.py)
WORKDIR /workspace/agent-commons
COPY --from=agent-commons requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY --from=agent-commons src src
COPY --from=agent-commons setup.py setup.py
COPY --from=agent-commons pyproject.toml pyproject.toml
RUN pip install -e .

# hugginggpt coordinator dependencies
WORKDIR /workspace/agsbench_hugginggpt
COPY coordinator/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY coordinator/src src
COPY coordinator/scripts scripts
ENV HUGGINGGPT_HOME=/workspace/agsbench_hugginggpt
ENV PYTHONPATH=/workspace/agsbench_hugginggpt/src
WORKDIR /workspace/agsbench_hugginggpt
