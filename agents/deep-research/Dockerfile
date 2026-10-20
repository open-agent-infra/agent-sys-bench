FROM python:3.12

# Pip mirror (override for China: --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/)
ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

# fast-service (provided via BuildKit --build-context)
WORKDIR /workspace/fast-service
COPY --from=fast-service requirements.txt requirements.txt
RUN pip install -r requirements.txt

# install dependencies for legoAgent
WORKDIR /workspace/legoAgent
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install grpcio grpcio-tools

# install fast-service
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

# copy necessary code to docker image
WORKDIR /workspace/legoAgent
COPY src src
COPY scripts scripts

# add legoAgent to PYTHONPATH
ENV PYTHONPATH=/workspace/legoAgent/src

# set workdir and default command
WORKDIR /workspace/legoAgent/scripts
CMD ["python", "launch_server.py"]
