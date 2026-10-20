FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/hugginggpt
COPY models_server/requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY models_server/src /workspace/hugginggpt/src/models_server

ENV PYTHONPATH=/workspace/hugginggpt/src/models_server
ENV HF_HOME=/mnt/data/huggingface
WORKDIR /workspace/hugginggpt/src/models_server
CMD ["python", "models_server.py"]
