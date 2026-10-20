FROM python:3.12

# Pip mirror (override for China: --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/)
ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip config set global.index-url ${PIP_INDEX_URL}

# fast-service (provided via BuildKit --build-context)
WORKDIR /workspace/fast-service
COPY --from=fast-service requirements.txt requirements.txt
RUN pip install 'setuptools<81' && pip install -r requirements.txt

# install dependencies for rag
WORKDIR /workspace/rag
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install grpcio grpcio-tools

# Pre-stage NLTK data so unstructured's sent_tokenize does not attempt
# runtime downloads (the default S3 source is unreliable from many regions).
# The data is shipped in .cache/nltk_data (populated by scripts/build_image.sh
# or by hand) and COPYd in here as a fixed image layer.
COPY .cache/nltk_data /usr/local/share/nltk_data
RUN python -c "import nltk; nltk.find('tokenizers/punkt_tab'); nltk.find('taggers/averaged_perceptron_tagger_eng')"

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
WORKDIR /workspace/rag
COPY src src
COPY scripts scripts
COPY config config

# add naive_rag to PYTHONPATH
ENV PYTHONPATH=/workspace/rag/src

# set workdir and default command
WORKDIR /workspace/rag/scripts
CMD ["python", "launch_services.py"]