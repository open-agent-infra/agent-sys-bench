#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
AGSBENCH_ENV="${AGENT_DIR}/../../.env"
[ -f "$AGSBENCH_ENV" ] && set -a && . "$AGSBENCH_ENV" && set +a

cd "$AGENT_DIR"

# Stage NLTK data (punkt_tab, averaged_perceptron_tagger_eng) into .cache/.
# The unstructured library calls nltk.sent_tokenize at runtime, which needs
# these data files. Without them, unstructured tries to download from
# utic-public-cf.s3.amazonaws.com — that source is blocked in many regions
# and causes a runtime hang. Baking the data into the image avoids the
# runtime download entirely.
NLTK_CACHE="${AGENT_DIR}/.cache/nltk_data"
if [ ! -f "${NLTK_CACHE}/tokenizers/punkt_tab" ] || [ ! -f "${NLTK_CACHE}/taggers/averaged_perceptron_tagger_eng" ]; then
  echo "Staging NLTK data (punkt_tab, averaged_perceptron_tagger_eng)..."
  mkdir -p "${NLTK_CACHE}/tokenizers" "${NLTK_CACHE}/taggers"
  BASE=https://gcore.jsdelivr.net/gh/nltk/nltk_data@gh-pages/packages
  for pair in "tokenizers/punkt_tab" "taggers/averaged_perceptron_tagger_eng"; do
    out="/tmp/$(basename ${pair}).zip"
    for i in 1 2 3 4 5; do
      curl -fsSL --max-time 90 -o "$out" "${BASE}/${pair}.zip" && break
      echo "  attempt $i failed, retrying in 3s..." >&2; sleep 3
    done
    python3 -c "import zipfile,sys; zipfile.ZipFile('$out').extractall('${NLTK_CACHE}/$(dirname ${pair})')" \
      || docker run --rm -v /tmp:/tmp -v "${NLTK_CACHE}:${NLTK_CACHE}" python:3.12 \
           python -c "import zipfile; zipfile.ZipFile('$out').extractall('${NLTK_CACHE}/$(dirname ${pair})')"
    rm -f "$out"
  done
  echo "  NLTK data staged at ${NLTK_CACHE}"
fi

docker build "$@" \
  --build-arg PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple/}" \
  --build-context fast-service=../../sdk/fast-service \
  --build-context scenario-protocol=../../sdk/scenario-protocol \
  -t rag:v1 .
