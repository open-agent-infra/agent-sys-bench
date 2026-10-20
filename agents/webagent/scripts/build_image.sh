#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGENT_DIR="$(dirname "$SCRIPT_DIR")"
AGSBENCH_ENV="${AGENT_DIR}/../../.env"
[ -f "$AGSBENCH_ENV" ] && set -a && . "$AGSBENCH_ENV" && set +a

cd "$AGENT_DIR"

# Stage MiniWoB++ HTML. Set GITHUB_PROXY (e.g. https://ghfast.top/) if
# github.com is unreachable. Clones into a temp dir and retries, since a
# stalled clone would otherwise leave a partial dir that skips this step.
if [ ! -d .cache/miniwob-plusplus ]; then
  mkdir -p .cache
  MINIWOB_REPO="${GITHUB_PROXY:-}https://github.com/Farama-Foundation/miniwob-plusplus.git"
  for i in 1 2 3 4 5 6 7 8; do
    rm -rf .cache/miniwob-plusplus.tmp
    timeout 600 git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 \
      clone --depth 1 "$MINIWOB_REPO" .cache/miniwob-plusplus.tmp \
      && mv .cache/miniwob-plusplus.tmp .cache/miniwob-plusplus && break
    echo "  clone attempt $i failed, retrying..." >&2; sleep $((i * 5))
  done
  [ -d .cache/miniwob-plusplus ] || { echo "Failed to clone MiniWoB++" >&2; exit 1; }
fi

# Stage NLTK data (punkt_tab) into .cache/. The browsergym.webarena and
# browsergym.webarena_verified packages call nltk.word_tokenize during
# evaluation, which needs punkt_tab. Without it, browsergym tries to
# download at import time — the default S3 source is blocked in many
# regions and causes a 16-minute hang. Baking the data into the image
# avoids the runtime download entirely.
NLTK_CACHE="${AGENT_DIR}/.cache/nltk_data"
if [ ! -f "${NLTK_CACHE}/tokenizers/punkt_tab" ]; then
  echo "Staging NLTK data (punkt_tab)..."
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
  --build-arg NPM_REGISTRY="${NPM_REGISTRY:-https://registry.npmjs.org/}" \
  --build-context fast-service=../../sdk/fast-service \
  --build-context scenario-protocol=../../sdk/scenario-protocol \
  -t webagent:v1 .
