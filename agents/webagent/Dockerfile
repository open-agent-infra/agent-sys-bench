FROM python:3.12

# Use aliyun apt mirror for China network
ARG APT_MIRROR=mirrors.aliyun.com
RUN sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/* 2>/dev/null; \
    sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list 2>/dev/null; true

# Pip mirror (override for China: --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/)
ARG PIP_INDEX_URL=https://pypi.org/simple/
ARG PIP_EXTRA_INDEX_URL=
RUN pip config set global.index-url ${PIP_INDEX_URL}
RUN if [ -n "${PIP_EXTRA_INDEX_URL}" ]; then pip config set global.extra-index-url ${PIP_EXTRA_INDEX_URL}; fi

# Playwright + Chromium needs a handful of system libs. Use Playwright's
# own dependency installer to grab them in one go.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git make ca-certificates curl wget tinyproxy socat \
    && rm -rf /var/lib/apt/lists/*

# Tinyproxy config for Chromium workaround (kernel network namespace issue).
RUN printf 'Port 8899\nListen 127.0.0.1\nAllow 127.0.0.1\nMaxClients 100\nLogLevel Error\n' > /etc/tinyproxy.conf

# (pip mirror already configured above)

WORKDIR /workspace/fast-service
COPY --from=fast-service requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /workspace/webagent
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Chromium system dependencies (playwright install chromium handles the binary).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
        libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 libasound2 \
        fonts-liberation fonts-unifont \
    && rm -rf /var/lib/apt/lists/* \
    && PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright playwright install chromium

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

# Wrapper package.
WORKDIR /workspace/webagent
COPY src src
COPY pyproject.toml pyproject.toml
COPY config config
RUN pip install -e .

# Force-upgrade playwright for Chromium 148+ (64KB page support on aarch64+64k kernels).
# browsergym-core pins playwright==1.44 (Chromium 125) which crashes on 64KB-page kernels.
# The chromium download may fail on restricted networks — make it non-fatal
# since the first install (line 40) already provides a working browser.
RUN pip install --no-deps 'playwright>=1.50' && \
    (PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright playwright install chromium || \
     echo 'WARN: Chromium upgrade failed, using browser from first install')

# Stage MiniWoB++ static HTML so file:// URLs resolve inside the container.
# Pinned commit per upstream's Makefile.
COPY .cache/miniwob-plusplus /opt/miniwob-plusplus

# Pre-stage NLTK data (punkt_tab, averaged_perceptron_tagger_eng) so that
# `browsergym.webarena` / `browsergym.webarena_verified` imports don't attempt
# runtime nltk.download() — the default S3 source is unreliable from many
# regions and the resulting 16-min hang breaks the agent loop.
COPY .cache/nltk_data /root/nltk_data

ENV MINIWOB_URL=file:///opt/miniwob-plusplus/miniwob/html/miniwob/

# Patch browsergym to read WEBAGENT_CHROMIUM_ARGS env var for extra Chromium flags.
RUN BGENV=$(python3 -c "import browsergym.core.env; print(browsergym.core.env.__file__)") && \
    sed -i '/args = \[arg for arg in args if arg is not None\]/a\        # --- webagent patch: inject extra chromium args from env ---\n        import os as _os, shlex as _shlex\n        _extra = _os.environ.get("WEBAGENT_CHROMIUM_ARGS", "--no-sandbox")\n        if _extra:\n            args.extend(_shlex.split(_extra))' "$BGENV"

WORKDIR /workspace/webagent
