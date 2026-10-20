# China Mirror Configuration

When building or pulling images from within mainland China, public registries and package indexes may be slow or blocked. Use the following mirrors.

## pip (PyPI)

Pass the mirror URL as a Docker build arg:

```bash
docker build --build-arg PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ -t <image> .
```

Or set it in your Dockerfile:

```dockerfile
ARG PIP_INDEX_URL=https://pypi.org/simple/
RUN pip install --index-url ${PIP_INDEX_URL} -r requirements.txt
```

## npm

Pass the registry as a Docker build arg:

```bash
docker build --build-arg NPM_REGISTRY=https://registry.npmmirror.com -t <image> .
```

## HuggingFace (datasets, models)

The SWE-bench evaluation harness (`4.evaluate.sh` in swe-mini-swe-agent)
downloads dataset metadata from `huggingface.co`. If that domain is blocked or
slow from your region, point `huggingface_hub` at a mirror by setting
`HF_ENDPOINT`:

```bash
# The 6.evaluate.sh scripts default to this mirror:
export HF_ENDPOINT=https://hf-mirror.com
bash agents/mini-swe-agent/scripts/6.evaluate.sh

# Or override per-invocation:
HF_ENDPOINT=https://huggingface.co bash agents/mini-swe-agent/scripts/6.evaluate.sh
```

Without `HF_ENDPOINT`, `huggingface_hub` falls back to `s3.amazonaws.com` for
dataset file downloads, which is blocked in many regions and causes a long
hang followed by failure.

## Docker Hub Images

Public Docker Hub access is often slow or rate-limited from within China. Use a registry mirror by prefixing the image name. The mirrors below are community-run and may rotate; if one fails or stalls mid-pull, retry with another.

| Mirror | Prefix | Notes |
|--------|--------|-------|
| 1ms.run | `docker.1ms.run/` | Redirect target of `hub.rat.dev`; generally reliable |
| rat.dev | `hub.rat.dev/` | Redirects to `docker.1ms.run` |
| unsee.tech | `docker-0.unsee.tech/` | Proxies Docker Hub; subject to upstream rate limits |
| DaoCloud | `docker.m.daocloud.io/` | Allow-list based; some images rejected |
| NJU | `docker.nju.edu.cn/` | Nanjing University mirror |
| SJTU | `docker.mirrors.sjtug.sjtu.edu.cn/` | Shanghai Jiao Tong University mirror |

Example:

```bash
# Instead of:
docker pull swebench/sweb.eval.x86_64.django_1776_django-16379:latest

# Use a mirror:
docker pull docker.1ms.run/swebench/sweb.eval.x86_64.django_1776_django-16379:latest
docker tag docker.1ms.run/swebench/sweb.eval.x86_64.django_1776_django-16379:latest \
           swebench/sweb.eval.x86_64.django_1776_django-16379:latest
```

If a pull stalls on a single layer (network reset mid-stream), kill it and retry — Docker resumes from cached layers:

```bash
# Retry loop for a flaky pull
for i in 1 2 3 4 5; do
  docker pull docker.1ms.run/<image>:<tag> && break
  echo "attempt $i failed, retrying in 5s..."; sleep 5
done
```

Alternatively, configure Docker daemon-level mirroring in `/etc/docker/daemon.json`:

```json
{
  "registry-mirrors": [
    "https://docker.1ms.run",
    "https://docker-0.unsee.tech",
    "https://docker.m.daocloud.io"
  ]
}
```

Then restart Docker:

```bash
sudo systemctl restart docker
```

## SWE-bench Instance Images

SWE-bench evaluation requires pre-built instance images. Pull them through a mirror:

```bash
docker pull docker.1ms.run/swebench/sweb.eval.x86_64.django_1776_django-16379:latest
docker tag docker.1ms.run/swebench/sweb.eval.x86_64.django_1776_django-16379:latest \
           swebench/sweb.eval.x86_64.django_1776_django-16379:latest
```

## WebArena Shopping Admin Image

The WebAgent's default WebArena verified task requires the pre-built Magento shopping-admin image. The `:latest` tag has been removed from upstream Docker Hub — use the pinned `:0.1.0` tag instead. Pull it through a mirror and re-tag to the canonical name:

```bash
# Pull the pinned tag through a mirror (try alternatives if one stalls — see above)
docker pull docker.1ms.run/am1n3e/webarena-verified-shopping_admin:0.1.0
docker tag docker.1ms.run/am1n3e/webarena-verified-shopping_admin:0.1.0 \
           am1n3e/webarena-verified-shopping_admin:latest
```

This image is ~3 GB across 7 layers. If a pull stalls on the final layer, kill and retry — Docker resumes from cached layers.

After the shopping-admin container starts, fix the Magento base URL (see `agents/webagent/README.md` for the `docker exec` commands).

If the shopping-admin image is genuinely unavailable (e.g., upstream removed it), WebAgent can still be smoke-tested against MiniWoB++ tasks, which are bundled inside the `webagent:v1` image and need no external server. Create a manifest under `.cache/` with one line per task:

```
/workspace/webagent/.cache/webarena_verified_data_jsons/miniwob_click.json
```

And a corresponding JSON entry:

```json
{"instance_id":"miniwob-click-001","task_id":"browsergym/miniwob.click-button","seed":1}
```

Then start the stack without `shopping-admin`:

```bash
docker compose -f benchmark/deployment/webagent-compose.yaml up -d \
  browsergym-service browsergym-loadgen cadvisor prometheus grafana
```

This is a fallback for environment validation only — the default benchmark uses the WebArena verified task.

## Playwright / Chromium (webagent)

The webagent Dockerfile downloads Chromium via `playwright install chromium`.
The default CDN (`cdn.playwright.dev`) is blocked in some regions. The
Dockerfile sets `PLAYWRIGHT_DOWNLOAD_HOST=https://npmmirror.com/mirrors/playwright`
to use the npmmirror.com mirror.

If the mirror is also unavailable, pre-stage the browser binary:
```bash
# On a machine with network access:
pip install playwright
playwright install chromium
# Copy the browser cache into the agent's .cache/:
cp -r ~/.cache/ms-playwright/ agents/webagent/.cache/ms-playwright/
# Then add to Dockerfile before the playwright install step:
# COPY .cache/ms-playwright /root/.cache/ms-playwright
```
