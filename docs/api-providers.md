# API Provider Configuration

AgentSysBench agents consume three categories of external API:

1. **LLM** — chat / responses endpoint (all 8 agents)
2. **Embedding** — vector embedding endpoint (rag, deep-research)
3. **Search** — web search endpoint (deep-research, openclaw)

The `scripts/configure-api.sh` tool configures all three in one pass. Run it
from the repo root after `scripts/setup.sh`.

## Quick Setup

### Interactive (prompts for everything)

```bash
bash scripts/configure-api.sh
```

### Non-interactive — DashScope for LLM + embedding, Firecrawl for search

```bash
bash scripts/configure-api.sh \
    --llm-preset dashscope --llm-key sk-xxx --llm-model deepseek-v4-pro \
    --search-provider firecrawl --search-key fc-xxx
```

### Non-interactive — DeepSeek for LLM, DashScope for embedding

DeepSeek does not serve embeddings, so you must provide a separate embedding
endpoint (DashScope, OpenAI, or local vLLM):

```bash
bash scripts/configure-api.sh \
    --llm-preset deepseek --llm-key sk-xxx \
    --embedding-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --embedding-key sk-yyy --embedding-model text-embedding-v3 \
    --search-provider firecrawl --search-key fc-xxx
```

### Single agent override

Configure only one agent without touching the others. This is useful when
one agent needs a different provider or model than the global default.

```bash
# Configure only deep-research: DashScope LLM + embedding + Firecrawl search
bash scripts/configure-api.sh --agent deep-research \
    --llm-preset dashscope --llm-key sk-xxx --llm-model deepseek-v4-pro \
    --search-provider firecrawl --search-key fc-xxx
```

Other common per-agent request:

```bash
# Point codex at a different LLM endpoint than the rest
bash scripts/configure-api.sh --agent codex \
    --llm-url http://10.0.0.5:8000/v1 --llm-key EMPTY --llm-model deepseek-v3

# Switch deep-research to DuckDuckGo search (no key, for smoke tests)
bash scripts/configure-api.sh --agent deep-research \
    --llm-preset dashscope --llm-key sk-xxx \
    --search-provider duckduckgo

# Give rag a separate embedding endpoint (different from its LLM)
bash scripts/configure-api.sh --agent rag \
    --llm-preset deepseek --llm-key sk-xxx \
    --embedding-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --embedding-key sk-yyy --embedding-model text-embedding-v3
```

The `--agent` flag accepts any of: `claude`, `codex`, `deep-research`,
`mini-swe-agent`, `openclaw`, `pi-autoresearch`, `rag`, `webagent`. Only the
named agent's `.env` and YAML configs are (re)generated; other agents keep
their existing configuration.

### Show current configuration

```bash
bash scripts/configure-api.sh --list
```

## Flag Reference

| Flag | Description |
|------|-------------|
| `--llm-preset dashscope\|deepseek\|openai` | LLM provider preset (sets URL + default model) |
| `--llm-url URL` | Custom LLM endpoint (alternative to preset) |
| `--llm-key KEY` | LLM API key |
| `--llm-model MODEL` | LLM model name |
| `--embedding-url URL` | Embedding endpoint (default: same as LLM) |
| `--embedding-key KEY` | Embedding API key (default: same as LLM key) |
| `--embedding-model MODEL` | Embedding model (default: provider-specific) |
| `--search-provider firecrawl\|duckduckgo\|none` | Search backend |
| `--search-key KEY` | Search API key (required for firecrawl) |
| `--agent NAME` | Configure only one agent |
| `--list` | Show current config and exit |

## LLM Protocol Requirements

Agents use three different API protocols to call LLMs:

| Protocol | Agents | Endpoint |
|----------|--------|----------|
| **Chat Completions** (`/v1/chat/completions`) | mini-swe-agent, webagent, rag, deep-research, openclaw | Standard OpenAI-compatible |
| **Responses API** (`/v1/responses`) | codex, pi-autoresearch | OpenAI's newer API format |
| **Anthropic Messages** (`/v1/messages`) | claude | Anthropic's native format |

Not all providers support all protocols:

| Provider | Chat Completions | Responses API | Anthropic Messages | Embedding |
|----------|-----------------|---------------|-------------------|-----------|
| **OpenAI** | ✅ | ✅ | ❌ | ✅ (`text-embedding-3-small`) |
| **DeepSeek** | ✅ | ❌ | ❌ | ❌ (use separate provider) |
| **DashScope** | ✅ | ✅ (Qwen models only) | ❌ | ✅ (`text-embedding-v3`) |
| **Local vLLM** | ✅ | ❌ | ❌ | ✅ (if running an embedding model) |

When a provider doesn't support a required protocol, a translation proxy is
inserted automatically.

## LLM Provider Details

### DashScope (Alibaba Cloud) — recommended for China networks

```bash
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx --llm-model deepseek-v4-pro
```

- Serves both LLM (`deepseek-v4-pro`) and embedding (`text-embedding-v3`) on
  the same endpoint with the same key.
- **Codex needs LiteLLM** — DashScope's Responses API only supports Qwen
  models, not DeepSeek models.
- **Pi needs LiteLLM** — Pi's DeepSeek native provider doesn't accept
  DashScope keys (401). Goes through LiteLLM with `PI_PROVIDER=openai`.
- Claude needs `anthropic-proxy`.

### DeepSeek

```bash
bash scripts/configure-api.sh --llm-preset deepseek --llm-key sk-xxx
```

- Chat Completions agents connect directly.
- **No embedding service** — you must pass `--embedding-url` / `--embedding-key`
  / `--embedding-model` separately (e.g., DashScope `text-embedding-v3`).
- **Codex needs LiteLLM** — DeepSeek doesn't support Responses API.
- **Pi uses DeepSeek's native provider** — no proxy needed.
- Claude needs `anthropic-proxy`.

### OpenAI

```bash
bash scripts/configure-api.sh --llm-preset openai --key sk-xxx
```

- All agents work directly — OpenAI supports both Chat Completions and
  Responses API natively.
- Serves embeddings (`text-embedding-3-small`) on the same endpoint.
- Claude still needs `anthropic-proxy`.
- **No LiteLLM needed.**

### Custom / Local vLLM

```bash
bash scripts/configure-api.sh --llm-url http://localhost:8000/v1 --llm-key EMPTY --llm-model Qwen/Qwen3-8B
```

For embedding, pass `--embedding-url` / `--embedding-model` pointing at your
vLLM embedding instance (or omit if you don't need rag/deep-research).

## Embedding API

RAG and DeepResearch require an embedding endpoint. When you use DashScope or
OpenAI as the LLM preset, the embedding defaults to the same endpoint + key
with a provider-specific model (`text-embedding-v3` for DashScope,
`text-embedding-3-small` for OpenAI).

To use a **different** embedding provider (e.g., DeepSeek for LLM but
DashScope for embedding):

```bash
bash scripts/configure-api.sh \
    --llm-preset deepseek --llm-key sk-xxx \
    --embedding-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --embedding-key sk-yyy \
    --embedding-model text-embedding-v3
```

The script patches:
- **RAG**: `agents/rag/config/api-mode/settings.yml` → `embedding_model: "<url>:::<model>"`
- **DeepResearch**: `agents/deep-research/config/api-mode/setup/core.service.embedding.yaml` → `location`, `model`, `api_key`

## Search API

DeepResearch and OpenClaw use a web search backend. The search provider is
selected via `--search-provider`:

| Provider | Key needed? | Notes |
|----------|-------------|-------|
| `firecrawl` | ✅ (`--search-key fc-xxx`) | Default; remote API at `api.firecrawl.dev` |
| `duckduckgo` | ❌ | Free, rate-limited; good for smoke tests |
| `none` | ❌ | Disable search (deep-research will fail at the search step) |

```bash
# Firecrawl (default)
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx \
    --search-provider firecrawl --search-key fc-xxx

# DuckDuckGo (no key)
bash scripts/configure-api.sh --llm-preset dashscope --llm-key sk-xxx \
    --search-provider duckduckgo
```

The script patches:
- **DeepResearch**: `deep_research.langchain_drs.yaml` → `search_api`; `core.service.search.yaml` → `default_engine`. The Firecrawl key is injected via the `FIRECRAWL_API_KEY` env var in `.env` (lego_agent expands `${FIRECRAWL_API_KEY}` at runtime).
- **OpenClaw**: `config/.env` → `FIRECRAWL_API_KEY`

### Adding a custom search provider

The DeepResearch search config (`agents/deep-research/config/api-mode/setup/core.service.search.yaml`) supports 11 backends out of the box: tavily, arxiv, duckduckgo, wiki, perplexity, googlesearch, exa, linkup, serper, searxng, firecrawl. To use one of these, set `--search-provider <name>` and add its API key to `.env` manually if required. For a completely custom search API (e.g., an internal service like EVA), add a new instance block to the search YAML and set `search_api` in `deep_research.langchain_drs.yaml` to match the instance's `engine` field.

## Proxy Architecture

When proxies are needed, they are started as sidecar containers in the docker
compose stack.

### LiteLLM Proxy

**Purpose:** Translates OpenAI Responses API → Chat Completions API.

```
Agent CLI → /v1/responses → LiteLLM → /v1/chat/completions → LLM Provider
```

Used by: Codex (always), Pi (when not using DeepSeek native provider).

The proxy image is built automatically by `configure-api.sh`:

```bash
docker build -t litellm-proxy:local -f - . << 'EOF'
FROM python:3.12
RUN pip install 'litellm[proxy]==1.105.1'
EXPOSE 4000
CMD ["litellm", "--port", "4000"]
EOF
```

Configuration: `agents/<name>/config/litellm-config.yaml`

### Codex Proxy

**Purpose:** Strips non-function tool types from Codex CLI requests.

Codex sends tool definitions with types like `namespace`, `web_search_preview`,
`file_search` that DeepSeek/DashScope reject. This proxy removes them before
forwarding to LiteLLM.

```
Codex CLI → codex-proxy (strip tools) → LiteLLM → LLM Provider
```

Used by: Codex only. Built into the `codex:v1` image.

### Anthropic Proxy

**Purpose:** Translates Anthropic Messages API → OpenAI Chat Completions API.

Claude Code CLI uses the Anthropic SDK internally, which sends requests in
Anthropic's format. This proxy translates them to OpenAI Chat Completions
format so any OpenAI-compatible provider can serve them.

```
Claude CLI → /v1/messages → anthropic-proxy → /v1/chat/completions → LLM Provider
```

Used by: Claude only. Source: `agents/claude/scripts/anthropic_proxy.py`

### OpenClaw LLM Proxy

**Purpose:** Instrumented forwarding (not protocol translation).

Unlike the others, this proxy doesn't change the protocol — it records every
LLM request/response for telemetry while forwarding to the upstream API. It's
part of AgentSysBench's instrumentation layer, not an API compatibility layer.

```
OpenClaw CLI → llm-proxy (record + forward) → LLM Provider
```

## Per-Agent Override

If you need to configure a specific agent differently from the global setting,
run `configure-api.sh` with `--agent <name>`, or edit its `config/.env`
directly. Re-running `configure-api.sh` without `--agent` overwrites all
agents' `.env` files and re-patches the YAML templates from their `.example`
counterparts.

Key variables per agent:

| Agent | Key Variables |
|-------|--------------|
| mini-swe-agent | `OPENAI_API_KEY`, `OPENAI_API_BASE`, `MINI_SWE_MODEL` |
| codex | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `CODEX_MODEL` |
| claude | `ANTHROPIC_API_KEY`, `UPSTREAM_BASE_URL`, `UPSTREAM_API_KEY`, `CLAUDE_MODEL` |
| openclaw | `DEEPSEEK_API_KEY`, `FIRECRAWL_API_KEY` |
| webagent | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `WEBAGENT_MODEL` |
| pi-autoresearch | `PI_PROVIDER`, `PI_MODEL`, `OPENAI_API_KEY` or `DEEPSEEK_API_KEY` |
| rag | `OPENAI_API_KEY` + `config/api-mode/settings.yml` (embedding, situating) |
| deep-research | `OPENAI_API_KEY`, `FIRECRAWL_API_KEY` + `config/api-mode/setup/*.yaml` |
