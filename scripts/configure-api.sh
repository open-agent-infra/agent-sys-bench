#!/bin/bash
# configure-api.sh — Centralized API configuration for AgentSysBench agents.
#
# Configures three categories of external API used by the benchmark agents:
#
#   1. LLM API        — chat / responses endpoint (all agents)
#                       Providers: dashscope, deepseek, openai, or custom
#
#   2. Embedding API  — vector embedding endpoint (rag, deep-research)
#                       Defaults to the LLM endpoint+key if not specified.
#                       DashScope and OpenAI serve embeddings on the same
#                       endpoint as LLM; DeepSeek does not serve embeddings,
#                       so deepseek users must pass --embedding-* separately.
#
#   3. Search API     — web search endpoint (deep-research, openclaw)
#                       Providers: firecrawl (key required), duckduckgo (free,
#                       no key), or none (disable search).
#
# Usage:
#   # Interactive — prompts for everything
#   bash scripts/configure-api.sh
#
#   # Non-interactive, DashScope for LLM + embedding, Firecrawl for search
#   bash scripts/configure-api.sh \
#       --llm-preset dashscope --llm-key sk-xxx --llm-model deepseek-v4-pro \
#       --search-provider firecrawl --search-key fc-xxx
#
#   # DeepSeek for LLM + DashScope for embedding (DeepSeek has no embedding API)
#   bash scripts/configure-api.sh \
#       --llm-preset deepseek --llm-key sk-xxx \
#       --embedding-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
#       --embedding-key sk-yyy --embedding-model text-embedding-v3
#
#   # Custom LLM (local vLLM), default embedding/search
#   bash scripts/configure-api.sh --llm-url http://localhost:8000/v1 --llm-key EMPTY --llm-model Qwen/Qwen3-8B
#
#   # gui-agent observes screenshots and needs a vision model; --vlm-model
#   # overrides it (default: kimi-k2.6 for dashscope, gpt-4o for openai,
#   # else --llm-model)
#   bash scripts/configure-api.sh --agent gui-agent \
#       --llm-preset dashscope --llm-key sk-xxx --vlm-model kimi-k2.6
#
#   # Configure a single agent only (others keep their existing config)
#   bash scripts/configure-api.sh --agent deep-research \
#       --llm-preset dashscope --llm-key sk-xxx --llm-model deepseek-v4-pro \
#       --search-provider firecrawl --search-key fc-xxx
#
#   # Show current configuration
#   bash scripts/configure-api.sh --list
#
# This script:
#   - Writes per-agent config/.env files (gitignored)
#   - Copies *.example templates to real files and patches keys into them
#     (real files are gitignored; *.example stays secret-free in the repo)
#   - Builds the litellm-proxy:local image if Responses-API translation is needed

set -e
AGSBENCH_HOME="$(cd "$(dirname "$0")/.." && pwd)"
AGSBENCH_ENV="${AGSBENCH_HOME}/.env"
[ -f "$AGSBENCH_ENV" ] && set -a && . "$AGSBENCH_ENV" && set +a
AGENTS_DIR="${AGSBENCH_HOME}/agents"
LLM_ENV="${AGSBENCH_HOME}/llm.env"

# Agents that have config/ directories
ALL_AGENTS=(claude codex deep-research gui-agent hugginggpt mini-swe-agent openclaw pi-autoresearch rag webagent)

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
err()   { echo -e "${RED}✗${NC} $*" >&2; }
header(){ echo -e "\n${BOLD}$*${NC}"; }

# ---------------------------------------------------------------------------
# Preset tables: LLM + default embedding model per provider
# ---------------------------------------------------------------------------
declare -A LLM_PRESET_URL LLM_PRESET_MODEL LLM_PRESET_RESPONSES LLM_PRESET_DESC
declare -A EMB_PRESET_MODEL EMB_PRESET_DESC

# --- LLM presets ---
LLM_PRESET_URL[dashscope]="https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_PRESET_MODEL[dashscope]="deepseek-v4-pro"
LLM_PRESET_RESPONSES[dashscope]="no"
LLM_PRESET_DESC[dashscope]="DashScope (Alibaba Cloud) — serves both LLM and embedding"

LLM_PRESET_URL[deepseek]="https://api.deepseek.com/v1"
LLM_PRESET_MODEL[deepseek]="deepseek-chat"
LLM_PRESET_RESPONSES[deepseek]="no"
LLM_PRESET_DESC[deepseek]="DeepSeek API — LLM only (no embedding service)"

LLM_PRESET_URL[openai]="https://api.openai.com/v1"
LLM_PRESET_MODEL[openai]="gpt-4o"
LLM_PRESET_RESPONSES[openai]="yes"
LLM_PRESET_DESC[openai]="OpenAI API — serves both LLM and embedding"

# --- Vision model for gui-agent (screenshot observations) ---
declare -A VLM_PRESET_MODEL
VLM_PRESET_MODEL[dashscope]="kimi-k2.6"
VLM_PRESET_MODEL[openai]="gpt-4o"

# --- Embedding default models (when embedding endpoint == LLM endpoint) ---
EMB_PRESET_MODEL[dashscope]="text-embedding-v3"
EMB_PRESET_MODEL[openai]="text-embedding-3-small"
EMB_PRESET_DESC[dashscope]="DashScope text-embedding-v3 (same endpoint as LLM)"
EMB_PRESET_DESC[openai]="OpenAI text-embedding-3-small (same endpoint as LLM)"

# --- Search providers ---
declare -A SEARCH_PROVIDER_DESC
SEARCH_PROVIDER_DESC[firecrawl]="Firecrawl remote API (api.firecrawl.dev) — requires API key"
SEARCH_PROVIDER_DESC[duckduckgo]="DuckDuckGo (free, no key, rate-limited) — good for smoke tests"
SEARCH_PROVIDER_DESC[none]="Disable web search (deep-research will fail at search step)"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
TARGET_AGENT=""
DO_LIST=false

# LLM
LLM_PRESET=""
LLM_URL=""
LLM_KEY=""
LLM_MODEL=""
RESPONSES_API=""
VLM_MODEL=""

# Embedding (optional, defaults to LLM)
EMBEDDING_URL=""
EMBEDDING_KEY=""
EMBEDDING_MODEL=""

# Search (optional)
SEARCH_PROVIDER=""
SEARCH_KEY=""

while [[ $# -gt 0 ]]; do
    case $1 in
        # Scope
        --agent)         TARGET_AGENT="$2"; shift 2 ;;
        --list)          DO_LIST=true; shift ;;
        # LLM
        --llm-preset)    LLM_PRESET="$2"; shift 2 ;;
        --llm-url)       LLM_URL="$2"; shift 2 ;;
        --llm-key)       LLM_KEY="$2"; shift 2 ;;
        --llm-model)     LLM_MODEL="$2"; shift 2 ;;
        --responses-api) RESPONSES_API="$2"; shift 2 ;;
        --vlm-model)     VLM_MODEL="$2"; shift 2 ;;
        # Embedding
        --embedding-url)   EMBEDDING_URL="$2"; shift 2 ;;
        --embedding-key)   EMBEDDING_KEY="$2"; shift 2 ;;
        --embedding-model) EMBEDDING_MODEL="$2"; shift 2 ;;
        # Search
        --search-provider) SEARCH_PROVIDER="$2"; shift 2 ;;
        --search-key)       SEARCH_KEY="$2"; shift 2 ;;
        # Help
        --help|-h)
            head -50 "$0" | grep '^#' | sed 's/^# \?//'
            exit 0 ;;
        *) err "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# --list: show current configuration
# ---------------------------------------------------------------------------
if [ "$DO_LIST" = true ]; then
    header "Current API Configuration"
    echo ""
    if [ -f "$LLM_ENV" ]; then
        echo "  Global config (llm.env):"
        sed 's/^/    /' "$LLM_ENV"
        echo ""
    else
        warn "No global llm.env found. Run configure-api.sh to create one."
        echo ""
    fi
    printf "  %-18s %-10s %-45s %s\n" "Agent" "Status" "LLM Endpoint" "Model"
    printf "  %-18s %-10s %-45s %s\n" "─────" "──────" "────────────" "─────"
    for agent in "${ALL_AGENTS[@]}"; do
        env_file="${AGENTS_DIR}/${agent}/config/.env"
        if [ -f "$env_file" ]; then
            url=$(grep -E '^OPENAI_BASE_URL=|^OPENAI_API_BASE=|^UPSTREAM_BASE_URL=' "$env_file" | head -1 | cut -d= -f2-)
            model=$(grep -E '^CODEX_MODEL=|^MINI_SWE_MODEL=|^CLAUDE_MODEL=|^WEBAGENT_MODEL=|^PI_MODEL=|^LLM_MODEL=' "$env_file" | head -1 | cut -d= -f2-)
            [ -z "$url" ] && url="(see config files)"
            [ -z "$model" ] && model="(see config files)"
            printf "  %-18s ${GREEN}%-10s${NC} %-45s %s\n" "$agent" "configured" "$url" "$model"
        else
            printf "  %-18s ${YELLOW}%-10s${NC} %-45s %s\n" "$agent" "not set" "-" "-"
        fi
    done
    echo ""
    exit 0
fi

# ---------------------------------------------------------------------------
# Apply LLM preset
# ---------------------------------------------------------------------------
if [ -n "$LLM_PRESET" ]; then
    if [ -z "${LLM_PRESET_URL[$LLM_PRESET]+x}" ]; then
        err "Unknown LLM preset: $LLM_PRESET. Available: ${!LLM_PRESET_URL[*]}"
        exit 1
    fi
    [ -z "$LLM_URL" ] && LLM_URL="${LLM_PRESET_URL[$LLM_PRESET]}"
    [ -z "$LLM_MODEL" ] && LLM_MODEL="${LLM_PRESET_MODEL[$LLM_PRESET]}"
    [ -z "$RESPONSES_API" ] && RESPONSES_API="${LLM_PRESET_RESPONSES[$LLM_PRESET]}"
fi

# ---------------------------------------------------------------------------
# Interactive prompts (when no --llm-* flags given)
# ---------------------------------------------------------------------------
if [ -z "$LLM_URL" ] && [ -z "$LLM_PRESET" ]; then
    echo ""
    echo -e "${BOLD}AgentSysBench — API Configuration${NC}"
    echo ""
    echo "  1) DashScope      ${LLM_PRESET_DESC[dashscope]}"
    echo "  2) DeepSeek       ${LLM_PRESET_DESC[deepseek]}"
    echo "  3) OpenAI         ${LLM_PRESET_DESC[openai]}"
    echo "  4) Local/Custom   vLLM, SGLang, or any OpenAI-compatible endpoint"
    echo ""
    read -p "  LLM provider [1-4]: " choice
    case $choice in
        1) LLM_PRESET="dashscope" ;;
        2) LLM_PRESET="deepseek" ;;
        3) LLM_PRESET="openai" ;;
        4) LLM_PRESET="" ;;
        *) err "Invalid choice"; exit 1 ;;
    esac
    if [ -n "$LLM_PRESET" ]; then
        LLM_URL="${LLM_PRESET_URL[$LLM_PRESET]}"
        [ -z "$RESPONSES_API" ] && RESPONSES_API="${LLM_PRESET_RESPONSES[$LLM_PRESET]}"
        read -p "  LLM model [${LLM_PRESET_MODEL[$LLM_PRESET]}]: " input_model
        LLM_MODEL="${input_model:-${LLM_PRESET_MODEL[$LLM_PRESET]}}"
    else
        read -p "  LLM base URL (e.g., http://localhost:8000/v1): " LLM_URL
        read -p "  LLM model name: " LLM_MODEL
        read -p "  Supports OpenAI Responses API? [y/N]: " resp_yn
        [ "$resp_yn" = "y" ] || [ "$resp_yn" = "Y" ] && RESPONSES_API="yes" || RESPONSES_API="no"
    fi
fi

if [ -z "$LLM_KEY" ]; then
    read -p "  LLM API key: " LLM_KEY
fi

# Defaults
[ -z "$LLM_MODEL" ] && LLM_MODEL="deepseek-chat"
[ -z "$RESPONSES_API" ] && RESPONSES_API="no"

# ---------------------------------------------------------------------------
# Embedding config — defaults to LLM endpoint+key
# ---------------------------------------------------------------------------
# Determine default embedding model based on LLM preset
DEFAULT_EMB_MODEL=""
if [ -n "$LLM_PRESET" ] && [ -n "${EMB_PRESET_MODEL[$LLM_PRESET]+x}" ]; then
    DEFAULT_EMB_MODEL="${EMB_PRESET_MODEL[$LLM_PRESET]}"
fi

# If embedding not explicitly set, inherit from LLM
if [ -z "$EMBEDDING_URL" ]; then
    EMBEDDING_URL="$LLM_URL"
fi
if [ -z "$EMBEDDING_KEY" ]; then
    EMBEDDING_KEY="$LLM_KEY"
fi
if [ -z "$EMBEDDING_MODEL" ] && [ -n "$DEFAULT_EMB_MODEL" ]; then
    EMBEDDING_MODEL="$DEFAULT_EMB_MODEL"
fi

# DeepSeek has no embedding service — warn if user didn't override
if [ "$LLM_PRESET" = "deepseek" ] && [ "$EMBEDDING_URL" = "$LLM_URL" ]; then
    warn "DeepSeek does not serve embeddings. RAG and DeepResearch will not work"
    warn "  without a separate embedding endpoint. Re-run with --embedding-url,"
    warn "  --embedding-key, and --embedding-model to provide one."
fi

# ---------------------------------------------------------------------------
# Search config — defaults to firecrawl
# ---------------------------------------------------------------------------
if [ -z "$SEARCH_PROVIDER" ]; then
    if [ -z "$SEARCH_KEY" ]; then
        # No search key given and no provider specified — ask
        echo ""
        echo "  Web search (deep-research, openclaw) needs a search backend."
        echo "  1) Firecrawl      ${SEARCH_PROVIDER_DESC[firecrawl]}"
        echo "  2) DuckDuckGo     ${SEARCH_PROVIDER_DESC[duckduckgo]}"
        echo "  3) None           ${SEARCH_PROVIDER_DESC[none]}"
        read -p "  Search provider [1-3, default=1]: " sch_choice
        case "${sch_choice:-1}" in
            1) SEARCH_PROVIDER="firecrawl" ;;
            2) SEARCH_PROVIDER="duckduckgo" ;;
            3) SEARCH_PROVIDER="none" ;;
            *) err "Invalid choice"; exit 1 ;;
        esac
    else
        SEARCH_PROVIDER="firecrawl"
    fi
fi

if [ "$SEARCH_PROVIDER" = "firecrawl" ] && [ -z "$SEARCH_KEY" ]; then
    read -p "  Firecrawl API key: " SEARCH_KEY
fi

[ -z "$VLM_MODEL" ] && VLM_MODEL="${VLM_PRESET_MODEL[${LLM_PRESET:-none}]:-$LLM_MODEL}"

# ---------------------------------------------------------------------------
# Determine proxy needs and pi provider
# ---------------------------------------------------------------------------
NEEDS_LITELLM=false
[ "$RESPONSES_API" = "no" ] && NEEDS_LITELLM=true

PI_PROVIDER="openai"
PI_LLM_URL="$LLM_URL"
if [ "$LLM_PRESET" = "deepseek" ]; then
    PI_PROVIDER="deepseek"
    PI_LLM_URL=""
elif [ "$NEEDS_LITELLM" = "true" ]; then
    PI_LLM_URL="http://localhost:4000/v1"
fi

# ---------------------------------------------------------------------------
# Which agents to configure
# ---------------------------------------------------------------------------
if [ -n "$TARGET_AGENT" ]; then
    found=false
    for a in "${ALL_AGENTS[@]}"; do [ "$a" = "$TARGET_AGENT" ] && found=true; done
    if [ "$found" = false ]; then
        err "Unknown agent: $TARGET_AGENT"
        echo "  Available: ${ALL_AGENTS[*]}"
        exit 1
    fi
    AGENTS_TO_CONFIGURE=("$TARGET_AGENT")
else
    AGENTS_TO_CONFIGURE=("${ALL_AGENTS[@]}")
fi

# ---------------------------------------------------------------------------
# Write global llm.env (only when configuring all agents)
# ---------------------------------------------------------------------------
if [ -z "$TARGET_AGENT" ]; then
    cat > "$LLM_ENV" << EOF
# Generated by configure-api.sh — $(date -Iseconds)
LLM_PROVIDER=${LLM_PRESET:-custom}
LLM_URL=${LLM_URL}
LLM_MODEL=${LLM_MODEL}
NEEDS_LITELLM=${NEEDS_LITELLM}
EMBEDDING_URL=${EMBEDDING_URL}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-none}
SEARCH_PROVIDER=${SEARCH_PROVIDER}
EOF
fi

# ---------------------------------------------------------------------------
# Helper: patch RAG settings (LLM + embedding + situating)
# ---------------------------------------------------------------------------
_patch_rag_settings() {
    # API-mode settings (mounted at /config/settings.yml by API-mode compose)
    local api_settings="${AGENTS_DIR}/rag/config/api-mode/settings.yml"
    local api_template="${api_settings}.example"
    if [ -f "$api_template" ]; then
        cp "$api_template" "$api_settings"
        sed -i "s|openai_api_key:.*\".*\"|openai_api_key: \"${LLM_KEY}\"|" "$api_settings"
        sed -i "s|openai_base_url:.*\".*\"|openai_base_url: \"${LLM_URL}\"|" "$api_settings"
        sed -i "s|^llm_model:.*|llm_model: \"${LLM_MODEL}\"|" "$api_settings"
        # situating LLM shares the same endpoint/key as the main LLM
        sed -i "s|situating_llm_model:.*|situating_llm_model: \"${LLM_MODEL}\"|" "$api_settings"
        sed -i "s|situating_device:.*\"http.*\"|situating_device: \"${LLM_URL}\"|" "$api_settings"
        sed -i "s|situating_api_key:.*\".*\"|situating_api_key: \"${LLM_KEY}\"|" "$api_settings"
        # embedding endpoint is encoded as "<base_url>:::<model>"
        if [ -n "$EMBEDDING_MODEL" ]; then
            sed -i "s|embedding_model:.*\".*:::.*\"|embedding_model: \"${EMBEDDING_URL}:::${EMBEDDING_MODEL}\"|" "$api_settings"
        fi
    fi
    # Root template (for GPU-mode users)
    local root_settings="${AGENTS_DIR}/rag/config/settings-api.yml"
    local root_template="${root_settings}.example"
    if [ -f "$root_template" ]; then
        cp "$root_template" "$root_settings"
        sed -i "s|openai_api_key:.*\".*\"|openai_api_key: \"${LLM_KEY}\"|" "$root_settings"
        sed -i "s|openai_base_url:.*\".*\"|openai_base_url: \"${LLM_URL}\"|" "$root_settings"
        sed -i "s|^llm_model:.*|llm_model: \"${LLM_MODEL}\"|" "$root_settings"
        sed -i "s|situating_llm_model:.*|situating_llm_model: \"${LLM_MODEL}\"|" "$root_settings"
        sed -i "s|situating_device:.*\"http.*\"|situating_device: \"${LLM_URL}\"|" "$root_settings"
        sed -i "s|situating_api_key:.*\".*\"|situating_api_key: \"${LLM_KEY}\"|" "$root_settings"
        if [ -n "$EMBEDDING_MODEL" ]; then
            sed -i "s|embedding_model:.*\".*:::.*\"|embedding_model: \"${EMBEDDING_URL}:::${EMBEDDING_MODEL}\"|" "$root_settings"
        fi
    fi
}

# ---------------------------------------------------------------------------
# Helper: patch HuggingGPT settings (LLM only; models run locally on GPU)
# ---------------------------------------------------------------------------
_patch_hugginggpt_settings() {
    # All four LLM stages (task planning, model selection, model inference,
    # response generation) share the same endpoint/key/model.
    local settings="${AGENTS_DIR}/hugginggpt/config/settings.yml"
    local template="${settings}.example"
    if [ -f "$template" ]; then
        cp "$template" "$settings"
        sed -i "s|^\(openai_api_key_[a-z_]*\):.*|\1: \"${LLM_KEY}\"|" "$settings"
        sed -i "s|^\(openai_base_url_[a-z_]*\):.*|\1: \"${LLM_URL}\"|" "$settings"
        sed -i "s|^\(llm_model_[a-z_]*\):.*|\1: \"${LLM_MODEL}\"|" "$settings"
    fi
}

# ---------------------------------------------------------------------------
# Helper: patch DeepResearch configs (LLM + embedding + search)
# ---------------------------------------------------------------------------
_patch_deep_research_configs() {
    # --- LLM config (quoted strings, not env-expanded — patch directly) ---
    local api_llm="${AGENTS_DIR}/deep-research/config/api-mode/setup/core.service.llm.yaml"
    local api_llm_tpl="${api_llm}.example"
    if [ -f "$api_llm_tpl" ]; then
        cp "$api_llm_tpl" "$api_llm"
        sed -i "s|api_key:.*\".*\"|api_key: \"${LLM_KEY}\"|" "$api_llm"
        sed -i "s|location:.*\"http.*\"|location: \"${LLM_URL}\"|" "$api_llm"
        sed -i "s|default_model:.*|default_model: \"${LLM_MODEL}\"|" "$api_llm"
        sed -i "s|model:.*\".*\"|model: \"${LLM_MODEL}\"|" "$api_llm"
    fi
    local root_llm="${AGENTS_DIR}/deep-research/config/core.service.llm-api.yaml"
    local root_llm_tpl="${root_llm}.example"
    if [ -f "$root_llm_tpl" ]; then
        cp "$root_llm_tpl" "$root_llm"
        sed -i "s|api_key:.*\".*\"|api_key: \"${LLM_KEY}\"|" "$root_llm"
        sed -i "s|location:.*\"http.*\"|location: \"${LLM_URL}\"|" "$root_llm"
    fi

    # --- Embedding config (copy .example, then patch) ---
    # The .example uses ${OPENAI_API_KEY} env ref which lego_agent expands
    # at runtime, so we only patch location/model — NOT the api_key.
    if [ -n "$EMBEDDING_MODEL" ]; then
        local api_emb="${AGENTS_DIR}/deep-research/config/api-mode/setup/core.service.embedding.yaml"
        local api_emb_tpl="${api_emb}.example"
        if [ -f "$api_emb_tpl" ]; then
            cp "$api_emb_tpl" "$api_emb"
            sed -i "s|default_model:.*|default_model: \"${EMBEDDING_MODEL}\"|" "$api_emb"
            sed -i "s|location:.*\"http.*\"|location: \"${EMBEDDING_URL}\"|" "$api_emb"
            sed -i "s|model:.*\".*\"|model: \"${EMBEDDING_MODEL}\"|" "$api_emb"
            # Leave api_key as ${OPENAI_API_KEY} env ref — the .env flow
            # sets OPENAI_API_KEY in the container env, and lego_agent
            # expands it at runtime. No real key in the YAML file.
        fi
    fi

    # --- Search config (copy .example, then patch default_engine) ---
    # The search YAML uses unquoted ${FIRECRAWL_API_KEY} which lego_agent
    # expands from the container env at runtime, so we don't patch the key.
    # We only set the default_engine and the search_api selector.
    local drs_cfg="${AGENTS_DIR}/deep-research/config/api-mode/setup/deep_research.langchain_drs.yaml"
    local drs_tpl="${drs_cfg}.example"
    if [ -f "$drs_tpl" ]; then
        cp "$drs_tpl" "$drs_cfg"
        sed -i "s|^search_api:.*|search_api: \"${SEARCH_PROVIDER}\"|" "$drs_cfg"
    fi
    local search_yaml="${AGENTS_DIR}/deep-research/config/api-mode/setup/core.service.search.yaml"
    local search_tpl="${search_yaml}.example"
    if [ -f "$search_tpl" ]; then
        cp "$search_tpl" "$search_yaml"
        sed -i "s|^default_engine:.*|default_engine: ${SEARCH_PROVIDER}|" "$search_yaml"
    fi
}

# Generate per-agent config
# ---------------------------------------------------------------------------
header "Configuring APIs for: ${AGENTS_TO_CONFIGURE[*]}"
echo "  LLM:        ${LLM_URL}  (model: ${LLM_MODEL})"
echo "  Embedding:  ${EMBEDDING_URL}  (model: ${EMBEDDING_MODEL:-none})"
echo "  Search:     ${SEARCH_PROVIDER}"
echo ""

for agent in "${AGENTS_TO_CONFIGURE[@]}"; do
    env_file="${AGENTS_DIR}/${agent}/config/.env"
    mkdir -p "$(dirname "$env_file")"

    case $agent in
        mini-swe-agent)
            cat > "$env_file" << EOF
OPENAI_API_KEY=${LLM_KEY}
OPENAI_API_BASE=${LLM_URL}
MINI_SWE_MODEL=openai/${LLM_MODEL}
MSWEA_COST_TRACKING=ignore_errors
EOF
            ;;
        gui-agent)
            cat > "$env_file" << EOF
OPENAI_API_KEY=${LLM_KEY}
OPENAI_BASE_URL=${LLM_URL}
OPENAI_API_BASE=${LLM_URL}
OS_WORLD_MODEL=${VLM_MODEL}
OS_WORLD_AGENT_KIND=$(case "$VLM_MODEL" in kimi*) echo kimi ;; *) echo owl15 ;; esac)
OS_WORLD_OBSERVATION_TYPE=screenshot
OS_WORLD_PROVIDER=docker
EOF
            ;;
        codex)
            cat > "$env_file" << EOF
OPENAI_API_KEY=${LLM_KEY}
OPENAI_BASE_URL=${LLM_URL}
OPENAI_API_BASE=${LLM_URL}
CODEX_MODEL=${LLM_MODEL}
CODEX_CPUSETS=0-1,2-3,4-5,6-7
CODEX_MEMORY=4g
EOF
            ;;
        claude)
            cat > "$env_file" << EOF
ANTHROPIC_API_KEY=${LLM_KEY}
ANTHROPIC_BASE_URL=http://anthropic-proxy:8090
CLAUDE_MODEL=${LLM_MODEL}
CLAUDE_CONCURRENCY=4
UPSTREAM_BASE_URL=${LLM_URL}
UPSTREAM_API_KEY=${LLM_KEY}
EOF
            ;;
        openclaw)
            # openclaw uses FIRECRAWL_API_KEY for its search proxy
            oc_search_key=""
            [ "$SEARCH_PROVIDER" = "firecrawl" ] && oc_search_key="$SEARCH_KEY"
            cat > "$env_file" << EOF
DEEPSEEK_API_KEY=${LLM_KEY}
UPSTREAM_BASE_URL=${LLM_URL}
OPENCLAW_MODEL_ID=${LLM_MODEL}
OPENCLAW_MODEL=deepseek-instrumented/${LLM_MODEL}
FIRECRAWL_API_KEY=${oc_search_key}
MOONSHOT_API_KEY=
OPENCLAW_SANDBOX=0
OPENCLAW_CONCURRENCY=4
OPENCLAW_MODE=single
EOF
            ;;
        webagent)
            cat > "$env_file" << EOF
OPENAI_API_KEY=${LLM_KEY}
OPENAI_BASE_URL=${LLM_URL}
WEBAGENT_MODEL=${LLM_MODEL}
WEBAGENT_CHROMIUM_ARGS=--no-sandbox --proxy-server=http://127.0.0.1:8899
WEBAGENT_CONCURRENCY=1
WA_SHOPPING_ADMIN=http://shopping-admin:80/admin
WA_SHOPPING=http://shopping-admin:80
WA_REDDIT=http://localhost:9999
WA_GITLAB=http://localhost:9998
WA_WIKIPEDIA=http://localhost:9997
WA_MAP=http://localhost:9996
WA_HOMEPAGE=http://localhost:9995
EOF
            ;;
        pi-autoresearch)
            if [ "$PI_PROVIDER" = "deepseek" ]; then
                cat > "$env_file" << EOF
PI_PROVIDER=deepseek
PI_MODEL=${LLM_MODEL}
DEEPSEEK_API_KEY=${LLM_KEY}
PI_THINKING=high
MAX_ITERATIONS=5
TASK_TIMEOUT=3600
PI_AUTORESEARCH_CONCURRENCY=1
EOF
            else
                cat > "$env_file" << EOF
PI_PROVIDER=openai
PI_MODEL=${LLM_MODEL}
OPENAI_API_KEY=${LLM_KEY}
OPENAI_BASE_URL=${PI_LLM_URL}
PI_THINKING=off
MAX_ITERATIONS=5
TASK_TIMEOUT=3600
PI_AUTORESEARCH_CONCURRENCY=1
EOF
            fi
            ;;
        rag)
            cat > "$env_file" << EOF
OPENAI_API_KEY=${LLM_KEY}
AGENT_DATA=./data
AGENT_CONFIG=./config
AGENT_RESULTS=./results
AGENT_CACHE=./cache
NLTK_DATA=/data/nltk_data
MONITORING_UID=$(id -u)
MONITORING_GID=$(id -g)
EOF
            _patch_rag_settings
            ;;
        deep-research)
            dr_search_key=""
            [ "$SEARCH_PROVIDER" = "firecrawl" ] && dr_search_key="$SEARCH_KEY"
            cat > "$env_file" << EOF
OPENAI_API_KEY=${LLM_KEY}
FIRECRAWL_API_KEY=${dr_search_key}
AGENT_CONFIG=./config
AGENT_DATA=./data
AGENT_RESULTS=./results
AGENT_CACHE=./cache
NLTK_DATA=/data/nltk_data
MONITORING_UID=$(id -u)
MONITORING_GID=$(id -g)
EOF
            _patch_deep_research_configs
            ;;
        hugginggpt)
            # Keys live in config/settings.yml (patched below); .env only
            # records the route for --list.
            cat > "$env_file" << EOF
OPENAI_BASE_URL=${LLM_URL}
LLM_MODEL=${LLM_MODEL}
EOF
            _patch_hugginggpt_settings
            ;;
    esac
    info "$agent"
done

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# LiteLLM proxy setup
# ---------------------------------------------------------------------------
if [ "$NEEDS_LITELLM" = "true" ]; then
    header "LiteLLM Proxy"
    echo "  Required for: codex, pi-autoresearch (Responses API translation)"
    echo ""

    for litellm_cfg in "${AGENTS_DIR}"/*/config/litellm-config.yaml; do
        [ -f "$litellm_cfg" ] || continue
        agent_name=$(echo "$litellm_cfg" | sed "s|.*/agents/\([^/]*\)/.*|\1|")
        skip=true
        for a in "${AGENTS_TO_CONFIGURE[@]}"; do [ "$a" = "$agent_name" ] && skip=false; done
        [ "$skip" = true ] && continue
        sed -i "s|api_base:.*|api_base: ${LLM_URL}|" "$litellm_cfg"
        info "${agent_name}/config/litellm-config.yaml"
    done

    if ! docker image inspect litellm-proxy:local >/dev/null 2>&1; then
        echo ""
        echo "  Building litellm-proxy:local..."
        PIP_INDEX="${PIP_INDEX_URL:-https://pypi.org/simple/}"
        docker build -q -t litellm-proxy:local -f - "${AGSBENCH_HOME}" << DOCKERFILE
FROM python:3.12
ARG PIP_INDEX_URL=${PIP_INDEX}
RUN pip config set global.index-url \${PIP_INDEX_URL} && pip install 'litellm[proxy]==1.105.1'
EXPOSE 4000
CMD ["litellm", "--port", "4000"]
DOCKERFILE
        info "litellm-proxy:local built"
    else
        info "litellm-proxy:local exists"
    fi
else
    header "LiteLLM Proxy"
    echo "  Not needed — endpoint supports Responses API natively."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
header "Done"
echo ""
printf "  %-18s %-22s %s\n" "Agent" "LLM Route" "Search"
printf "  %-18s %-22s %s\n" "─────" "─────────" "──────"
for agent in "${AGENTS_TO_CONFIGURE[@]}"; do
    case $agent in
        codex)
            if [ "$NEEDS_LITELLM" = "true" ]; then
                printf "  %-18s %-22s %s\n" "$agent" "→ litellm → ${LLM_URL}" "${SEARCH_PROVIDER}"
            else
                printf "  %-18s %-22s %s\n" "$agent" "→ ${LLM_URL} (direct)" "${SEARCH_PROVIDER}"
            fi ;;
        pi-autoresearch)
            if [ "$PI_PROVIDER" = "deepseek" ]; then
                printf "  %-18s %-22s %s\n" "$agent" "→ ${LLM_URL} (deepseek native)" "${SEARCH_PROVIDER}"
            elif [ "$NEEDS_LITELLM" = "true" ]; then
                printf "  %-18s %-22s %s\n" "$agent" "→ litellm → ${LLM_URL}" "${SEARCH_PROVIDER}"
            else
                printf "  %-18s %-22s %s\n" "$agent" "→ ${LLM_URL} (direct)" "${SEARCH_PROVIDER}"
            fi ;;
        claude)
            printf "  %-18s %-22s %s\n" "$agent" "→ anthropic-proxy → ${LLM_URL}" "${SEARCH_PROVIDER}" ;;
        openclaw)
            printf "  %-18s %-22s %s\n" "$agent" "→ llm-proxy → ${LLM_URL}" "${SEARCH_PROVIDER}" ;;
        rag)
            printf "  %-18s %-22s %s\n" "$agent" "→ ${LLM_URL}" "n/a" ;;
        deep-research)
            printf "  %-18s %-22s %s\n" "$agent" "→ ${LLM_URL}" "${SEARCH_PROVIDER}" ;;
        *)
            printf "  %-18s %-22s %s\n" "$agent" "→ ${LLM_URL} (direct)" "n/a" ;;
    esac
done

echo ""
echo "  Global config:  ${LLM_ENV}"
echo "  Provider docs:  docs/api-providers.md"
echo "  Mirror docs:    docs/china-mirrors.md"
echo ""
echo "  To override a single agent:"
echo "    bash scripts/configure-api.sh --agent <name> --llm-preset dashscope --llm-key sk-xxx"
echo ""
