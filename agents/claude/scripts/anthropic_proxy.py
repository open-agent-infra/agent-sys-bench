"""Anthropic-to-OpenAI API proxy for Claude Code CLI.

Claude Code CLI sends Anthropic-format requests (POST /v1/messages) but
DashScope only serves OpenAI-format (/v1/chat/completions).  This proxy
translates between the two so Claude CLI can work against DashScope.

Usage:
    UPSTREAM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
    UPSTREAM_API_KEY=sk-... \
    python anthropic_proxy.py [--port 8090]

Then set:
    ANTHROPIC_BASE_URL=http://localhost:8090/v1
    ANTHROPIC_API_KEY=<anything>  # proxy forwards the real key

Claude Code CLI will POST to http://localhost:8090/v1/messages, this proxy
translates to OpenAI chat/completions and translates the response back.

Supports streaming (SSE) which Claude Code CLI uses by default.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

UPSTREAM_BASE = os.environ.get(
    "UPSTREAM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
UPSTREAM_KEY = os.environ.get("UPSTREAM_API_KEY", "")
PORT = int(os.environ.get("PROXY_PORT", "8090"))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
log = logging.getLogger("anthropic_proxy")

app = FastAPI()

_CLIENT: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=30.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
    return _CLIENT


# ---------- Anthropic -> OpenAI message translation -------------------------

def _anthropic_content_to_openai(content: Any) -> tuple:
    """Convert Anthropic content blocks to OpenAI message content + tool_calls.

    Returns (text_content, tool_calls_list).
    """
    if isinstance(content, str):
        return content, []

    text_parts = []
    tool_calls = []

    if isinstance(content, list):
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    text_parts.append(block.get("text", ""))
                elif btype == "thinking":
                    # Pass thinking as a special text prefix
                    text_parts.append(f"<think>{block.get('thinking', '')}</think>")
                elif btype == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
                elif btype == "tool_result":
                    # This is in user messages
                    pass
                elif btype == "image":
                    # Pass image blocks (base64)
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        text_parts.append(f"[image: {source.get('media_type', 'image/png')}]")

    return "\n".join(text_parts) if text_parts else "", tool_calls


def _anthropic_messages_to_openai(messages: list) -> list:
    """Convert Anthropic message list to OpenAI message list."""
    result = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            # Check for tool_result blocks
            if isinstance(content, list):
                tool_results = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]
                text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") != "tool_result"]
                non_dict = [b for b in content if not isinstance(b, dict)]

                for tr in tool_results:
                    tr_content = tr.get("content", "")
                    if isinstance(tr_content, list):
                        parts = []
                        for c in tr_content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                parts.append(c.get("text", ""))
                            elif isinstance(c, str):
                                parts.append(c)
                        tr_text = "\n".join(parts)
                    elif isinstance(tr_content, str):
                        tr_text = tr_content
                    else:
                        tr_text = str(tr_content)
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tr_text,
                    })

                remaining = text_blocks + non_dict
                if remaining:
                    text, _ = _anthropic_content_to_openai(remaining)
                    if text:
                        result.append({"role": "user", "content": text})
            else:
                text, _ = _anthropic_content_to_openai(content)
                result.append({"role": "user", "content": text or ""})

        elif role == "assistant":
            text, tool_calls = _anthropic_content_to_openai(content)
            oai_msg: Dict[str, Any] = {"role": "assistant"}
            if text:
                oai_msg["content"] = text
            else:
                oai_msg["content"] = None
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            result.append(oai_msg)

    return result


def _anthropic_tools_to_openai(tools: list) -> list:
    """Convert Anthropic tool definitions to OpenAI function-calling format."""
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return result


def _build_openai_request(anthropic_body: dict) -> dict:
    """Build an OpenAI chat/completions request from an Anthropic messages request."""
    messages = _anthropic_messages_to_openai(anthropic_body.get("messages", []))

    # Prepend system message if present
    system = anthropic_body.get("system")
    if system:
        if isinstance(system, str):
            messages.insert(0, {"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic system can be a list of content blocks
            parts = []
            for block in system:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            messages.insert(0, {"role": "system", "content": "\n".join(parts)})

    oai_req: Dict[str, Any] = {
        "model": anthropic_body.get("model", "deepseek-v4-pro"),
        "messages": messages,
        "max_tokens": anthropic_body.get("max_tokens", 8192),
    }

    # Temperature
    if "temperature" in anthropic_body:
        oai_req["temperature"] = anthropic_body["temperature"]

    # Top-p
    if "top_p" in anthropic_body:
        oai_req["top_p"] = anthropic_body["top_p"]

    # Stop sequences
    if "stop_sequences" in anthropic_body:
        oai_req["stop"] = anthropic_body["stop_sequences"]

    # Tools
    if "tools" in anthropic_body and anthropic_body["tools"]:
        oai_req["tools"] = _anthropic_tools_to_openai(anthropic_body["tools"])

    # Streaming
    if anthropic_body.get("stream", False):
        oai_req["stream"] = True
        oai_req["stream_options"] = {"include_usage": True}

    return oai_req


# ---------- OpenAI -> Anthropic response translation -------------------------

def _anthropic_usage(usage: dict) -> dict:
    """Map OpenAI usage to Anthropic's: input_tokens excludes cache hits,
    which are reported separately as cache_read_input_tokens."""
    prompt = usage.get("prompt_tokens", 0) or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    return {
        "input_tokens": max(prompt - cached, 0),
        "output_tokens": usage.get("completion_tokens", 0) or 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": cached,
    }


def _openai_response_to_anthropic(oai_resp: dict, model: str) -> dict:
    """Convert a non-streaming OpenAI response to Anthropic messages format."""
    choice = oai_resp.get("choices", [{}])[0]
    message = choice.get("message", {})
    usage = oai_resp.get("usage", {})

    content_blocks = []

    # Check for reasoning_content (DeepSeek thinking)
    reasoning = message.get("reasoning_content")
    if reasoning:
        content_blocks.append({
            "type": "thinking",
            "thinking": reasoning,
        })

    # Text content
    text = message.get("content")
    if text:
        # Strip thinking tags if present (from our earlier translation)
        content_blocks.append({
            "type": "text",
            "text": text,
        })

    # Tool calls
    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {"raw": func.get("arguments", "")}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
            "name": func.get("name", ""),
            "input": args,
        })

    stop_reason = "end_turn"
    finish = choice.get("finish_reason", "stop")
    if finish == "tool_calls":
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    elif finish == "stop":
        stop_reason = "end_turn"

    return {
        "id": f"msg_{oai_resp.get('id', uuid.uuid4().hex[:12])}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": _anthropic_usage(usage),
    }


# ---------- Streaming translation -------------------------------------------

async def _stream_anthropic_from_openai(
    oai_response: httpx.Response,
    model: str,
    msg_id: str,
):
    """Translate OpenAI SSE stream to Anthropic SSE stream format."""

    # Send message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': 0, 'output_tokens': 0, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0}}})}\n\n"

    content_block_idx = 0
    current_tool_id = None
    current_tool_name = None
    in_text_block = False
    in_thinking_block = False
    in_tool_block = False
    final_usage = {}
    stop_reason = "end_turn"
    accumulated_reasoning = ""
    accumulated_text = ""

    async for line in oai_response.aiter_lines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # Handle usage in final chunk
            if chunk.get("usage"):
                final_usage = chunk["usage"]

            # Handle reasoning_content (DeepSeek thinking)
            reasoning_content = delta.get("reasoning_content", "")
            if reasoning_content:
                if not in_thinking_block:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_idx, 'content_block': {'type': 'thinking', 'thinking': ''}})}\n\n"
                    in_thinking_block = True
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_idx, 'delta': {'type': 'thinking_delta', 'thinking': reasoning_content}})}\n\n"
                accumulated_reasoning += reasoning_content

            # Handle text content
            text_content = delta.get("content", "")
            if text_content:
                if in_thinking_block:
                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_idx})}\n\n"
                    content_block_idx += 1
                    in_thinking_block = False

                if not in_text_block:
                    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': content_block_idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                    in_text_block = True
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_idx, 'delta': {'type': 'text_delta', 'text': text_content}})}\n\n"
                accumulated_text += text_content

            # Handle tool calls
            tool_calls = delta.get("tool_calls", [])
            for tc in tool_calls:
                tc_idx = tc.get("index", 0)
                func = tc.get("function", {})
                tc_id = tc.get("id")
                tc_name = func.get("name")
                tc_args = func.get("arguments", "")

                if tc_id:  # New tool call
                    # Close previous blocks
                    if in_text_block:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_idx})}\n\n"
                        content_block_idx += 1
                        in_text_block = False
                    if in_thinking_block:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_idx})}\n\n"
                        content_block_idx += 1
                        in_thinking_block = False
                    if in_tool_block:
                        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_idx})}\n\n"
                        content_block_idx += 1

                    current_tool_id = tc_id
                    current_tool_name = tc_name or ""
                    in_tool_block = True
                    tool_start_data = {
                        'type': 'content_block_start',
                        'index': content_block_idx,
                        'content_block': {
                            'type': 'tool_use',
                            'id': current_tool_id,
                            'name': current_tool_name,
                            'input': {},
                        },
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(tool_start_data)}\n\n"

                if tc_args:
                    yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': content_block_idx, 'delta': {'type': 'input_json_delta', 'partial_json': tc_args}})}\n\n"

            # Handle finish
            if finish_reason:
                if finish_reason == "tool_calls":
                    stop_reason = "tool_use"
                elif finish_reason == "length":
                    stop_reason = "max_tokens"
                else:
                    stop_reason = "end_turn"

    # Close any open blocks
    if in_thinking_block or in_text_block or in_tool_block:
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': content_block_idx})}\n\n"

    # If no content blocks were started, emit an empty text block
    if content_block_idx == 0 and not in_text_block and not in_thinking_block and not in_tool_block:
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # Send message_delta with stop reason. Prompt usage only arrives in the
    # last OpenAI chunk (after message_start went out with zeros), so carry
    # the full usage here; Anthropic clients merge message_delta usage.
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': _anthropic_usage(final_usage)})}\n\n"

    # Send message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# ---------- API endpoints ---------------------------------------------------

@app.post("/v1/messages")
async def messages(request: Request):
    """Handle Anthropic messages API requests."""
    body = await request.json()
    model = body.get("model", "deepseek-v4-pro")
    is_stream = body.get("stream", False)

    # Get API key from agsbench_request header or env
    api_key = UPSTREAM_KEY
    auth_header = request.headers.get("x-api-key") or request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = api_key or auth_header[7:]

    log.info(
        "Translating Anthropic request: model=%s, stream=%s, msgs=%d, tools=%d",
        model,
        is_stream,
        len(body.get("messages", [])),
        len(body.get("tools", [])),
    )

    oai_req = _build_openai_request(body)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    client = _get_client()
    upstream_url = f"{UPSTREAM_BASE.rstrip('/')}/chat/completions"

    if is_stream:
        try:
            oai_response = await client.post(
                upstream_url,
                json=oai_req,
                headers=headers,
            )
            oai_response.raise_for_status()
        except httpx.HTTPStatusError as e:
            log.error("Upstream error: %s %s", e.response.status_code, e.response.text[:500])
            return JSONResponse(
                status_code=e.response.status_code,
                content={"type": "error", "error": {"type": "api_error", "message": e.response.text[:500]}},
            )
        except Exception as e:
            log.error("Upstream connection error: %s", e)
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "api_error", "message": str(e)}},
            )

        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        return StreamingResponse(
            _stream_anthropic_from_openai(oai_response, model, msg_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        try:
            oai_response = await client.post(
                upstream_url,
                json=oai_req,
                headers=headers,
            )
            oai_response.raise_for_status()
            oai_data = oai_response.json()
        except httpx.HTTPStatusError as e:
            log.error("Upstream error: %s %s", e.response.status_code, e.response.text[:500])
            return JSONResponse(
                status_code=e.response.status_code,
                content={"type": "error", "error": {"type": "api_error", "message": e.response.text[:500]}},
            )
        except Exception as e:
            log.error("Upstream connection error: %s", e)
            return JSONResponse(
                status_code=502,
                content={"type": "error", "error": {"type": "api_error", "message": str(e)}},
            )

        anthropic_resp = _openai_response_to_anthropic(oai_data, model)
        return JSONResponse(content=anthropic_resp)


@app.get("/v1/models")
async def list_models():
    """Return a minimal model list for Claude CLI model validation."""
    return JSONResponse(content={
        "data": [
            {"id": "deepseek-v4-pro", "object": "model"},
            {"id": "deepseek-v4-flash", "object": "model"},
            {"id": "qwen-plus", "object": "model"},
            {"id": "qwen-max", "object": "model"},
        ]
    })


@app.head("/")
@app.head("/v1")
@app.get("/")
async def root_health():
    """Claude Code CLI does a HEAD / or HEAD /v1 on startup."""
    return JSONResponse(content={"status": "ok"})


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    log.info("Starting Anthropic-to-OpenAI proxy on port %d", PORT)
    log.info("Upstream: %s", UPSTREAM_BASE)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
