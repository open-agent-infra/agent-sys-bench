"""Codex responses-API proxy.

Codex 0.133+ requires `wire_api = "responses"` — the chat path was removed
in https://github.com/openai/codex/discussions/7782. DeepSeek only implements
`/v1/chat/completions`. LiteLLM ships a responses→chat adapter but it loses
the function_call / function_call_output pairing on multi-turn requests:

  DeepseekException: "An assistant message with 'tool_calls' must be followed
  by tool messages responding to each 'tool_call_id'."

So we do the translation ourselves:

  POST /v1/responses (from agsbench_codex, SSE stream expected)
    -> translate to chat/completions, preserving id pairing
    -> forward to LITELLM_UPSTREAM /v1/chat/completions (non-streamed)
    -> synthesize a responses-API SSE event sequence from the chat completion
    -> stream it back to codex

LiteLLM still handles model routing, auth and observability — we just bypass
its broken responses adapter by talking to its chat endpoint directly.
"""

import json
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

UPSTREAM = os.environ.get("LITELLM_UPSTREAM", "http://litellm-proxy:4000")
PORT = int(os.environ.get("PROXY_PORT", "5000"))
# LiteLLM proxy refuses to start without a master key, and then only accepts
# that key. Codex sends the provider key, so swap it in when forwarding.
LITELLM_MASTER_KEY = os.environ.get("LITELLM_MASTER_KEY", "")
FWD_KEYS = ("authorization", "content-type", "accept", "openai-organization")


def _upstream_headers(request: Request) -> Dict[str, str]:
    headers = {k: v for k, v in request.headers.items() if k.lower() in FWD_KEYS}
    if LITELLM_MASTER_KEY:
        headers = {k: v for k, v in headers.items() if k.lower() != "authorization"}
        headers["authorization"] = f"Bearer {LITELLM_MASTER_KEY}"
    return headers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger("codex.proxy")

app = FastAPI()

# One long-lived client. Codex re-tries fast; per-request AsyncClient was
# running into ephemeral-port / pool exhaustion.
_CLIENT: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=30.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )
    return _CLIENT


# ---------- request: responses → chat/completions ---------------------------


def _content_to_text(content: Any) -> str:
    """Flatten a responses-API content list (`[{type:..., text:...}, ...]`)
    into a single string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if not isinstance(p, dict):
                continue
            if p.get("type") in ("input_text", "output_text", "text"):
                parts.append(p.get("text", ""))
        return "".join(parts)
    return ""


def _tool_to_chat(tool: Any) -> Optional[Dict[str, Any]]:
    """Translate a responses tool spec to chat/completions shape.

    responses: {type: "function", name, description, parameters}
    chat:      {type: "function", function: {name, description, parameters}}

    Drops non-function variants (local_shell, custom_tool, file_search, ...)
    that DeepSeek doesn't accept.
    """
    if not isinstance(tool, dict):
        return None
    if tool.get("type") != "function":
        return None
    name = tool.get("name")
    description = tool.get("description", "")
    parameters = tool.get("parameters", {})
    if name is None and isinstance(tool.get("function"), dict):
        fn = tool["function"]
        name = fn.get("name")
        description = fn.get("description", description)
        parameters = fn.get("parameters", parameters)
    if not name:
        return None
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _responses_to_chat(body: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a /v1/responses request body to /v1/chat/completions."""
    chat: Dict[str, Any] = {"model": body.get("model")}
    messages: List[Dict[str, Any]] = []

    instr = body.get("instructions")
    if isinstance(instr, str) and instr:
        messages.append({"role": "system", "content": instr})

    items = body.get("input")
    if isinstance(items, str):
        messages.append({"role": "user", "content": items})
        items = []
    if not isinstance(items, list):
        items = []

    # Buffer consecutive function_calls so codex's parallel-tool-call
    # batches collapse to a single assistant message with multiple
    # tool_calls. Without this, [fc(A), fc(B), fco(A), fco(B)] would
    # translate to [a.tc[A], a.tc[B], t(A), t(B)] — A is unpaired before
    # B and DeepSeek rejects with "tool_calls must be followed by tool
    # messages".
    pending_tool_calls: List[Dict[str, Any]] = []

    # Reasoning round-trip: DeepSeek's v4-flash (thinking mode) requires
    # `reasoning_content` on every assistant message in history. A
    # single chat turn can produce BOTH text and tool_calls (codex
    # splits them into separate input items, but they share the same
    # reasoning), so the stash is "sticky" — set when a `reasoning` item
    # arrives, applied to all subsequent assistant outputs, and cleared
    # only on a turn boundary (tool result / user / system message) or
    # the next `reasoning` item.
    reasoning_stash: Dict[str, str] = {"text": ""}

    def _peek_reasoning() -> str:
        return reasoning_stash["text"]

    def _clear_reasoning() -> None:
        reasoning_stash["text"] = ""

    def _flush_tool_calls():
        if not pending_tool_calls:
            return
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": list(pending_tool_calls),
        }
        rc = _peek_reasoning()
        if rc:
            msg["reasoning_content"] = rc
        messages.append(msg)
        pending_tool_calls.clear()

    for item in items:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t == "function_call":
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name", "")
            args = item.get("arguments", "")
            if isinstance(args, (dict, list)):
                args = json.dumps(args)
            pending_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args or ""},
                }
            )
            continue

        # Any non-function_call item closes the pending batch.
        _flush_tool_calls()

        if t == "message":
            role = item.get("role", "user")
            text = _content_to_text(item.get("content"))
            # OpenAI's responses-API uses role="developer" as a lower-
            # priority system message. DeepSeek's chat API only knows
            # system/user/assistant/tool — coerce.
            if role == "developer":
                role = "system"
            msg: Dict[str, Any] = {"role": role, "content": text}
            if role == "assistant":
                rc = _peek_reasoning()
                if rc:
                    msg["reasoning_content"] = rc
            else:
                # Non-assistant message ends the current turn for the
                # purpose of reasoning_content stickiness.
                _clear_reasoning()
            messages.append(msg)
        elif t == "function_call_output":
            call_id = item.get("call_id") or item.get("id")
            out = item.get("output", "")
            if not isinstance(out, str):
                out = json.dumps(out)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": out,
                }
            )
            # Tool result ends the assistant turn.
            _clear_reasoning()
        elif t == "reasoning":
            # Stash the reasoning text so the next assistant message can
            # carry it as `reasoning_content`. Required by DeepSeek
            # v4-flash thinking mode. Walks both `summary` (text variant)
            # and `content` (some OpenAI shapes).
            parts: List[str] = []
            for s in item.get("summary") or []:
                if isinstance(s, dict) and s.get("type") in (
                    "summary_text",
                    "text",
                ):
                    parts.append(s.get("text", ""))
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") in (
                    "reasoning_text",
                    "text",
                ):
                    parts.append(c.get("text", ""))
            text = "".join(parts).strip()
            # New reasoning item begins a new turn; replace any prior
            # stash (each chat turn carries its own reasoning).
            reasoning_stash["text"] = text
            continue
        else:
            # Unknown — best effort: pull any text out.
            txt = _content_to_text(item.get("content"))
            if txt:
                messages.append({"role": item.get("role", "user"), "content": txt})

    _flush_tool_calls()

    # Pairing safety net. For every assistant.tool_calls message, ensure
    # each tool_call_id has a following tool message before the next
    # non-tool message. Inject a placeholder for any orphan — codex
    # sometimes drops a function_call_output (timeout, interruption,
    # crash) and DeepSeek rejects the whole turn if any pairing is broken.
    repaired: List[Dict[str, Any]] = []
    n_injected = 0
    i = 0
    while i < len(messages):
        m = messages[i]
        repaired.append(m)
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids_needed = [tc.get("id") for tc in m["tool_calls"]]
            ids_seen: set = set()
            j = i + 1
            while (
                j < len(messages)
                and messages[j].get("role") == "tool"
                and messages[j].get("tool_call_id") in ids_needed
            ):
                ids_seen.add(messages[j]["tool_call_id"])
                repaired.append(messages[j])
                j += 1
            for cid in ids_needed:
                if cid not in ids_seen:
                    repaired.append(
                        {
                            "role": "tool",
                            "tool_call_id": cid,
                            "content": "(no output captured)",
                        }
                    )
                    n_injected += 1
            i = j
        else:
            i += 1
    if n_injected:
        logger.warning(
            "pairing safety net: injected %d placeholder tool message(s)", n_injected
        )

    chat["messages"] = repaired

    tools_in = body.get("tools")
    if isinstance(tools_in, list):
        tools_out = [t for t in (_tool_to_chat(x) for x in tools_in) if t]
        if tools_out:
            chat["tools"] = tools_out

    tc = body.get("tool_choice")
    if isinstance(tc, str):
        chat["tool_choice"] = tc
    elif isinstance(tc, dict) and tc.get("type") == "function":
        name = tc.get("name") or (tc.get("function") or {}).get("name")
        if name:
            chat["tool_choice"] = {"type": "function", "function": {"name": name}}

    for k in ("temperature", "top_p", "stop", "parallel_tool_calls"):
        if k in body:
            chat[k] = body[k]
    if "max_output_tokens" in body:
        chat["max_tokens"] = body["max_output_tokens"]
    elif "max_tokens" in body:
        chat["max_tokens"] = body["max_tokens"]

    # We synthesize SSE ourselves; ask upstream for a complete response.
    chat["stream"] = False
    return chat


# ---------- response: chat/completions → responses SSE ----------------------


def _now() -> int:
    return int(time.time())


def _sse(event: str, data: Dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8")


def _chat_to_response_obj(
    chat_resp: Dict[str, Any], model: str, request_id: str
) -> Dict[str, Any]:
    """Build the final `response` object from a chat completion."""
    choice = (chat_resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content") or ""
    tool_calls = message.get("tool_calls") or []
    usage = chat_resp.get("usage") or {}

    output: List[Dict[str, Any]] = []
    # Emit a reasoning item BEFORE the message/function_call so codex
    # stores it and includes it in the next turn's input items. Required
    # for DeepSeek v4-flash thinking mode — the reasoning_content must
    # be passed back on every multi-turn request.
    if reasoning_content:
        output.append(
            {
                "id": f"rs_{uuid.uuid4().hex[:24]}",
                "type": "reasoning",
                "status": "completed",
                "summary": [{"type": "summary_text", "text": reasoning_content}],
            }
        )
    if content:
        output.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": content, "annotations": []}
                ],
            }
        )
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") or {}
        output.append(
            {
                "id": f"fc_{uuid.uuid4().hex[:24]}",
                "type": "function_call",
                "status": "completed",
                "call_id": tc.get("id"),
                "name": fn.get("name", ""),
                "arguments": fn.get("arguments", "") or "",
            }
        )

    # Propagate per-call token usage so codex's turn.completed aggregates
    # correctly. DeepSeek puts reasoning tokens under
    # completion_tokens_details.reasoning_tokens; cache hit info under
    # prompt_tokens_details.cached_tokens.
    completion_details = usage.get("completion_tokens_details") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    response_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "input_tokens_details": {
            "cached_tokens": prompt_details.get("cached_tokens", 0) or 0,
        },
        "output_tokens_details": {
            "reasoning_tokens": completion_details.get("reasoning_tokens", 0) or 0,
        },
    }

    return {
        "id": request_id,
        "object": "response",
        "created_at": _now(),
        "status": "completed",
        "model": model,
        "output": output,
        "usage": response_usage,
    }


def _build_sse_events(response_obj: Dict[str, Any]) -> List[bytes]:
    """Synthesize the SSE event sequence codex expects from a complete
    response object."""
    in_progress = {**response_obj, "status": "in_progress", "output": [], "usage": None}
    events: List[bytes] = []
    events.append(
        _sse("response.created", {"type": "response.created", "response": in_progress})
    )
    events.append(
        _sse(
            "response.in_progress",
            {"type": "response.in_progress", "response": in_progress},
        )
    )

    for idx, item in enumerate(response_obj["output"]):
        if item["type"] == "reasoning":
            text = ""
            for s in item.get("summary") or []:
                if isinstance(s, dict) and s.get("type") == "summary_text":
                    text += s.get("text", "")
            rs_in_progress = {
                "id": item["id"],
                "type": "reasoning",
                "status": "in_progress",
                "summary": [],
            }
            events.append(
                _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": idx,
                        "item": rs_in_progress,
                    },
                )
            )
            events.append(
                _sse(
                    "response.reasoning_summary_part.added",
                    {
                        "type": "response.reasoning_summary_part.added",
                        "item_id": item["id"],
                        "output_index": idx,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""},
                    },
                )
            )
            events.append(
                _sse(
                    "response.reasoning_summary_text.delta",
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": item["id"],
                        "output_index": idx,
                        "summary_index": 0,
                        "delta": text,
                    },
                )
            )
            events.append(
                _sse(
                    "response.reasoning_summary_text.done",
                    {
                        "type": "response.reasoning_summary_text.done",
                        "item_id": item["id"],
                        "output_index": idx,
                        "summary_index": 0,
                        "text": text,
                    },
                )
            )
            events.append(
                _sse(
                    "response.reasoning_summary_part.done",
                    {
                        "type": "response.reasoning_summary_part.done",
                        "item_id": item["id"],
                        "output_index": idx,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": text},
                    },
                )
            )
            events.append(
                _sse(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": idx,
                        "item": item,
                    },
                )
            )
        elif item["type"] == "message":
            text = item["content"][0]["text"] if item.get("content") else ""
            msg_in_progress = {
                "id": item["id"],
                "type": "message",
                "status": "in_progress",
                "role": "assistant",
                "content": [],
            }
            events.append(
                _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": idx,
                        "item": msg_in_progress,
                    },
                )
            )
            events.append(
                _sse(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": item["id"],
                        "output_index": idx,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                )
            )
            events.append(
                _sse(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "item_id": item["id"],
                        "output_index": idx,
                        "content_index": 0,
                        "delta": text,
                    },
                )
            )
            events.append(
                _sse(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": item["id"],
                        "output_index": idx,
                        "content_index": 0,
                        "text": text,
                    },
                )
            )
            events.append(
                _sse(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": item["id"],
                        "output_index": idx,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": text,
                            "annotations": [],
                        },
                    },
                )
            )
            events.append(
                _sse(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": idx,
                        "item": item,
                    },
                )
            )
        elif item["type"] == "function_call":
            args = item.get("arguments", "") or ""
            fc_in_progress = {
                "id": item["id"],
                "type": "function_call",
                "status": "in_progress",
                "call_id": item.get("call_id"),
                "name": item.get("name", ""),
                "arguments": "",
            }
            events.append(
                _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": idx,
                        "item": fc_in_progress,
                    },
                )
            )
            events.append(
                _sse(
                    "response.function_call_arguments.delta",
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": item["id"],
                        "output_index": idx,
                        "delta": args,
                    },
                )
            )
            events.append(
                _sse(
                    "response.function_call_arguments.done",
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": item["id"],
                        "output_index": idx,
                        "arguments": args,
                    },
                )
            )
            events.append(
                _sse(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": idx,
                        "item": item,
                    },
                )
            )

    events.append(
        _sse(
            "response.completed",
            {"type": "response.completed", "response": response_obj},
        )
    )
    return events


# ---------- HTTP routes ------------------------------------------------------


@app.get("/health")
async def health():
    return {"ok": True, "upstream": UPSTREAM}


@app.post("/v1/responses")
async def responses(request: Request):
    # Tag every log line for this request with the same request_id so a
    # post-processor can pair the "start" / "upstream_returned" / "sent"
    # events. The task_id is set by codex via the X-Codex-Task-Id header
    # (config.toml env_http_headers → CODEX_TASK_ID env), and tells us
    # which task each LLM call belongs to.
    request_uid = uuid.uuid4().hex[:12]
    task_id = request.headers.get("x-codex-task-id", "-")
    t_start = time.time()

    raw = await request.body()
    try:
        body = json.loads(raw)
    except Exception as e:
        return JSONResponse({"error": {"message": f"bad json: {e}"}}, status_code=400)

    model = body.get("model", "")
    requested_stream = bool(body.get("stream", True))  # codex defaults to streaming
    chat_body = _responses_to_chat(body)
    n_msgs = len(chat_body.get("messages") or [])
    n_tools = len(chat_body.get("tools") or [])
    n_tool_calls = sum(1 for m in chat_body.get("messages", []) if m.get("tool_calls"))
    n_tool_outs = sum(
        1 for m in chat_body.get("messages", []) if m.get("role") == "tool"
    )
    logger.info(
        "REQ_START req=%s task=%s model=%s msgs=%d tools=%d tool_calls=%d tool_outs=%d stream(req=%s)",
        request_uid,
        task_id,
        model,
        n_msgs,
        n_tools,
        n_tool_calls,
        n_tool_outs,
        requested_stream,
    )

    headers = _upstream_headers(request)
    headers["content-type"] = "application/json"

    url = f"{UPSTREAM.rstrip('/')}/v1/chat/completions"
    client = _get_client()
    t_upstream_start = time.time()
    try:
        upstream = await client.post(
            url, headers=headers, content=json.dumps(chat_body).encode("utf-8")
        )
    except Exception as e:
        logger.exception(
            "upstream chat/completions request failed (req=%s)", request_uid
        )
        return JSONResponse(
            {"error": {"message": f"proxy upstream error: {e}"}}, status_code=502
        )

    if upstream.status_code >= 400:
        logger.warning(
            "upstream %s -> %d: %s", url, upstream.status_code, upstream.text[:400]
        )
        # Dump the failing chat body to /tmp so we can see exactly what went
        # to upstream (truncated body fields kept reasonable for log volume).
        try:
            import os as _os, time as _time

            _dump_path = (
                f"/tmp/codex_proxy_400_{int(_time.time()*1000)}_"
                f"{_os.getpid()}_{uuid.uuid4().hex[:6]}.json"
            )
            with open(_dump_path, "w") as _f:
                json.dump(
                    {
                        "upstream_status": upstream.status_code,
                        "upstream_body": upstream.text,
                        "chat_body_sent": chat_body,
                    },
                    _f,
                    indent=2,
                )
            logger.warning("dumped failing body to %s", _dump_path)
        except Exception as _e:
            logger.warning("failed to dump 400 body: %s", _e)
        if requested_stream:
            error_event = {
                "type": "response.failed",
                "response": {
                    "id": f"resp_{uuid.uuid4().hex[:24]}",
                    "status": "failed",
                    "error": {
                        "code": "upstream_error",
                        "message": upstream.text,
                    },
                },
            }
            return StreamingResponse(
                iter([_sse("response.failed", error_event)]),
                media_type="text/event-stream",
                status_code=upstream.status_code,
            )
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            media_type="application/json",
        )

    try:
        chat_resp = upstream.json()
    except Exception:
        logger.exception("upstream chat returned non-JSON: %s", upstream.text[:400])
        return Response(
            content=upstream.content, status_code=502, media_type="application/json"
        )

    request_id = f"resp_{uuid.uuid4().hex[:24]}"
    response_obj = _chat_to_response_obj(chat_resp, model=model, request_id=request_id)

    # Per-call usage log line (single line, structured) so we can post-
    # process to get authoritative token totals + cache hit rate. Tagged
    # with the same req= / task= as REQ_START / REQ_SENT for pairing.
    u = chat_resp.get("usage") or {}
    cdetail = u.get("completion_tokens_details") or {}
    pdetail = u.get("prompt_tokens_details") or {}
    t_upstream_end = time.time()
    logger.info(
        "REQ_USAGE req=%s task=%s model=%s prompt=%d completion=%d reasoning=%d cached=%d total=%d "
        "upstream_delay=%.4f",
        request_uid,
        task_id,
        chat_resp.get("model", model),
        u.get("prompt_tokens", 0) or 0,
        u.get("completion_tokens", 0) or 0,
        cdetail.get("reasoning_tokens", 0) or 0,
        pdetail.get("cached_tokens", 0) or 0,
        u.get("total_tokens", 0) or 0,
        t_upstream_end - t_upstream_start,
    )

    # response_id (resp_xxx) is what codex sees. Log it so post-processors
    # can cross-reference codex's --json stream with our proxy log.
    response_id_emitted = response_obj.get("id", "")
    logger.info(
        "REQ_SENT req=%s task=%s response_id=%s total_delay=%.4f",
        request_uid,
        task_id,
        response_id_emitted,
        time.time() - t_start,
    )

    if not requested_stream:
        return JSONResponse(response_obj)

    events = _build_sse_events(response_obj)

    async def gen():
        for ev in events:
            yield ev

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Pass chat/completions through, stripping non-function tools."""
    raw = await request.body()
    try:
        body = json.loads(raw)
        tools = body.get("tools")
        if isinstance(tools, list):
            kept = [
                t for t in tools if isinstance(t, dict) and t.get("type") == "function"
            ]
            if len(kept) != len(tools):
                logger.info(
                    "chat: stripped %d non-function tools, kept %d",
                    len(tools) - len(kept),
                    len(kept),
                )
                body["tools"] = kept
            raw = json.dumps(body).encode("utf-8")
    except Exception:
        pass

    headers = _upstream_headers(request)
    url = f"{UPSTREAM.rstrip('/')}/v1/chat/completions"
    client = _get_client()
    upstream = await client.post(url, headers=headers, content=raw)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def passthrough(path: str, request: Request):
    """Generic passthrough for any other path / verb."""
    raw = await request.body()
    headers = _upstream_headers(request)
    url = f"{UPSTREAM.rstrip('/')}/{path}"
    client = _get_client()
    upstream = await client.request(request.method, url, headers=headers, content=raw)
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/octet-stream"),
    )


if __name__ == "__main__":
    import uvicorn

    logger.info("codex-proxy listening on :%d -> %s", PORT, UPSTREAM)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
