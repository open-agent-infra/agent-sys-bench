"""OpenAI-compatible LLM proxy with fast-service instrumentation.

This proxy sits between openclaw and an upstream OpenAI-compatible API (the
default upstream is DeepSeek at https://api.deepseek.com). openclaw is
configured to point its `baseUrl` at this proxy, so every LLM call passes
through here and is recorded with per-call telemetry that chat.jsonl cannot
expose: server-side wall-clock latency, prefill-vs-decode split (when the
upstream reports it), upstream request id, request/response bytes.

Architecture:
- A FastAPI app exposes the OpenAI-compatible endpoints openclaw expects
  (`/v1/chat/completions`, `/v1/models`).
- Inside each request handler, the work is decomposed into three
  fast-service-decorated functions running in `local_function` mode:
    1. `parse_request` — pydantic-validate + extract metadata
    2. `forward_to_upstream` — httpx POST to the real upstream
    3. `record_per_call_usage` — emit a per-call JSON line to llm_calls.jsonl
       (token counts, costs, timing, openclaw session id when present)
- fast-service writes its standard `e2e_delay.csv` / `remote_delay.csv` for
  the same three functions, so we can cross-check the JSON-line audit log
  against the canonical fast-service telemetry.

Usage:

    DEEPSEEK_API_KEY=$(cat /data/agent-sys-bench \
    python -m agsbench_openclaw.llm_proxy \
        --config /data/agent-sys-bench \
        --port 1801

The API key is held server-side; openclaw should be configured with a dummy
key so secrets never leak into chat.jsonl.
"""

"""NOTE: do NOT add `from __future__ import annotations` to this module.

fast-service inspects `__annotations__` to build cls_dict by class object.
With PEP 563 deferred annotations, that dict contains *strings* instead of
classes, and ContextParamUsage.use_context() silently returns False — the
MonitorLayer is then skipped and no telemetry is recorded.
"""

import json
import os
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from fast_service import (
    FastServiceConfig,
    FastServiceManager,
    RequestContext,
)


# ----- fast-service registration ---------------------------------------------
LLM_PROXY_FSM = FastServiceManager()


class ParsedRequest(BaseModel):
    upstream_url: str
    method: str
    headers: dict
    body: dict
    request_id: str
    openclaw_session_id: str | None = None
    stream: bool = False
    model: str
    n_messages: int
    prompt_chars: int

    class Config:
        arbitrary_types_allowed = True


class UpstreamResponse(BaseModel):
    status_code: int
    body: dict
    headers: dict
    upstream_id: str | None = None
    is_stream: bool = False

    class Config:
        arbitrary_types_allowed = True


class UsageRecord(BaseModel):
    timestamp: float
    request_id: str
    openclaw_session_id: str | None
    model: str
    upstream_id: str | None
    status_code: int
    total_ms: float
    parse_ms: float
    upstream_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cache_read_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    request_bytes: int
    response_bytes: int
    finish_reason: str | None
    error: str | None
    # Free-form per-task marker set by the driver via POST /admin/tag, so a
    # long-running proxy can split its single audit log into per-task slices
    # without restarts. Null until/unless a tag is active.
    task_tag: str | None = None


# Module-level current tag. Set by POST /admin/tag, read at every record_per_call_usage.
# Mutating from FastAPI handlers is safe — single uvicorn worker, async event loop.
_CURRENT_TAG: str | None = None


# DeepSeek-V4-Flash (and other reasoning models) require the assistant's
# `reasoning_content` from the previous turn to be echoed back on follow-up
# requests:
#   HTTP 400  "The reasoning_content in the thinking mode must be passed
#              back to the API."
# Most OpenAI-compatible clients (claw-eval, openclaw's default OpenAI
# wrapper) strip non-standard fields like `reasoning_content`, so multi-turn
# tool-using flows break the moment any tool result follows an assistant
# turn that emitted tool_calls. Repro: openclaw/scripts/repros/
# deepseek_thinking_tool_empty_choices.py — case D fails, case E (with the
# reasoning echoed back) succeeds.
#
# We fix it at the proxy: after each upstream response with thinking on,
# stash `reasoning_content` keyed by every tool_call_id that turn emitted.
# On the next inbound request, walk the assistant messages: any that have
# `tool_calls` but no `reasoning_content` get the stashed value reinjected.
# Bounded to avoid unbounded growth across long sessions.
_REASONING_STASH: dict[str, str] = {}
_REASONING_STASH_ORDER: list[str] = []
_REASONING_STASH_MAX = 2000


def _stash_reasoning_for_tool_calls(tool_call_ids: list[str], reasoning: str) -> None:
    if not reasoning:
        return
    for tcid in tool_call_ids:
        if not tcid:
            continue
        if tcid in _REASONING_STASH:
            # Refresh ordering: pop then re-append so we evict oldest first.
            _REASONING_STASH_ORDER.remove(tcid)
        elif len(_REASONING_STASH_ORDER) >= _REASONING_STASH_MAX:
            old = _REASONING_STASH_ORDER.pop(0)
            _REASONING_STASH.pop(old, None)
        _REASONING_STASH[tcid] = reasoning
        _REASONING_STASH_ORDER.append(tcid)


def _reinject_reasoning_into_messages(messages: list) -> int:
    """For each assistant message with tool_calls but no reasoning_content,
    look up the stash by its first tool_call_id and inject. Mutates in place.
    Returns the number of messages we patched, for telemetry."""
    patched = 0
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        if m.get("reasoning_content"):
            continue
        tool_calls = m.get("tool_calls") or []
        if not tool_calls:
            continue
        first_id = (tool_calls[0] or {}).get("id")
        if first_id and first_id in _REASONING_STASH:
            m["reasoning_content"] = _REASONING_STASH[first_id]
            patched += 1
    return patched


# NOTE: parse_request / forward_to_upstream / record_per_call_usage are
# marked `monitor_only=True` because their argument types include `bytes`
# and FastAPI Headers objects that aren't safely JSON-serializable by
# fast-service's IntermediateLayer (it hits "Circular reference" while
# walking the args). Instead we add a separate `archive_messages` function
# below whose args are plain dicts (the OpenAI request body and the
# upstream response body) — that function IS subject to intermediate
# storage, so the prompt + completion text for every LLM call lands on
# disk under .cache/openclaw/llm_proxy/intermediate/requests|responses/.
@LLM_PROXY_FSM.fast_service(monitor_only=True)
def parse_request(
    method: str,
    path: str,
    headers: dict,
    body_bytes: bytes,
    upstream_base_url: str,
    fsm_context: RequestContext = None,
) -> ParsedRequest:
    """Parse the incoming OpenAI-style request. Strip auth, attach our key."""
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        body = {}

    # Try to find an openclaw session marker in headers or body
    session_id = (
        headers.get("x-openclaw-session-id")
        or headers.get("x-session-id")
        or body.get("openclaw_session_id")
    )

    messages = body.get("messages") or []
    # Re-inject reasoning_content into assistant tool_call turns whose
    # client-side echo lost it. Only meaningful when thinking is enabled;
    # cheap no-op when nothing in the stash matches. We do NOT decorate
    # the outgoing body with a "patched count" field since DeepSeek may
    # reject unknown top-level keys — the side-effect on `messages` is
    # what reaches upstream.
    thinking_on = (body.get("thinking") or {}).get("type") == "enabled"
    if thinking_on:
        _reinject_reasoning_into_messages(messages)
    prompt_chars = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            prompt_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict):
                    prompt_chars += len(str(part.get("text") or ""))

    # Rebuild upstream URL: <upstream_base_url> + path[len("/v1"):]
    # If the upstream base already ends in /v1, drop the leading /v1 from path
    suffix = path
    base = upstream_base_url.rstrip("/")
    if base.endswith("/v1") and suffix.startswith("/v1"):
        suffix = suffix[3:]
    upstream_url = base + suffix

    # Replace Authorization header with our server-side key
    upstream_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get(
        "UPSTREAM_API_KEY"
    )
    new_headers = {
        k: v
        for k, v in headers.items()
        if k.lower()
        not in {
            "host",
            "content-length",
            "authorization",
            "x-forwarded-for",
            "x-forwarded-proto",
        }
    }
    if upstream_key:
        new_headers["Authorization"] = f"Bearer {upstream_key}"

    return ParsedRequest(
        upstream_url=upstream_url,
        method=method,
        headers=new_headers,
        body=body,
        request_id=str(uuid.uuid4()),
        openclaw_session_id=session_id,
        stream=bool(body.get("stream")),
        model=str(body.get("model") or "unknown"),
        n_messages=len(messages),
        prompt_chars=prompt_chars,
    )


@LLM_PROXY_FSM.fast_service(monitor_only=True)
def forward_to_upstream(
    req: ParsedRequest,
    upstream_timeout_s: float = 300.0,
    fsm_context: RequestContext = None,
) -> UpstreamResponse:
    """POST/GET to the upstream OpenAI-compatible endpoint and return the response.

    Two openclaw compatibility quirks handled here:
    1. Streaming requests are forwarded as streaming and merged into a single
       non-streamed JSON before returning (openclaw expects either a SSE
       stream or a full JSON; since FastAPI's JSONResponse doesn't stream,
       we always coalesce to JSON).
    2. DeepSeek-V4-Flash returns assistant messages with empty `content` and
       the actual answer in `reasoning_content`. openclaw treats empty
       content as "incomplete turn". We fold reasoning_content into content
       when content is empty, so openclaw sees a normal response.
    """
    is_stream = bool(req.body.get("stream"))

    with httpx.Client(timeout=upstream_timeout_s) as client:
        if req.method.upper() == "GET":
            r = client.get(req.upstream_url, headers=req.headers)
            try:
                body = r.json() if r.content else {}
            except Exception:
                body = {"_raw": r.text[:4000]}
            return UpstreamResponse(
                status_code=r.status_code,
                body=body,
                headers=dict(r.headers),
                upstream_id=body.get("id") if isinstance(body, dict) else None,
                is_stream=False,
            )

        # POST path. If streaming, coalesce SSE chunks into a single response.
        if is_stream:
            # Force non-streaming upstream call to get a clean JSON.
            # Also strip stream_options since DeepSeek rejects it without stream=true.
            non_stream_body = {
                k: v
                for k, v in req.body.items()
                if k not in ("stream", "stream_options")
            }
            non_stream_body["stream"] = False
            r = client.request(
                req.method,
                req.upstream_url,
                headers=req.headers,
                json=non_stream_body,
            )
        else:
            r = client.request(
                req.method,
                req.upstream_url,
                headers=req.headers,
                json=req.body,
            )

    try:
        body = r.json() if r.content else {}
    except Exception:
        body = {"_raw": r.text[:4000]}

    # Fold reasoning_content into content where content is empty, and
    # stash reasoning_content for any tool_calls so we can re-inject on the
    # client's follow-up request (DeepSeek thinking-mode contract).
    if isinstance(body, dict):
        for ch in body.get("choices") or []:
            msg = ch.get("message") if isinstance(ch, dict) else None
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            reasoning = msg.get("reasoning_content")
            tool_calls = msg.get("tool_calls") or []
            if reasoning and tool_calls:
                tcids = [(tc or {}).get("id") for tc in tool_calls]
                _stash_reasoning_for_tool_calls(tcids, reasoning)
            if (not content) and reasoning:
                msg["content"] = reasoning
                msg["_proxy_folded_reasoning"] = True

    return UpstreamResponse(
        status_code=r.status_code,
        body=body,
        headers=dict(r.headers),
        upstream_id=body.get("id") if isinstance(body, dict) else None,
        is_stream=is_stream,
    )


@LLM_PROXY_FSM.fast_service(monitor_only=True)
def record_per_call_usage(
    rec: UsageRecord,
    log_path: str,
    fsm_context: RequestContext = None,
) -> None:
    """Append a JSON-line usage record to the per-call audit log."""
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(rec.model_dump_json() + "\n")


class ArchivedExchange(BaseModel):
    """Return type for archive_messages — fast-service requires pydantic."""

    messages: list = []
    response_message: dict = {}
    usage: dict = {}


@LLM_PROXY_FSM.fast_service
def archive_messages(
    request_body: dict,
    response_body: dict,
    fsm_context: RequestContext = None,
) -> ArchivedExchange:
    """Pure pass-through whose only purpose is to be subject to
    fast-service's IntermediateLayer.

    Both arguments are plain JSON-safe dicts (the OpenAI-style request
    body — which contains the `messages` array and `tools` schema — and
    the upstream response body — which contains the `choices[].message`
    with `content` / `reasoning_content` / `tool_calls`). The
    intermediate-storage layer writes args and returns to disk per
    request_id, so after a run we have a complete log of every prompt
    and every completion exchanged with the upstream LLM under
    intermediate/requests/<request_id>/<function>.json and
    intermediate/responses/<request_id>/<function>.json.

    The return value is also persisted (into responses/), so the
    OpenAI-style message object is trivially recoverable from either
    side.
    """
    choices = response_body.get("choices") or []
    first_choice_msg = (choices[0].get("message") if choices else {}) or {}
    return ArchivedExchange(
        messages=request_body.get("messages") or [],
        response_message=first_choice_msg,
        usage=response_body.get("usage") or {},
    )


async def _stream_proxy(
    parsed: "ParsedRequest",
    body_bytes: bytes,
    t0: float,
    t_parse_done: float,
    usage_log_path: str,
) -> StreamingResponse:
    """Forward an upstream SSE stream to the client verbatim, while still
    capturing the final usage chunk for our audit log.

    Why this exists: openclaw's `--local` mode is happy with either a
    streamed or non-streamed response and we coalesce to JSON for it.
    claw-eval, on the other hand, calls the OpenAI SDK with stream=True
    and parses the response chunk-by-chunk; if we hand back a plain
    JSONResponse it sees zero stream deltas and bails with "empty
    choices". So whenever the client asked for streaming, we honor it.

    We also inject `stream_options.include_usage = true` if the client
    didn't, so DeepSeek emits a final SSE event with the usage object
    (otherwise streamed responses have no usage block at all).
    """
    upstream_body = dict(parsed.body)
    so = upstream_body.get("stream_options") or {}
    if not so.get("include_usage"):
        so = {**so, "include_usage": True}
        upstream_body["stream_options"] = so

    upstream_id = None
    last_usage: dict | None = None
    last_finish: str | None = None
    accumulated_content: list[str] = []
    accumulated_reasoning: list[str] = []
    # SSE delta.tool_calls is an indexed array: ids arrive in the first
    # chunk that mentions a given index, then later chunks append to the
    # same index's function.arguments. We only need the ids for the
    # reasoning_content stash, so we keep a dict[index]->id.
    tool_call_ids: dict[int, str] = {}

    # Open the upstream stream before committing to a 200 SSE response, so
    # upstream errors (4xx/5xx JSON bodies, e.g. rate limits or content
    # moderation) reach the client with their real status instead of being
    # tee'd into an "SSE" stream that ends without a finish_reason.
    client = httpx.AsyncClient(timeout=300.0)
    try:
        r = await client.send(
            client.build_request(
                "POST",
                parsed.upstream_url,
                headers=parsed.headers,
                json=upstream_body,
            ),
            stream=True,
        )
    except Exception as e:  # noqa: BLE001
        await client.aclose()
        r = None
        err_status, err_body = 502, {"error": str(e)}
    else:
        if r.status_code >= 400:
            raw = await r.aread()
            await r.aclose()
            await client.aclose()
            try:
                err_body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                err_body = {"_raw": raw[:4000].decode("utf-8", errors="replace")}
            err_status = r.status_code
            r = None
    if r is None:
        t_done = time.time()
        rec = UsageRecord(
            timestamp=t_done,
            request_id=parsed.request_id,
            openclaw_session_id=parsed.openclaw_session_id,
            model=parsed.model,
            upstream_id=None,
            status_code=err_status,
            total_ms=(t_done - t0) * 1000.0,
            parse_ms=(t_parse_done - t0) * 1000.0,
            upstream_ms=(t_done - t_parse_done) * 1000.0,
            prompt_tokens=None,
            completion_tokens=None,
            cache_read_tokens=None,
            reasoning_tokens=None,
            total_tokens=None,
            request_bytes=len(body_bytes),
            response_bytes=len(json.dumps(err_body)),
            finish_reason=None,
            error=json.dumps(err_body)[:2000],
            task_tag=_CURRENT_TAG,
        )
        try:
            record_per_call_usage(rec=rec, log_path=usage_log_path)
        except Exception:
            pass
        return JSONResponse(content=err_body, status_code=err_status)

    async def iter_stream():
        nonlocal upstream_id, last_usage, last_finish
        try:
            async for chunk in r.aiter_bytes():
                if not chunk:
                    continue
                # Best-effort parse of SSE events for instrumentation.
                # We split on \n\n and look for `data: {...}` lines; we
                # always pass the raw bytes through unchanged.
                try:
                    for ev in chunk.decode("utf-8", errors="replace").split("\n\n"):
                        if not ev.startswith("data: "):
                            continue
                        payload = ev[len("data: ") :].strip()
                        if payload == "[DONE]":
                            continue
                        obj = json.loads(payload)
                        if not upstream_id and obj.get("id"):
                            upstream_id = obj["id"]
                        if obj.get("usage"):
                            last_usage = obj["usage"]
                        for ch in obj.get("choices") or []:
                            if ch.get("finish_reason"):
                                last_finish = ch["finish_reason"]
                            d = ch.get("delta") or {}
                            if d.get("content"):
                                accumulated_content.append(d["content"])
                            if d.get("reasoning_content"):
                                accumulated_reasoning.append(d["reasoning_content"])
                            for tc in d.get("tool_calls") or []:
                                if not isinstance(tc, dict):
                                    continue
                                idx = tc.get("index", 0)
                                if tc.get("id"):
                                    tool_call_ids[idx] = tc["id"]
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
                yield chunk
        finally:
            await r.aclose()
            await client.aclose()

    async def with_audit():
        try:
            async for chunk in iter_stream():
                yield chunk
        finally:
            t_done = time.time()
            usage = last_usage or {}
            prompt_details = usage.get("prompt_tokens_details") or {}
            cache_read = prompt_details.get("cached_tokens")
            completion_details = usage.get("completion_tokens_details") or {}
            reasoning = completion_details.get("reasoning_tokens")
            content = "".join(accumulated_content)
            reasoning_text = "".join(accumulated_reasoning)
            response_bytes = len(content) + len(reasoning_text)
            rec = UsageRecord(
                timestamp=t_done,
                request_id=parsed.request_id,
                openclaw_session_id=parsed.openclaw_session_id,
                model=parsed.model,
                upstream_id=upstream_id,
                status_code=200,
                total_ms=(t_done - t0) * 1000.0,
                parse_ms=(t_parse_done - t0) * 1000.0,
                upstream_ms=(t_done - t_parse_done) * 1000.0,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                cache_read_tokens=cache_read,
                reasoning_tokens=reasoning,
                total_tokens=usage.get("total_tokens"),
                request_bytes=len(body_bytes),
                response_bytes=response_bytes,
                finish_reason=last_finish,
                error=None,
                task_tag=_CURRENT_TAG,
            )
            try:
                record_per_call_usage(rec=rec, log_path=usage_log_path)
            except Exception:
                pass
            # If this stream emitted tool_calls + reasoning_content, stash
            # the reasoning so we can re-inject on the client's next request.
            try:
                if reasoning_text and tool_call_ids:
                    _stash_reasoning_for_tool_calls(
                        list(tool_call_ids.values()),
                        reasoning_text,
                    )
            except Exception:
                pass
            try:
                archive_messages(
                    request_body=parsed.body,
                    response_body={
                        "id": upstream_id,
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": content,
                                    "reasoning_content": reasoning_text,
                                },
                                "finish_reason": last_finish,
                            }
                        ],
                        "usage": usage,
                    },
                )
            except Exception:
                pass

    return StreamingResponse(with_audit(), media_type="text/event-stream")


# ----- FastAPI app -----------------------------------------------------------
def make_app(
    upstream_base_url: str,
    usage_log_path: str,
) -> FastAPI:
    app = FastAPI(title="openclaw LLM proxy")

    @app.get("/health")
    async def health():
        return {"status": "ok", "upstream": upstream_base_url}

    @app.post("/admin/tag")
    async def set_tag(request: Request):
        """Set or clear the task_tag stamped on subsequent audit rows.

        Body: {"tag": "<string-or-null>"}. A driver calls this before each
        task so per-task audit rows are identifiable without restarting the
        proxy or rotating the log. POST with tag=null (or omitted) to clear.
        """
        global _CURRENT_TAG
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        new_tag = payload.get("tag")
        _CURRENT_TAG = str(new_tag) if new_tag else None
        return {"tag": _CURRENT_TAG}

    @app.get("/admin/tag")
    async def get_tag():
        return {"tag": _CURRENT_TAG}

    @app.api_route(
        "/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    async def proxy(path: str, request: Request):
        full_path = "/v1/" + path
        body_bytes = await request.body()
        headers = dict(request.headers)

        # Optional reasoning-effort override (set via PROXY_REASONING_EFFORT
        # env var on the proxy process). Rewrites the request body before
        # forwarding upstream. Useful when openclaw's per-model manifest
        # caps the effort but the upstream model supports a higher level
        # (e.g. DeepSeek-V4-Flash: openclaw caps at "high" but DeepSeek
        # accepts "max"). Empty = pass-through.
        effort_override = os.environ.get("PROXY_REASONING_EFFORT", "").strip()
        if effort_override and body_bytes:
            try:
                parsed_body = json.loads(body_bytes)
                if isinstance(parsed_body, dict) and "reasoning_effort" in parsed_body:
                    parsed_body["reasoning_effort"] = effort_override
                    body_bytes = json.dumps(parsed_body).encode("utf-8")
            except json.JSONDecodeError:
                pass

        t0 = time.time()
        parsed = parse_request(
            method=request.method,
            path=full_path,
            headers=headers,
            body_bytes=body_bytes,
            upstream_base_url=upstream_base_url,
        )
        t_parse_done = time.time()

        # Streaming pass-through (clients like claw-eval require true SSE).
        # We tee the upstream stream back to the client verbatim and
        # accumulate just enough state to record a usage row + a fake
        # archive entry at the end. include_usage is injected if missing
        # so DeepSeek emits a final usage chunk.
        if parsed.stream and request.method.upper() == "POST":
            return await _stream_proxy(
                parsed,
                body_bytes,
                t0,
                t_parse_done,
                usage_log_path,
            )

        try:
            upstream = forward_to_upstream(parsed)
            err = None
        except Exception as e:
            upstream = UpstreamResponse(
                status_code=502,
                body={"error": str(e)},
                headers={},
                upstream_id=None,
            )
            err = str(e)
        t_upstream_done = time.time()

        # Extract token usage from upstream response (OpenAI shape)
        usage = (upstream.body or {}).get("usage") or {}
        finish_reason = None
        choices = (upstream.body or {}).get("choices")
        if isinstance(choices, list) and choices:
            finish_reason = choices[0].get("finish_reason")

        prompt_details = usage.get("prompt_tokens_details") or {}
        cache_read = prompt_details.get("cached_tokens")
        completion_details = usage.get("completion_tokens_details") or {}
        reasoning = completion_details.get("reasoning_tokens")

        rec = UsageRecord(
            timestamp=t_upstream_done,
            request_id=parsed.request_id,
            openclaw_session_id=parsed.openclaw_session_id,
            model=parsed.model,
            upstream_id=upstream.upstream_id,
            status_code=upstream.status_code,
            total_ms=(t_upstream_done - t0) * 1000.0,
            parse_ms=(t_parse_done - t0) * 1000.0,
            upstream_ms=(t_upstream_done - t_parse_done) * 1000.0,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            cache_read_tokens=cache_read,
            reasoning_tokens=reasoning,
            total_tokens=usage.get("total_tokens"),
            request_bytes=len(body_bytes),
            response_bytes=len(json.dumps(upstream.body)) if upstream.body else 0,
            finish_reason=finish_reason,
            error=err,
            task_tag=_CURRENT_TAG,
        )
        record_per_call_usage(rec=rec, log_path=usage_log_path)

        # Persist the full prompt + completion via fast-service's
        # IntermediateLayer (only successful upstream responses, so we
        # don't fill disk with replays of error pages).
        if isinstance(upstream.body, dict) and upstream.status_code < 400:
            try:
                archive_messages(
                    request_body=parsed.body,
                    response_body=upstream.body,
                )
            except Exception:  # noqa: BLE001
                # Storage failure must never break the actual proxy response
                pass

        return JSONResponse(content=upstream.body, status_code=upstream.status_code)

    return app


def serve(
    fast_service_config_path: str,
    upstream_base_url: str,
    usage_log_path: str,
    host: str = "0.0.0.0",
    port: int = 1801,
) -> None:
    """Initialize fast-service + uvicorn and serve forever."""
    import logging

    log = logging.getLogger("agsbench_openclaw.llm_proxy")

    # Initialize fast-service in local_function mode so the decorated
    # functions execute in-process but emit telemetry to monitor_storage.
    cfg = FastServiceConfig.load_from_file(fast_service_config_path)
    log.info(
        "fast-service config: impl=%s monitor=%s storage=%s registered=%s",
        cfg.service_impl,
        cfg.monitor,
        getattr(cfg.monitor_storage, "store_dir", None),
        list(LLM_PROXY_FSM.registered_service.keys()),
    )
    LLM_PROXY_FSM.execute(cfg)
    log.info("fast-service execute() done; functions now monitored.")

    app = make_app(upstream_base_url, usage_log_path)
    uvicorn.run(app, host=host, port=port, log_level="info")


# NOTE: this module must NOT be invoked as `python -m agsbench_openclaw.llm_proxy`.
# Doing so makes `parse_request.__module__ == "__main__"`, which won't match
# the `module_name: agsbench_openclaw.llm_proxy` entries in the fast-service YAML
# config, and the rewriter would silently drop them (no instrumentation).
# Use `python scripts/serve_proxy.py` instead, which imports this module
# normally so the function `__module__` resolves to `agsbench_openclaw.llm_proxy`.
