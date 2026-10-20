"""Instrumented passthrough proxy for openclaw's native Firecrawl provider.

openclaw bundles a `firecrawl` web_search provider whose `baseUrl` is
configurable via `plugins.entries.firecrawl.config.webSearch.baseUrl`.
By default it points at https://api.firecrawl.dev/v1/search; we redirect
it through this proxy (which forwards to api.firecrawl.dev) so we can
capture per-search timing CSVs and persist every (query → results[])
pair to disk under .cache/openclaw/search_proxy/intermediate/.

Wire format: this proxy speaks the Firecrawl `/v1/search` API natively
(`{query, limit, sources?}` → `{success, data: [{url, title,
description, ...}]}`), since openclaw's stock firecrawl plugin sends
that shape unchanged. No translation needed.

DO NOT add `from __future__ import annotations` — same PEP-563 gotcha
that bites llm_proxy.py.
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
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from fast_service import (
    FastServiceConfig,
    FastServiceManager,
    RequestContext,
)


SEARCH_PROXY_FSM = FastServiceManager()
DEFAULT_FIRECRAWL_URL = "https://api.firecrawl.dev"


class ParsedSearchRequest(BaseModel):
    request_id: str
    query: str
    limit: int
    raw_body: dict

    class Config:
        arbitrary_types_allowed = True


class FirecrawlResults(BaseModel):
    request_id: str
    success: bool
    # Raw upstream body — passed back to openclaw verbatim, so the wire
    # response shape (v1: data=[]; v2: data={web:[]}) matches whatever
    # openclaw's firecrawl plugin expects for that version.
    raw_response: dict = {}
    # Flattened canonical results — list of {url, title, description|content, ...}
    # extracted from raw_response. Used for our search_calls.jsonl + archive.
    results: list = []
    upstream_id: str = ""
    upstream_ms: float = 0.0
    error: str = ""

    class Config:
        arbitrary_types_allowed = True


class SearchUsageRecord(BaseModel):
    timestamp: float
    request_id: str
    query: str
    n_results: int
    upstream_ms: float
    total_ms: float
    status_code: int
    error: str = ""


class ArchivedSearch(BaseModel):
    query: str
    results: list = []


@SEARCH_PROXY_FSM.fast_service(monitor_only=True)
def parse_search_request(
    body_bytes: bytes,
    fsm_context: RequestContext = None,
) -> ParsedSearchRequest:
    """Extract the search query + count from a Firecrawl /v1/search body."""
    try:
        body = json.loads(body_bytes) if body_bytes else {}
    except json.JSONDecodeError:
        body = {}
    query = str(body.get("query") or "").strip()
    limit = int(body.get("limit") or 5)
    return ParsedSearchRequest(
        request_id=str(uuid.uuid4()),
        query=query,
        limit=limit,
        raw_body=body,
    )


@SEARCH_PROXY_FSM.fast_service(monitor_only=True)
def fetch_from_firecrawl(
    req: ParsedSearchRequest,
    firecrawl_base_url: str,
    firecrawl_api_key: str,
    upstream_path: str = "/v2/search",
    timeout_s: float = 30.0,
    fsm_context: RequestContext = None,
) -> FirecrawlResults:
    """Pass-through to upstream Firecrawl search endpoint.

    openclaw's bundled firecrawl plugin hits /v2/search; older callers may
    use /v1/search. The proxy accepts both paths and forwards to whichever
    upstream_path the caller asked for.
    """
    if not req.query:
        return FirecrawlResults(
            request_id=req.request_id, success=False, error="empty query"
        )
    t0 = time.time()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            r = client.post(
                f"{firecrawl_base_url.rstrip('/')}{upstream_path}",
                headers={
                    "Authorization": f"Bearer {firecrawl_api_key}",
                    "Content-Type": "application/json",
                },
                json=req.raw_body or {"query": req.query, "limit": req.limit},
            )
        upstream_ms = (time.time() - t0) * 1000.0
        if not r.is_success:
            return FirecrawlResults(
                request_id=req.request_id,
                success=False,
                upstream_ms=upstream_ms,
                error=f"firecrawl {r.status_code}: {r.text[:300]}",
            )
        data = r.json()
        raw_data = data.get("data")
        # v1 returns data as a list; v2 returns it as a dict
        # {web: [...], news: [...], ...}. Flatten to a list for our
        # canonical FirecrawlResults schema while preserving the
        # category-tagged structure in raw_body for downstream archival.
        if isinstance(raw_data, list):
            results_list = raw_data
        elif isinstance(raw_data, dict):
            results_list = []
            for category, items in raw_data.items():
                if isinstance(items, list):
                    for it in items:
                        if isinstance(it, dict):
                            results_list.append({**it, "_category": category})
        else:
            results_list = []
        return FirecrawlResults(
            request_id=req.request_id,
            success=bool(data.get("success", True)),
            raw_response=data,
            results=results_list,
            upstream_id=str(data.get("id") or ""),
            upstream_ms=upstream_ms,
        )
    except Exception as exc:  # noqa: BLE001
        return FirecrawlResults(
            request_id=req.request_id,
            success=False,
            upstream_ms=(time.time() - t0) * 1000.0,
            error=str(exc),
        )


@SEARCH_PROXY_FSM.fast_service(monitor_only=True)
def record_search_usage(
    rec: SearchUsageRecord,
    log_path: str,
    fsm_context: RequestContext = None,
) -> None:
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(rec.model_dump_json() + "\n")


@SEARCH_PROXY_FSM.fast_service
def archive_search(
    query: str,
    results: list,
    fsm_context: RequestContext = None,
) -> ArchivedSearch:
    """fast-service IntermediateLayer captures args + return value to disk."""
    return ArchivedSearch(query=query, results=results)


def make_app(
    firecrawl_base_url: str,
    firecrawl_api_key: str,
    usage_log_path: str,
) -> FastAPI:
    app = FastAPI(title="openclaw firecrawl search proxy (instrumented passthrough)")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "upstream": firecrawl_base_url,
            "purpose": "instrumented passthrough for openclaw native firecrawl provider",
        }

    async def _handle(body_bytes: bytes, upstream_path: str) -> JSONResponse:
        t0 = time.time()
        parsed = parse_search_request(body_bytes=body_bytes)
        fc = fetch_from_firecrawl(
            req=parsed,
            firecrawl_base_url=firecrawl_base_url,
            firecrawl_api_key=firecrawl_api_key,
            upstream_path=upstream_path,
        )
        total_ms = (time.time() - t0) * 1000.0

        rec = SearchUsageRecord(
            timestamp=time.time(),
            request_id=parsed.request_id,
            query=parsed.query,
            n_results=len(fc.results),
            upstream_ms=fc.upstream_ms,
            total_ms=total_ms,
            status_code=200 if fc.success else 502,
            error=fc.error,
        )
        record_search_usage(rec=rec, log_path=usage_log_path)

        if fc.success:
            try:
                archive_search(query=parsed.query, results=fc.results)
            except Exception:  # noqa: BLE001
                pass

        # Pass back the upstream JSON verbatim so the v1 (list) vs v2
        # (dict-of-categories) shape matches whatever openclaw asked for.
        # On error fall back to a minimal shape with the error message.
        if fc.success and fc.raw_response:
            return JSONResponse(content=fc.raw_response, status_code=200)
        return JSONResponse(
            content={
                "success": False,
                "data": [],
                "error": fc.error or "unknown",
            },
            status_code=502,
        )

    @app.post("/v1/search")
    async def search_v1(request: Request):
        return await _handle(await request.body(), "/v1/search")

    @app.post("/v2/search")
    async def search_v2(request: Request):
        return await _handle(await request.body(), "/v2/search")

    return app


def serve(
    fast_service_config_path: str,
    firecrawl_base_url: str,
    firecrawl_api_key: str,
    usage_log_path: str,
    host: str = "0.0.0.0",
    port: int = 1802,
) -> None:
    import logging

    log = logging.getLogger("agsbench_openclaw.search_proxy")
    cfg = FastServiceConfig.load_from_file(fast_service_config_path)
    log.info(
        "fast-service config: impl=%s monitor=%s store_intermediate=%s upstream=%s",
        cfg.service_impl,
        cfg.monitor,
        cfg.store_intermediate,
        firecrawl_base_url,
    )
    SEARCH_PROXY_FSM.execute(cfg)
    log.info("search proxy ready on http://%s:%d", host, port)

    app = make_app(firecrawl_base_url, firecrawl_api_key, usage_log_path)
    uvicorn.run(app, host=host, port=port, log_level="info")
