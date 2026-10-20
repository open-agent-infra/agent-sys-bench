"""Fast-service proxy for vLLM chat/completions calls.

Runs on GB200 as a sidecar service. The agent's call_llm() posts to this
proxy instead of vLLM directly. Fast-service instruments the call
automatically, providing latency/request/response data in fsm-monitor.
"""

import os
from typing import Any, Dict, Optional

import httpx
from pydantic import BaseModel, Field
from fast_service import RequestContext

from agsbench_gui_agent.fsm import agsbench_gui_agent_fsm


class LlmCallRequest(BaseModel):
    payload: Dict[str, Any] = Field(description="OpenAI chat/completions request body")
    model: str = "kimi-k2.6"


class LlmCallResponse(BaseModel):
    status_code: int = 200
    response: Dict[str, Any] = {}
    error: Optional[str] = None


@agsbench_gui_agent_fsm.fast_service
def llm_call(
    req: LlmCallRequest,
    fsm_context: RequestContext = None,
) -> LlmCallResponse:
    api_key = os.environ.get("KIMI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    api_url = os.environ.get("KIMI_API_URL") or os.environ.get(
        "OPENAI_BASE_URL", "https://api.moonshot.ai/v1"
    )
    if not api_url.endswith("/chat/completions"):
        api_url = api_url.rstrip("/") + "/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        response = httpx.post(
            api_url,
            headers=headers,
            json=req.payload,
            timeout=1200,
            verify=False,
        )
        if response.status_code != 200:
            return LlmCallResponse(
                status_code=response.status_code,
                error=response.text[:500],
            )
        return LlmCallResponse(
            status_code=200,
            response=response.json(),
        )
    except Exception as e:
        return LlmCallResponse(
            status_code=500,
            error=str(e)[:500],
        )
