"""Wire upstream OSWorld's official `KimiAgent` into our service.

Per OSWorld's `mm_agents/kimi/README.md`, the official agent for Kimi
K2.5 / K2.6 OSWorld evaluation is `mm_agents.kimi.kimi_agent.KimiAgent`.
It owns its own prompt templates (incl. THINKING-mode variants), its
own response parser (`parse_response_to_cot_and_action`), and its own
Moonshot HTTP call (hard-coded to `https://api.moonshot.ai/v1/chat/
completions`, reads `KIMI_API_KEY`). We don't subclass it — we just
re-export and let the runner instantiate it with the leaderboard-
documented settings.

`mm_agents` is installed into the image from OSWorld at the commit
pinned in `scripts/build_image.sh` (see the Dockerfile overlay step).
"""

import os

from mm_agents.kimi.kimi_agent import KimiAgent as _UpstreamKimiAgent  # type: ignore  # noqa: E402
from mm_agents import owl15_agent as _owl15_mod  # type: ignore  # noqa: E402
from mm_agents.owl15_agent import Owl15Agent  # type: ignore  # noqa: E402,F401


# --------------------------------------------------------------------
# Patch owl15_agent.get_image_url so it returns a base64 data URL
# instead of uploading the screenshot to Alibaba OSS. Upstream's
# `push_oss` reads `os.environ['access_key_id']` etc. — we don't ship
# those credentials, and our local vLLM accepts inline base64 data
# URLs over the OpenAI vision API just fine.
# --------------------------------------------------------------------
def _data_url_for_base64(image_b64: str) -> str:
    # `image_b64` here is the bytes-or-str payload owl15_agent's predict()
    # passes — it has already been base64-encoded into a string.
    if isinstance(image_b64, bytes):
        image_b64 = image_b64.decode("ascii")
    return f"data:image/png;base64,{image_b64}"


_owl15_mod.get_image_url = _data_url_for_base64


# --------------------------------------------------------------------
# Owl15Agent hardcodes `self.single_image_token = 2645` (its comment
# says "for 1920*1080"). That value sets how many visual tokens the
# screenshot is squeezed into via Qwen's smart_resize — too low and the
# model can't read fine UI text or place clicks accurately. Expose the
# value as an env var so we can tune.
# --------------------------------------------------------------------
def _make_set_single_image_token(_orig_init):
    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        tok = os.environ.get("OS_WORLD_SINGLE_IMAGE_TOKEN")
        if tok:
            try:
                self.single_image_token = int(tok)
            except ValueError:
                pass

    return _patched_init


Owl15Agent.__init__ = _make_set_single_image_token(Owl15Agent.__init__)


# --------------------------------------------------------------------
# Bypass smart_resize for the standard 1920x1080 desktop screenshots.
# GUI-Owl-1.5 was trained on this exact resolution; the upstream
# update_image_size_ rounds 1080→1088 (factor=32) which breaks the
# model's coordinate grounding (outputs [946,306] instead of [986,34]).
# Sending the raw 1920x1080 image produces correct click coordinates.
# We must patch the name on the owl15_agent MODULE (not just
# qwen_vl_utils) because owl15_agent imports it at the top level.
# --------------------------------------------------------------------
from mm_agents.utils.qwen_vl_utils import update_image_size_ as _orig_update_image_size


def _patched_update_image_size(image_ele, **kwargs):
    w, h = image_ele["width"], image_ele["height"]
    if (w, h) == (1920, 1080):
        image_ele.update(
            {
                "resized_height": h,
                "resized_width": w,
                "seq_len": 0,
            }
        )
        return image_ele
    return _orig_update_image_size(image_ele, **kwargs)


_owl15_mod.update_image_size_ = _patched_update_image_size


class KimiAgent(_UpstreamKimiAgent):
    """Thin shim: identical to upstream KimiAgent, but stashes the raw
    Moonshot response so our `llm_pick_action_call` wrapper can read
    `.usage` (prompt / completion / total tokens) and the
    `reasoning_content` for per-call telemetry. Upstream's call_llm
    returns only `response['choices'][0]['message']`.
    """

    _last_response = None  # full response JSON from the latest successful call

    def call_llm(self, payload, model):
        import json
        import os
        import time

        import httpx

        # Provider-specific request fields, e.g. DashScope serves kimi-k2.6
        # in non-thinking mode unless `enable_thinking` is set; with upstream's
        # thinking=True prompts that leaves empty ◁think▷ history the model
        # starts echoing, which the parser then rejects.
        extra = os.environ.get("OS_WORLD_LLM_EXTRA_BODY", "")
        if extra:
            payload = {**payload, **json.loads(extra)}

        llm_proxy_url = os.environ.get("OS_WORLD_LLM_PROXY_URL", "")
        if llm_proxy_url:
            return self._call_llm_via_proxy(payload, model, llm_proxy_url)

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
        for _ in range(20):
            response = httpx.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=1200,
                verify=False,
            )
            if response.status_code != 200:
                time.sleep(5)
                continue
            response_json = response.json()
            finish_reason = response_json["choices"][0].get("finish_reason")
            if finish_reason is not None and finish_reason == "stop":
                self._last_response = response_json
                return response_json["choices"][0]["message"]
            time.sleep(5)

    def _call_llm_via_proxy(self, payload, model, proxy_url):
        import time
        import httpx

        proxy_url = proxy_url.rstrip("/")
        endpoint = f"{proxy_url}/os_world.llm_request.llm_call"
        for _ in range(20):
            resp = httpx.post(
                endpoint,
                json={"req": {"payload": payload, "model": model}},
                timeout=1200,
            )
            if resp.status_code != 200:
                time.sleep(5)
                continue
            result = resp.json()
            if result.get("error"):
                time.sleep(5)
                continue
            response_json = result.get("response", {})
            if not response_json.get("choices"):
                time.sleep(5)
                continue
            finish_reason = response_json["choices"][0].get("finish_reason")
            if finish_reason is not None and finish_reason == "stop":
                self._last_response = response_json
                return response_json["choices"][0]["message"]
            time.sleep(5)


__all__ = ["KimiAgent", "Owl15Agent"]
