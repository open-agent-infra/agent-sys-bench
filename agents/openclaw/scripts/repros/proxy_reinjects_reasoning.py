"""End-to-end check that the proxy's reasoning-stash + reinject logic
makes the previously-failing case D succeed via the proxy.

Differs from `deepseek_thinking_tool_empty_choices.py` in two ways:

1. Routes through our proxy at 127.0.0.1:1801 (DEEPSEEK_API_KEY held
   server-side; the client uses a dummy auth string).
2. The client deliberately strips `reasoning_content` from the assistant
   message before sending turn 2 — i.e. simulates exactly what claw-eval
   does. The proxy should detect this via its stash and reinject the
   reasoning_content before forwarding upstream.

Expectations:
    F. turn 1 -> 200, captures reasoning_content (in the proxy's stash too)
    G. turn 2 *with* stripped reasoning -> 200, OK   (the proxy fixed it)
    H. turn 2 with stripped reasoning + UNRELATED tool_call_id -> 400
       (sanity check: the fix is targeted, not a blanket retry)

The proxy must already be running on 127.0.0.1:1801, configured with
DEEPSEEK_API_KEY. Start it with `scripts/serve_proxy.py --port 1801`.
"""

from __future__ import annotations

import json
import sys
import urllib.request
import urllib.error
import uuid

PROXY_URL = "http://127.0.0.1:1801/v1/chat/completions"
MODEL = "deepseek-v4-flash"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_drug_interaction",
            "description": "Look up whether two drugs interact.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drug_a": {"type": "string"},
                    "drug_b": {"type": "string"},
                },
                "required": ["drug_a", "drug_b"],
            },
        },
    }
]

USER = {
    "role": "user",
    "content": "Do perindopril and spironolactone interact in renal impairment?",
}


def call(messages, *, label: str) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 64,
        "thinking": {"type": "enabled"},
    }
    req = urllib.request.Request(
        PROXY_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer dummy-proxy-holds-real-key",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            payload = json.loads(r.read())
        status = r.status
    except urllib.error.HTTPError as e:
        payload = json.loads(e.read())
        status = e.code

    choices = payload.get("choices") or []
    msg = (choices[0].get("message") if choices else None) or None
    finish = choices[0].get("finish_reason") if choices else None
    print(f"\n--- {label} ---")
    print(f"  HTTP {status}  n_choices={len(choices)}  finish={finish}")
    if msg:
        tc = msg.get("tool_calls") or []
        if tc:
            print(f"  -> tool_calls={[t['function']['name'] for t in tc]}")
        elif msg.get("content"):
            print(f"  -> content: {msg['content'][:160]!r}")
    else:
        err = payload.get("error", {})
        print(f"  >>> NO CHOICES <<<  error={err.get('message')}")
    return payload


def main() -> int:
    # F. Turn 1 through the proxy with thinking ON. We need the assistant's
    # tool_calls so we can build a stripped-reasoning turn 2.
    pre = call([USER], label="F. turn1 via proxy, thinking ON")
    msg = pre["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        print(
            "[setup] turn 1 did not return tool_calls; cannot continue", file=sys.stderr
        )
        return 1

    # G. Turn 2: forward the assistant message WITHOUT reasoning_content
    # (this is what claw-eval / vanilla OpenAI clients do). The proxy's
    # reinjection logic should rescue it.
    stripped_assistant = {
        "role": "assistant",
        "content": msg.get("content"),  # may be empty for thinking-on
        "tool_calls": tool_calls,
        # NOTE: deliberately no "reasoning_content" here.
    }
    tool_result = {
        "role": "tool",
        "tool_call_id": tool_calls[0]["id"],
        "content": "Yes — both raise serum potassium; risk increased in renal impairment.",
    }
    call(
        [USER, stripped_assistant, tool_result],
        label="G. turn2 via proxy, reasoning stripped (proxy should reinject)",
    )

    # H. Sanity: if the tool_call_id is something the proxy has never seen,
    # the stash lookup misses and DeepSeek 400s as before. This proves the
    # G success is from the targeted reinjection, not a blanket retry.
    unknown_id = f"call_unknown_{uuid.uuid4().hex[:8]}"
    fake_tool_calls = [
        {
            **tool_calls[0],
            "id": unknown_id,
        }
    ]
    fake_assistant = {**stripped_assistant, "tool_calls": fake_tool_calls}
    fake_tool_result = {**tool_result, "tool_call_id": unknown_id}
    call(
        [USER, fake_assistant, fake_tool_result],
        label="H. turn2 via proxy, UNKNOWN tool_call_id (stash miss; expected to 400)",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
