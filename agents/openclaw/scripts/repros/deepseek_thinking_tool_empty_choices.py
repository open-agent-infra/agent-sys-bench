"""Minimal reproducer for what we initially called "empty choices on
follow-up turn with thinking enabled" but is actually a documented
DeepSeek-V4-Flash contract that our agent driver violates:

    HTTP 400: "The reasoning_content in the thinking mode must be
              passed back to the API."

In thinking mode, DeepSeek returns the assistant message with both
`content` AND `reasoning_content`. On the *next* request, the client
must echo the *same* `reasoning_content` back as part of the assistant
message in `messages[]`. claw-eval (and openclaw's default OpenAI client)
strip `reasoning_content` since it's not part of the OpenAI-standard
message shape, which is why follow-up turns 400 out.

We narrowed it down by varying two factors independently:

    A. turn 1, thinking OFF -> 200, OK  (sanity)
    B. turn 1, thinking ON  -> 200, OK  (so thinking alone isn't the issue)
    C. turn 2 with tool msg, thinking OFF -> 200, OK  (so tool history alone isn't either)
    D. turn 2 with tool msg, thinking ON, NO reasoning echo -> 400  (combination)
    E. turn 2 with tool msg, thinking ON,    reasoning echo -> 200, OK  (confirmed fix)

The fix lives at the *proxy* layer (instead of every benchmark driver):
the proxy can stash each assistant response's `reasoning_content`
keyed by upstream_id and re-inject it on the next request when the
client echoes back the assistant message without it.

Run:

    DEEPSEEK_API_KEY=$(cat /data/agent-sys-bench \
      python openclaw/scripts/repros/deepseek_thinking_tool_empty_choices.py

Spend: 5 calls ~ $0.01.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error

URL = "https://api.deepseek.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"


def call(messages: list, *, thinking: bool, label: str) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 256,
        "thinking": {"type": "enabled" if thinking else "disabled"},
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            payload = json.loads(r.read())
        status = r.status
    except urllib.error.HTTPError as e:
        payload = json.loads(e.read())
        status = e.code

    choices = payload.get("choices") or []
    usage = payload.get("usage") or {}
    msg = (choices[0].get("message") if choices else None) or None
    finish = choices[0].get("finish_reason") if choices else None
    print(f"\n--- {label} ---")
    print(
        f"  HTTP {status}  thinking={'on' if thinking else 'off'}  n_choices={len(choices)}  finish={finish}"
    )
    print(
        f"  usage: prompt={usage.get('prompt_tokens')} completion={usage.get('completion_tokens')}"
    )
    if msg:
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            print(f"  -> tool_calls: {[tc['function']['name'] for tc in tool_calls]}")
        elif content:
            print(f"  -> content: {content[:200]!r}")
        else:
            print(f"  -> message has neither content nor tool_calls: keys={list(msg)}")
    else:
        print(f"  >>> EMPTY CHOICES <<<  full payload keys: {list(payload)}")
        if "error" in payload:
            print(f"      error: {payload['error']}")
    return payload


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

# Turn 1: user only.
TURN1 = [
    {
        "role": "user",
        "content": "Do perindopril and spironolactone interact in patients with renal impairment?",
    },
]

# Turn 2: user + assistant tool_call + tool result. This is what claw-eval
# (and any tool-using agent) sends on the second model call.
TURN2 = TURN1 + [
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_repro_1",
                "type": "function",
                "function": {
                    "name": "lookup_drug_interaction",
                    "arguments": json.dumps(
                        {"drug_a": "perindopril", "drug_b": "spironolactone"}
                    ),
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "call_repro_1",
        "content": "Yes — both can raise serum potassium; concurrent use risks hyperkalemia, particularly in renal impairment.",
    },
]


def main() -> int:
    if "DEEPSEEK_API_KEY" not in os.environ:
        print("DEEPSEEK_API_KEY not set", file=sys.stderr)
        return 2

    # Sanity checks first: each of the three confounding factors alone should be fine.
    call(TURN1, thinking=False, label="A. turn1, thinking OFF")
    r_b = call(TURN1, thinking=True, label="B. turn1, thinking ON")
    call(TURN2, thinking=False, label="C. turn2 (with tool message), thinking OFF")

    # The combination — what we observed failing in C06.
    call(
        TURN2,
        thinking=True,
        label="D. turn2 (with tool message), thinking ON, NO reasoning echo",
    )

    # The fix: echo the prior assistant's reasoning_content back. We need
    # to issue turn 1 *with* thinking on first (to obtain reasoning_content),
    # build a new turn-2 history that preserves it on the assistant message,
    # then re-call with thinking on.
    pre = call(
        TURN1,
        thinking=True,
        label="(pre-E) turn1 with thinking, captures reasoning_content",
    )
    pre_msg = pre["choices"][0]["message"]
    rc = pre_msg.get("reasoning_content")
    if not rc:
        print("\n[setup] no reasoning_content returned — cannot run E", file=sys.stderr)
        return 0
    turn2_with_echo = [
        TURN1[0],
        {
            "role": "assistant",
            "content": pre_msg.get("content"),
            "reasoning_content": rc,
            "tool_calls": pre_msg.get("tool_calls"),
        },
        {
            "role": "tool",
            "tool_call_id": pre_msg["tool_calls"][0]["id"],
            "content": "Yes — both can raise serum potassium; concurrent use risks hyperkalemia, particularly in renal impairment.",
        },
    ]
    call(
        turn2_with_echo,
        thinking=True,
        label="E. turn2 with tool message, thinking ON, reasoning echoed back",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
