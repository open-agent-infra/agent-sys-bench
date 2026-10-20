# Claw-Eval PoC — third benchmark target through the same proxy

[Claw-Eval](https://claw-eval.github.io) is a multi-turn user-agent
benchmark with LLM-graded rubrics (completion, robustness, communication,
safety). It exercises a different code path than PinchBench / WildClawBench:
its runner (`claw-eval run`) speaks plain OpenAI Chat Completions over HTTPS,
**hardcodes `use_stream=True`**, and drives the model directly — there is
no `openclaw agent` CLI in the loop. This is a stronger cross-validation
than PinchBench: anything that speaks OpenAI-compatible HTTP can route
through our instrumented proxy.

## Tasks

Picked 3 tasks from the
[DeepSeek-V4-Flash leaderboard](https://claw-eval.github.io/#/model/deepseek_v4_flash)
covering distinct shapes (Chinese vs English, with-tools vs pure-dialogue,
short vs long horizon):

| task | language | tools needed | leaderboard task_score (judged) |
| --- | --- | --- | ---: |
| `C01zh_mortgage_prepay` | zh | none (financial advice dialogue) | 0.86 |
| `C06zh_clinical_pharmacy` | zh | `web_search` (drug interactions) | 0.74 |
| `C18en_statistical_analysis_spss` | en | none (stats consult dialogue) | 0.81 |

## How it runs

Claw-Eval reads model endpoints from a YAML config. We point all three
endpoints (`model`, `judge`, `user_agent_model`) at our proxy on `1801`:

```yaml
model:
  api_key: dummy-proxy-holds-real-key
  base_url: http://127.0.0.1:1801/v1
  model_id: deepseek-v4-flash
  context_window: 1000000
  extra_body:
    thinking: {type: disabled}
judge:
  ...same proxy URL, judging via DeepSeek (self-judge for this PoC)...
user_agent_model:
  ...same proxy URL, scripted persona at runtime...
```

```bash
DEEPSEEK_API_KEY=$(cat /data/agent-sys-bench \
  python openclaw/scripts/serve_proxy.py --port 1801 &

cd /data/agent-sys-bench
for t in C01zh_mortgage_prepay C06zh_clinical_pharmacy C18en_statistical_analysis_spss; do
  claw-eval run \
    --task /data/agent-sys-bench \
    --config /tmp/claweval_config.yaml \
    --trials 1 --no-judge
done
```

## Two pipeline changes this PoC forced

### 1. Streaming pass-through in the proxy

Claw-Eval's `runner/providers/openai_compat.py:294` hardcodes
`use_stream=True`. Our earlier proxy collected the full upstream response
before parsing usage — which works for the non-streaming
`openclaw agent` flow but stalls a streaming client. We added a
`_stream_proxy` path that:

- injects `stream_options.include_usage=True` into the upstream request
  (so DeepSeek emits a final SSE chunk with usage even when the client
  forgot to ask for one),
- forwards every SSE chunk verbatim via `StreamingResponse`,
- accumulates chunks in parallel for the audit log,
- emits the per-call record from a `finally` block so partial failures
  still get one row.

This change is benchmark-neutral: WildClawBench and PinchBench
(non-streaming) keep using the legacy non-stream path on the same code,
selected by `parsed.get("stream")`.

### 2. Reasoning-content echo bug (DeepSeek thinking-mode contract)

C06 first attempt — with `thinking: {type: enabled}` — failed at turn 2.
The error claw-eval reported was `RuntimeError: Model returned empty
choices`, but the underlying upstream response (visible in our proxy's
audit log) was actually:

```
HTTP 400 invalid_request_error:
  The `reasoning_content` in the thinking mode must be passed back to the API.
```

DeepSeek thinking-mode is a **stateless echo contract**: the assistant
turn returns both `content` and `reasoning_content`; on the next
request, the client must include the same `reasoning_content` back in
the assistant message in `messages[]`. OpenAI-standard clients
(claw-eval, openclaw's default OpenAI wrapper) strip `reasoning_content`
since it's not in the OpenAI message spec, so the very next tool turn
400s out. Reproduced in isolation at
`scripts/repros/deepseek_thinking_tool_empty_choices.py`:

```
A. turn1, thinking OFF                                  -> 200 OK
B. turn1, thinking ON                                   -> 200 OK
C. turn2 (with tool message), thinking OFF              -> 200 OK
D. turn2 (with tool message), thinking ON, no echo      -> 400  <-- the bug
E. turn2 (with tool message), thinking ON, echo back    -> 200 OK
```

So neither thinking-mode alone nor tool_calls alone are the issue —
it's specifically a follow-up turn after the client lost
`reasoning_content`.

**Fix lives at the proxy** (one place instead of every benchmark
driver): after each upstream response with thinking on, the proxy
stashes `reasoning_content` keyed by the tool_call ids that turn
emitted (`_REASONING_STASH` in `llm_proxy.py`). On the next inbound
request, the proxy walks `messages[]` for any assistant turn with
tool_calls but no `reasoning_content`, and re-injects the stashed
value before forwarding upstream. Both the JSON (openclaw) and the SSE
streaming (claw-eval) response paths populate the stash. End-to-end
verification at `scripts/repros/proxy_reinjects_reasoning.py`:

```
F. turn1 via proxy, thinking ON                              -> 200 OK
G. turn2 via proxy, reasoning STRIPPED (proxy reinjects)     -> 200 OK
H. turn2 via proxy, UNKNOWN tool_call_id (stash miss; 400)   -> 400 (sanity)
```

The H case confirms the fix is targeted — only known tool_call_ids
get reinjection.

## Results

Re-ran all three with the fix in place: **thinking enabled** (proxy
reinjects reasoning_content) and **judge enabled**, per-task tagged via
the proxy's `POST /admin/tag` so audit rows split cleanly:

| task | turns | proxy calls | proxy in / out / cache | wall (s) | task_score (judged) | leaderboard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C01zh_mortgage_prepay | 3 | 8 | 11 356 / 6 382 / 3 584 (31%) | 56 | **0.98** | 0.86 |
| C06zh_clinical_pharmacy | 9 | 18 | 106 780 / 17 368 / 59 136 (55%) | 135 | **0.92** | 0.74 |
| C18en_statistical_analysis_spss | 21 | 33 | 209 610 / 22 323 / 155 136 (74%) | 230 | **0.91** | 0.81 |

Total spend across all three: **$0.031** at DeepSeek list price (59 LLM
calls).

What this tells us:

- **All three pass**, all three exceed the published leaderboard
  number for DeepSeek-V4-Flash. A few caveats worth keeping in mind
  before reading too much into the deltas:
  - Single-trial scores; the leaderboard reports averages across
    multiple seeds. Our +0.10 ~ +0.18 deltas are within the variance
    we'd expect from sampling alone.
  - We self-judge with DeepSeek-V4-Flash via the same proxy. The
    public leaderboard's grader is a stronger model; self-judging is
    known to be more lenient.
  - Our web_search backend (Firecrawl, see `search_proxy.py`) is
    not the same one the leaderboard ran against. C06 in particular
    is sensitive to search recall.
- **Thinking on changed agent behavior on C06 dramatically**: 25 turns
  (looping on web_search) without thinking → 9 turns with thinking.
  The model's chain-of-thought is itself doing the disambiguation that
  it was previously offloading to repeated search queries. This is
  the single biggest behavioral difference between the unjudged and
  judged rounds.
- **Cache hit ratio rises with horizon**: 31% on a 3-turn task vs 74%
  on a 21-turn task, exactly as expected — the system prompt + early
  conversation is the same on every subsequent turn and gets served
  from DeepSeek's prefix cache.
- **Proxy call count > assistant turn count** because claw-eval issues
  one call per assistant turn **and** one per user_agent round. The
  proxy sees both, transparently — and the `task_tag` makes them
  attributable.

The earlier unjudged round (thinking disabled, `--no-judge`) is left
here for the record — it's what we hit first when the reasoning_content
echo bug surfaced:

| task | turns | proxy calls | task_score (`--no-judge`) |
| --- | ---: | ---: | ---: |
| C01zh_mortgage_prepay | 3 | (lost) | 0.20 |
| C06zh_clinical_pharmacy | 25 | 28 | 0.20 |
| C18en_statistical_analysis_spss | 3 | 6 | 0.20 |

With `--no-judge`, `completion` and `communication` clamp to 0 and the
floor is `robustness × safety × 0.2 = 0.20` — which is why every task
came back as 0.20 regardless of agent behavior. Useful only as a
plumbing check; not a score signal.

## Cross-validation

The proxy audit log + claw-eval trace agree on call counts and broad
token totals. Two specific consistency checks:

- C18 trace reports `model_tokens: 9489 (6787 in / 2702 out)`. Our
  proxy log totals **16 276 in / 3 874 out** across **6 calls** —
  higher because the proxy also captures the 3 user_agent calls
  (~9 489 input, ~1 172 output combined), which claw-eval's
  per-task `model_tokens` excludes. Sum-by-role matches.
- C06 trace `tool_use=29` ≡ chat.jsonl-equivalent `toolUse` count
  the proxy doesn't see directly — but the proxy log shows
  `finish_reason=tool_calls` on 22/28 calls and `stop` on 6, which is
  consistent with 22 tool-calling turns (some emitting multiple
  parallel `tool_use` blocks per call) plus 6 talk-only turns.

Same `postprocess.py` works on the Claw-Eval trace once we map their
`message` records to the chat.jsonl `assistant`/`user`/`toolUse`/`toolResult`
schema — fields differ by name but the data is the same shape. (We did
not vendor that adapter for the PoC; the proxy audit alone is enough to
show the pipeline reaches into a third benchmark.)

## Takeaway

Three benchmarks now route through the same fast-service-instrumented
proxy with zero per-benchmark code in the proxy itself:

| benchmark | driver | transport | tools? |
| --- | --- | --- | --- |
| WildClawBench | `openclaw agent` CLI | non-stream | yes (web_search, exec, ...) |
| PinchBench | `python benchmark.py` (openclaw CLI) | non-stream | yes |
| Claw-Eval | `claw-eval run` (direct OpenAI client) | **stream** | optional |

For the AgentSysBench paper this means the openclaw row in §6.3 can
draw from whichever benchmark scope makes the cleanest narrative —
PinchBench's 147-task breadth, WildClawBench's 5-model × 60-task grid,
or Claw-Eval's multi-turn user-agent rubric — and the per-call telemetry
schema is identical across all three.

## Open items (resolved)

- **C01 audit loss**: `scripts/serve_proxy.py` now rotates
  `llm_calls.jsonl` at boot to `llm_calls.<utc-launch-ts>.jsonl`. The
  proxy also exposes `POST /admin/tag {"tag": "..."}`, which stamps a
  `task_tag` column on every subsequent audit row — so a long-running
  proxy can serve a whole task batch and we slice per-task post-hoc.
  Done in `40d82f8`.
- **Thinking+tool_calls regression**: turned out to be a documented
  thinking-mode contract (reasoning_content must echo back), not an
  empty-choices regression — see § "Reasoning-content echo bug"
  above. Fixed in the proxy via stash + reinject. Reproduced in
  isolation at `scripts/repros/deepseek_thinking_tool_empty_choices.py`
  and verified end-to-end at `scripts/repros/proxy_reinjects_reasoning.py`.
  No upstream report needed — the contract is documented, our client
  was just non-conformant.
- **Judge cost**: re-ran all three with judge enabled (results above).
  Total marginal spend: $0.031, well under the ~$0.15 budget.
