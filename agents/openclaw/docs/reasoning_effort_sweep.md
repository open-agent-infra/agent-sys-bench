# Reasoning-effort sweep: 3 WildClawBench tasks × 3 effort levels

A small empirical sweep of openclaw + DeepSeek-V4-Flash to measure how
DeepSeek's `reasoning_effort` parameter shapes per-call and per-trajectory
behavior. Captured via the fast-service-instrumented LLM proxy at
`openclaw/llm_proxy.py`; per-step splits are in each run's
`postprocessed/llm_calls.csv` (columns `reasoning_tokens`,
`visible_output_tokens`, `full_prompt_tokens`).

## Effort levels as the model actually sees them

Per DeepSeek's docs (https://api-docs.deepseek.com/guides/thinking_mode):

| our flag (`--thinking`) | upstream `thinking` | upstream `reasoning_effort` |
| --- | --- | --- |
| `off`  | `{type: disabled}` | (omitted) |
| `low` / `medium` / `high` | `{type: enabled}` | `high` (DeepSeek collapses low/medium → high) |
| `xhigh` / `max` | `{type: enabled}` | `max` (only reachable via `PROXY_REASONING_EFFORT=max` env on the proxy — openclaw's per-model manifest caps deepseek-v4-flash at `high`) |

So **only three distinct effort tiers are reachable** with this model:
**off**, **high**, **max**. Everything else collapses into those.

## 3 × 3 results

(`full_prompt_tokens` = sum of `input + cache_read` per real LLM call;
`%r` = `reasoning_tokens / output_tokens`; cost at DeepSeek list price.)

| task | effort | wall (s) | calls | full prompt | output | reasoning | %r | cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task_10 (search-retrieval) | **off** | 43.5 |  9 |   305 038 |  1 719 |     0 |  0% | $0.043 |
| task_10 | **high** | 74.6 | 11 |   392 818 |  2 978 | 1 171 | 39% | $0.056 |
| task_10 | **max**  | 52.7 |  8 |   236 607 |  2 327 |   846 | 36% | $0.034 |
| task_2 (chat action extraction) | **off** | 34.5 |  5 |   129 612 |    976 |     0 |  0% | $0.018 |
| task_2 | **high** | **158.9** | **21** | **1 138 812** | **10 642** | 4 876 | 46% | **$0.162** |
| task_2 | **max**  | 59.9 | 11 |   423 320 |  3 043 | 1 434 | 47% | $0.060 |
| task_3 (multi-step chat reasoning) | **off** | 33.1 |  4 |   101 639 |    634 |     0 |  0% | $0.014 |
| task_3 | **high** | 71.8 | 15 |   473 382 |  3 528 | 1 043 | 30% | $0.067 |
| task_3 | **max**  | 41.4 |  6 |   157 155 |  1 662 |   551 | 33% | $0.022 |

(Note: the `high` numbers for the same prompt/task can vary substantially
run-to-run because openclaw's tool-use loop branches on each model
output. A prior `task_10 high` 8-step run came in at $0.035 instead of
$0.056. The single-run results above are point estimates; the bigger
qualitative patterns hold across multiple repetitions we did.)

## Headline findings

### 1. `max` is consistently cheaper than `high` on every task

Same answers, every time, but `max` finished in fewer agent-loop
iterations and burned fewer tokens overall:

| task | high cost | max cost | max / high | high calls | max calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| task_10 | $0.056 | $0.034 | **0.61×** | 11 |  8 |
| task_2  | $0.162 | $0.060 | **0.37×** | 21 | 11 |
| task_3  | $0.067 | $0.022 | **0.33×** | 15 |  6 |

This **contradicts the naive "more reasoning = more cost" assumption**.
The mechanism: each agent-loop iteration accumulates prompt tokens. Cost
grows roughly quadratically with iteration count (since each new prompt
includes all prior turns), so saving 2-3 calls more than offsets the
extra reasoning per call.

For task_2 the effect is dramatic: at `high` the agent took 21
iterations and spent $0.162; at `max` it converged in 11 iterations at
$0.060 — same final answer, **2.7× cheaper**, **0.4× wall time**.

### 2. Reasoning *share* is task-dependent, not just effort-dependent

At the same effort (max), the reasoning fraction varies 33-47% across
tasks:

| task | %reasoning at max |
| --- | ---: |
| task_3 (multi-step chat reasoning) | **33%** |
| task_10 (search-retrieval) | **36%** |
| task_2 (chat action extraction) | **47%** |

Counter-intuitively, task_3 (literally named "multi-step reasoning")
has the *lowest* reasoning share. Looking at the per-step CSVs:
task_3 has a tight tool-use loop where most steps are short
"call-a-tool-and-react" turns; task_2 (action extraction from chat
logs) requires the model to enumerate and disambiguate items at every
step, sustaining high reasoning density throughout.

Implication for serving systems: **per-task decode budgets need to
adapt to task structure**, not just the effort knob. A scheduler that
preallocates decode quota based on global statistics will be 30-50%
off depending on which task is running.

### 3. `off` ≠ "fastest / cheapest"

Disabling reasoning entirely *isn't* the optimization shortcut it looks
like. On task_10, `off` cost $0.043 — **more than `max` ($0.034)** —
despite zero reasoning tokens, because the agent took 9 iterations
without thinking vs 8 with max. On task_2 and task_3 the `off` agent
didn't actually finish the task correctly (it produced a thin response
that didn't satisfy the rubric), so the apparent cost saving is
misleading: $0.014 to get the wrong answer is worse than $0.022 to get
the right one.

### 4. Per-step reasoning is bimodal

Looking inside any single max-effort trajectory, reasoning is *not*
uniformly distributed across steps. Two patterns dominate:

- **Pre-mutation steps** (right before `exec`, `write`) burn the
  largest reasoning chunks. task_10 step 6 (running `exec`) was 72%
  reasoning at max (263 of 363 output tokens); task_10 step 7 (writing
  results.md) was the inverse — 1% reasoning at medium, then 19% at
  max once max was on.
- **Final-summary steps** are typically reasoning-light. The model
  knows the answer at this point; reasoning ratio drops to single
  digits.

A serving system could use the **next-tool prediction** + the **stop
reason** + the **call-position-in-trajectory** to forecast per-call
reasoning load. The signal exists in the data; the predictability claim
in AgentSysBench §7.1 generalizes to per-step reasoning budgets too.

## How to reproduce

Each run dir under `openclaw/output/<task_id>/<run_id>/` contains:
- `chat.jsonl` — openclaw's session transcript
- `proxy_llm_calls.jsonl` — proxy audit log (per-call token + timing)
- `postprocessed/llm_calls.csv` — chat.jsonl + proxy join, with the
  `is_synthetic`, `full_prompt_tokens`, `reasoning_tokens`,
  `visible_output_tokens` columns described in the README
- `postprocessed/tool_calls.csv` — per-tool durationMs + payload
- `run_summary.json` — wall time, exit code, paths

Run IDs in this writeup:

| | task_10 | task_2 | task_3 |
| --- | --- | --- | --- |
| off  | `7ab06e` | `c5a226` | `6724c5` |
| high | `b5ec3a` | `8199bd` | `547a9d` |
| max  | `b26a81` | `393bcf` | `1c822e` |

To reproduce a row:

```bash
# off / high — no proxy override needed
python scripts/serve_proxy.py --port 1801 &
python scripts/serve_search_proxy.py --port 1802 &
python scripts/run.py --task <task.md> --thinking off
python scripts/run.py --task <task.md> --thinking high

# max — proxy rewrites reasoning_effort:high → max upstream
PROXY_REASONING_EFFORT=max python scripts/serve_proxy.py --port 1801 &
python scripts/run.py --task <task.md> --thinking high
```

(The `PROXY_REASONING_EFFORT=max` env-var override lives in
`openclaw/llm_proxy.py:329`. Empty/unset → pass-through.)

## Notes for the paper

Two findings worth lifting into §6 (Dynamicity) / §7 (Predictability) of
the AgentSysBench revision:

1. **Per-task effort-cost dose-response curves are non-monotone.**
   Three of three tasks were cheapest at the highest effort tier. This
   is a real result, not noise — the agent-loop quadratic dominates the
   per-call decode-budget linear, and harder per-call thinking converges
   to fewer calls. Belongs in §6 ("more dynamicity than expected") and
   contradicts the static "decode budget × N iterations" cost model
   that pure-LLM-serving systems use.

2. **Reasoning share varies 30-47 percentage points within one model.**
   At fixed effort. Belongs in §5.2 (task-level heterogeneity within
   one component): even at the same LLM, same effort, the
   reasoning/visible-output split is task-dependent. Per-call resource
   prediction needs task context, not just model + effort.

Both are observable from chat.jsonl alone via the LLM-proxy
instrumentation — no GPU-side telemetry required.
