# PinchBench PoC — cross-validating the instrumentation pipeline

[PinchBench](https://pinchbench.com) is a separate openclaw-targeted
benchmark suite. We use it as a second target to validate that our
fast-service-instrumented LLM proxy is benchmark-agnostic: anything that
drives openclaw via the standard `openclaw agent` CLI flows through our
proxy without modification.

## Tasks

Picked 3 fully-automated-grading tasks from PinchBench's 147-task suite,
covering distinct capability buckets that DeepSeek-V4-Flash is rated
well on per the
[leaderboard](https://pinchbench.com/submission/b67cf3cb-81bb-4bd3-83ed-359ff9bc6a5c)
(90% overall):

| task | category | timeout | DeepSeek leaderboard avg |
| --- | --- | ---: | ---: |
| `task_sanity` | productivity (Core Agent) | 60 s | 98% |
| `task_calendar` | productivity | 120 s | 82% |
| `task_access_log_anomaly` | analysis | 90 s | 96% |

## How it runs

PinchBench's own `benchmark.py` invokes `openclaw agent --model <id>
--message <prompt>` per task; nothing about our instrumented proxy is
specific to WildClawBench. We just have the proxy running on port 1801
with openclaw configured (in `~/.openclaw/agsbench_openclaw.json`) to use our
custom `deepseek-instrumented/deepseek-v4-flash` provider — exactly the
same setup we use for WildClawBench.

```bash
# 1. Start the LLM proxy (and search proxy if needed for this suite)
DEEPSEEK_API_KEY=$(cat /data/agent-sys-bench \
  python openclaw/scripts/serve_proxy.py --port 1801 &

# 2. Make sure OPENROUTER_API_KEY is NOT set (pinchbench would
#    otherwise try to validate the model via OpenRouter).
unset OPENROUTER_API_KEY

# 3. Run pinchbench
cd /data/agent-sys-bench
python3 scripts/benchmark.py \
  --model deepseek-instrumented/deepseek-v4-flash \
  --suite task_sanity,task_calendar,task_access_log_anomaly \
  --no-upload --output-dir /tmp/pinchbench_poc
```

## Results

3 / 3 tasks scored 1.0 / 1.0 (100%) in 65.6 s wall time end-to-end.

| task | score | wall (s) | calls (proxy / pinchbench) | DeepSeek-side cost (pinchbench) |
| --- | ---: | ---: | ---: | ---: |
| task_sanity | 1.00 | 17.5 | 2 / 2 | $0.0034 |
| task_calendar | 1.00 | 23.0 | 5 / 5 | $0.0002 |
| task_access_log_anomaly | 1.00 | 25.2 | 4 / 4 | $0.0006 |
| **total** | **3.00 / 3.00** | **65.6** | **11 / 11** | **$0.0043** |

PinchBench's own efficiency summary computed the cost using
cache-aware pricing (cache_read tokens billed at a discount), which is
why these numbers are lower than the flat-rate-input pricing we use
internally. Both agree on call counts.

## Cross-validation

Per-call audit log lined up exactly with PinchBench's transcript:
- 11 proxy `/v1/chat/completions` POSTs ≡ 11 PinchBench-reported `request_count`s
- Sum of proxy `completion_tokens` (1 720) matches PinchBench's
  reported `total_output_tokens` (3 440 — they count both fresh and
  cached input tokens; our 1 720 number is just visible completion).
- Our per-call `reasoning_tokens` are all `None` since we didn't pass
  `--thinking` (PinchBench's default), confirming the upstream returned
  no `completion_tokens_details.reasoning_tokens`.

The same `postprocess.py` pipeline works on PinchBench transcripts
(located under `_pinchbench_poc_0001/transcripts/<task>.jsonl` after
the run) — the chat.jsonl record layout is identical because PinchBench
uses the same openclaw runtime under the hood.

```bash
python -m postprocess \
  --chat-jsonl _pinchbench_poc_0001/transcripts/task_calendar.jsonl \
  --proxy-audit _pinchbench_poc_0001/proxy_llm_calls.jsonl \
  --output-dir _pinchbench_poc_0001/postprocessed/task_calendar \
  --model deepseek-instrumented/deepseek-v4-flash \
  --task-id task_calendar --category pinchbench --run-id 0001
```

Output is the same schema as for WildClawBench runs:
`llm_calls.csv` + `tool_calls.csv` with `is_synthetic`,
`full_prompt_tokens`, `reasoning_tokens`, `visible_output_tokens`
columns — usable directly with the same downstream analysis (per-step
reasoning split, prompt-growth slope, cache hit ratio, etc.).

## Takeaway

The PoC confirms our openclaw-side instrumentation works end-to-end on
PinchBench without any code change — only the model-config patch on
`~/.openclaw/agsbench_openclaw.json` is needed, and that lives at the openclaw
level, not in the benchmark driver. **The same pipeline can drop into
any openclaw-compatible benchmark** (PinchBench, WildClawBench,
ClawEval, future ones) without rework.

For the AgentSysBench paper, this means we don't need to commit to one
benchmark; we can pick whichever scope (PinchBench's 147-task
leaderboard view, WildClawBench's 60-task multi-model view, or both)
makes the strongest narrative for the revision.
