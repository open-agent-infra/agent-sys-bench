# AgentSysBench Benchmark Analysis

Presentation layer — produces the 4-metric report (figures + CSV tables)
for each default experiment setting. Consumes the SDK's `AnalysisData` +
`Metrics`; agent-agnostic.

## Quick start

Run a deployment first (see [benchmark/deployment](../deployment/README.md));
its outputs land in `benchmark/data/<setting>/`. Then, from the repo root:

```bash
pip install pandas numpy matplotlib
python benchmark/analysis/run.py                              # every setting with data
python benchmark/analysis/run.py swe-mini-swe-agent qa-rag    # subset
```

Output lands in `benchmark/analysis/out/<setting>/`:
```
latency_hist.png, latency_stats.csv
breakdown_bar.png, breakdown_examples.png, breakdown.csv
io_scatter.png, io_stats.csv
token_breakdown.png, token_stats.csv
summary.txt
tables/requests.csv, tables/latency.csv, tables/io.csv, tables/tokens.csv
```

`tables/` holds the normalised per-row data (one schema for every agent,
joined on `request_id`; see `sdk/analysis/schema.py`) — use it as the input
for custom analyses instead of re-parsing raw trajectories.
`inspect_run.py` prints a per-request summary (e2e, LLM calls, tokens,
seconds per category) and one request's call timeline; it doubles as an
example of working with those tables:

```bash
python benchmark/analysis/inspect_run.py swe-mini-swe-agent              # per-request summary + slowest request timeline
python benchmark/analysis/inspect_run.py swe-mini-swe-agent <request_id> # timeline of one request
```

The metric definitions and per-agent extraction rules are in
[sdk/analysis/README.md](../../sdk/analysis/README.md).

## Raw data layout

Each `2.run.sh` moves the previous run's data to
`benchmark/data/_archive/<setting>-<timestamp>/` before starting, so
`benchmark/data/<setting>/` only ever holds one run (set
`AGSBENCH_KEEP_DATA=1` to append instead). `3.cleanup.sh` makes the
root-owned files readable on the host.

Paths relative to `benchmark/data/<setting>/`:

| Setting | Per-task trajectory | Latency trace (fast-service) |
|---|---|---|
| qa-rag | `runs/results/server/intermediate/requests/` | `runs/results/server/monitor/rag_server/e2e_delay.csv` |
| multimodal-hugginggpt | `traces/requests/` | `monitoring/coordinator/hugginggpt_server/e2e_delay.csv` |
| ai-search-deep-research | `runs/results/server/intermediate/requests/` | `runs/results/server/monitor/server/e2e_delay.csv` |
| swe-mini-swe-agent | `trajectories/<instance>/<instance>.traj.json`, `trajectories/preds.json` | `traces/mini-swe-agent/server/e2e_delay.csv` |
| terminal-use-codex | `trajectories/<task>/*.traj.json`, `*.codex_stdout.jsonl`, `results.json` | `traces/codex/server/e2e_delay.csv` |
| browser-use-webagent | `trajectories/<task>/*.traj.json`, `*.session.jsonl` | `monitoring/server/e2e_delay.csv` |
| computer-use-gui-agent | `trajectories/<task>/*.traj.json` | `traces/monitoring/os-world-service/e2e_delay.csv` |
| tool-rich-assistant-claude | `runs/latest/tasks/<task>/<run>/agent.stream.jsonl`, `llm_calls.csv`, `tool_calls.csv` | `runs/latest/fast-service-monitor/server/e2e_delay.csv` |
| office-work-openclaw | `output/<task>/<run>/chat.jsonl`, `run_summary.json` | — (wall time in `run_summary.json`) |
| auto-research-pi-autoresearch | `trajectories/<task>/*.traj.json`, `trajectories/<task>/pi-sessions/*.jsonl`, `trajectories/preds.json` (MLE-bench grades) | — |

Resource metrics (CPU / memory per container) are not part of the report;
they are in each deployment's Prometheus / Grafana while it is running.

## The 4 metrics

### 1. e2e request latency (`report_latency`)
- **Histogram** of per-request e2e latency with p50/p90/p95 vertical lines.
- **CSV table**: n, mean, std, min, p50, p90, p95, p99, max.
- Source: `RequestSummary.e2e_latency` (top-level trace delay or
  trajectory first→last timestamp).

### 2. latency breakdown (`report_breakdown`)
- **Stacked bar**: each category's mean delay as a fraction of e2e mean.
  For react agents, the untracked residual is labeled `LLM` (the LLM
  call is external). For pipeline agents, residual = `Other`.
- **Example breakdowns**: k-means (k=3) on per-request breakdown vectors
  picks representative requests whose breakdowns differ most — shows
  how individual requests deviate from the average.
- **CSV table**: category, fraction.

### 3. IO size (`report_io`)
- **Grouped bar** (log scale): mean input vs output bytes per category.
- **CSV table**: category, input_mean, input_p95, output_mean,
  output_p95, n_calls.
- Source: `IORecord` — bytes computed via `content_bytes()` (UTF-8 length
  of OpenAI-style message content, including text/thinking/tool_use/
  image_url blocks). For LLM calls, input = full conversation history
  up to that turn (quadratic growth).

### 4. token consumption (`report_tokens`)
- **Stacked bar**: per-category input (with cache_read overlay) +
  output tokens.
- **CSV table**: category, input_tokens, output_tokens,
  cache_read_tokens, reasoning_tokens, n_calls.
- **cache_rate** = total cache_read / total input (in summary.txt).
- Source: `TokenRecord` — per-call usage from each agent's token source
  (see `sdk/analysis/README.md` §1).

## Report summary (`summary.txt`)

One file per setting, human-readable:
```
agent: swe-mini-swe-agent
output: benchmark/analysis/out/swe-mini-swe-agent

[latency] available=True
  n=5 mean=127.68s p50=... p90=... p95=...
[breakdown] available=True
  e2e_mean=133.68s fractions={ LLM:0.51, Sandbox:0.46, Other:0.03 }
[io] available=True
  categories=['LLM', 'e2e'] n_total=48
[tokens] available=True
  totals={'input_tokens': 236037, ...} cache_rate=0.957
```

## Architecture

```
benchmark/analysis/
├── run.py         orchestrator: extract → metrics → report, prints problem matrix
├── inspect_run.py per-request summary + single-request timeline
├── report.py      matplotlib figures + CSV tables (agent-agnostic)
├── examples/      reference reports from short validation runs
└── out/           generated reports (one subdir per setting, gitignored)
```

`run.py` flow:
1. `extract.extract(agent, data_dir)` → `AnalysisData` (4 DataFrames).
2. `metrics.compute_all(AnalysisData)` → `Metrics` dict.
3. `report.report_all(Metrics, agent, out_dir)` → figures + CSVs + summary.

The runner also prints a **problem matrix** (per agent × per metric:
`OK (n)` or `MISSING: reason`) to stdout, so you can see at a glance
which agents have data issues.

## Per-experiment configuration

Each setting can override defaults via a `benchmark/analysis/<setting>/config.py`
(not yet implemented — currently all settings use the same report logic).
Planned overrides: title, warmup-N filter, category-map additions,
tool-name→Search overrides.

## Verified output

`examples/<setting>/` holds the reports from the short validation runs
(DashScope deepseek-v4-pro, 1–2 requests per setting):

| setting | #req | e2e mean | latency breakdown | cache_rate |
|---|---|---|---|---|
| qa-rag | 2 | 10.3 s | VecDB .54 / LLM .41 / Search .03 / Embedding .02 | — (no cache) |
| ai-search-deep-research | 2 | 334 s | LLM .89 / Search .11 (2.28x concurrent overlap) | — (no cache) |
| swe-mini-swe-agent | 2 | 268 s | LLM .90 / Sandbox .09 | .93 |
| terminal-use-codex | 1 | 37 s | LLM .58 / Sandbox .42 | .69 |
| browser-use-webagent | 2 | 4.7 s | LLM .84 / Sandbox .16 | — (no cache) |
| tool-rich-assistant-claude | 2 | 59 s | LLM 1.00 / Sandbox <.01 | .74 |
| office-work-openclaw | 1 | 601 s | LLM .70 / Sandbox .25 / Search .05 | .98 |
| multimodal-hugginggpt | 2 | 52 s | LLM .52 / Model .48 (3 GPUs) | — (no cache) |
| auto-research-pi-autoresearch | 1 | 1073 s | Sandbox .72 / LLM .28 (1 GPU) | .93 |

All four metrics are available for each of these. computer-use-gui-agent
(needs KVM) has no reference report in `examples/`.

Missing metrics are documented in `summary.txt` with the reason.

## Dependencies

- `pandas` (DataFrames)
- `numpy` (histogram, k-means, percentiles)
- `matplotlib` (Agg backend — static figures, no display needed)

No sklearn, no seaborn — minimal deps for reproducibility.
