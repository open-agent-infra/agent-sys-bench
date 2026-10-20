"""AgentSysBench analysis report — presentation layer.

Consumes the Metrics dict from ``sdk.analysis.metrics.compute_all`` and
produces static figures (matplotlib Agg) + CSV tables for the 4 standard
metrics. Agent-agnostic.

Output layout (per setting):
  <out_dir>/
    latency_hist.png, latency_stats.csv
    breakdown_bar.png, breakdown.csv
    io_scatter.png, io_stats.csv
    token_breakdown.png, token_stats.csv
    summary.txt
"""

import os
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import config as C  # sdk/analysis config (category colors)


def _save_fig(fig, path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _csv(path, rows, headers):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)


# ── 1. latency ───────────────────────────────────────────────────────
def report_latency(metrics: dict, out_dir: str):
    m = metrics["latency"]
    if not m.get("available"):
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    edges = m.get("hist_edges") or []
    counts = m.get("hist_counts") or []
    if edges and counts:
        ax.bar(edges[:-1], counts, width=edges[1] - edges[0] * 0.9 if len(edges) > 1 else 1,
               align="edge", alpha=0.7)
    for p, lbl, col in (("p50", "p50", "g"), ("p90", "p90", "orange"), ("p95", "p95", "r")):
        if p in m:
            ax.axvline(m[p], color=col, linestyle="--", linewidth=1, label=f"{lbl}={m[p]:.2f}s")
    ax.set_xlabel("e2e latency (s)")
    ax.set_ylabel("count")
    ax.set_title(f"e2e request latency (n={m['n']})")
    ax.legend(fontsize=8)
    _save_fig(fig, os.path.join(out_dir, "latency_hist.png"))
    _csv(os.path.join(out_dir, "latency_stats.csv"),
         [[m.get("n"), m.get("mean"), m.get("std"), m.get("min"), m.get("p50"),
           m.get("p90"), m.get("p95"), m.get("p99"), m.get("max")]],
         ["n", "mean", "std", "min", "p50", "p90", "p95", "p99", "max"])


# ── 2. breakdown ─────────────────────────────────────────────────────
def report_breakdown(metrics: dict, out_dir: str):
    m = metrics["breakdown"]
    if not m.get("available"):
        return
    frac = m.get("fraction", {})
    if not frac:
        return
    cats = [c for c in C.CATEGORIES if c in frac] + [c for c in frac if c not in C.CATEGORIES]
    vals = [frac[c] for c in cats]
    colors = [C.CATEGORY_COLORS.get(c, "#AAAAAA") for c in cats]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(cats, vals, color=colors)
    ax.set_ylabel("fraction of e2e latency")
    ax.set_title(f"latency breakdown (e2e mean={m.get('e2e_mean',0):.2f}s, n={m.get('n_requests',0)})")
    ax.set_ylim(0, max(1.0, max(vals) * 1.1))
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    _save_fig(fig, os.path.join(out_dir, "breakdown_bar.png"))
    _csv(os.path.join(out_dir, "breakdown.csv"),
         [[c, frac[c]] for c in cats], ["category", "fraction"])
    # example breakdowns
    examples = m.get("examples") or []
    if examples:
        fig, ax = plt.subplots(figsize=(5, 3))
        n_ex = len(examples)
        ecats = sorted({c for ex in examples for c in ex["breakdown"]})
        import numpy as np
        bottoms = np.zeros(n_ex)
        for cat in ecats:
            vals = [ex["breakdown"].get(cat, 0) for ex in examples]
            ax.bar(range(n_ex), vals, bottom=bottoms,
                   color=C.CATEGORY_COLORS.get(cat, "#AAAAAA"), label=cat)
            bottoms += np.array(vals)
        ax.set_xticks(range(n_ex))
        ax.set_xticklabels([ex["request_id"][:12] for ex in examples], fontsize=7, rotation=15)
        ax.set_ylabel("fraction")
        ax.set_title("example request breakdowns (cluster reps)")
        ax.legend(fontsize=7, loc="best")
        _save_fig(fig, os.path.join(out_dir, "breakdown_examples.png"))


# ── 3. IO ────────────────────────────────────────────────────────────
def report_io(metrics: dict, out_dir: str):
    m = metrics["io"]
    if not m.get("available"):
        return
    per = m.get("per_category", {})
    if not per:
        return
    cats = [c for c in C.CATEGORIES if c in per] + [c for c in per if c not in C.CATEGORIES]
    fig, ax = plt.subplots(figsize=(5, 3))
    x = range(len(cats))
    inp = [per[c]["input_mean_bytes"] for c in cats]
    outp = [per[c]["output_mean_bytes"] for c in cats]
    ax.bar([i - 0.2 for i in x], inp, width=0.4, label="input", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], outp, width=0.4, label="output", color="#55A868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, fontsize=7, rotation=15)
    ax.set_ylabel("mean bytes")
    ax.set_yscale("log")
    ax.set_title(f"IO size per category (n={m.get('n_total',0)} rows)")
    ax.legend(fontsize=8)
    _save_fig(fig, os.path.join(out_dir, "io_scatter.png"))
    _csv(os.path.join(out_dir, "io_stats.csv"),
         [[c, per[c]["input_mean_bytes"], per[c]["input_p95_bytes"],
           per[c]["output_mean_bytes"], per[c]["output_p95_bytes"], per[c]["n_calls"]]
          for c in cats],
         ["category", "input_mean_bytes", "input_p95_bytes", "output_mean_bytes", "output_p95_bytes", "n_calls"])


# ── 4. tokens ────────────────────────────────────────────────────────
def report_tokens(metrics: dict, out_dir: str):
    m = metrics["tokens"]
    if not m.get("available"):
        return
    per = m.get("per_category", {})
    if not per:
        return
    cats = [c for c in C.CATEGORIES if c in per] + [c for c in per if c not in C.CATEGORIES]
    fig, ax = plt.subplots(figsize=(5, 3))
    import numpy as np
    x = np.arange(len(cats))
    inp = [per[c]["input_tokens"] for c in cats]
    outp = [per[c]["output_tokens"] for c in cats]
    cache = [per[c]["cache_read_tokens"] for c in cats]
    ax.bar(x, inp, label="input", color="#4C72B0")
    ax.bar(x, cache, bottom=inp, label="cache_read", color="#C44E52", alpha=0.6)
    ax.bar(x, outp, bottom=[i + c for i, c in zip(inp, cache)], label="output", color="#55A868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, fontsize=7, rotation=15)
    ax.set_ylabel("tokens")
    ax.set_title(f"token consumption (cache_rate={m.get('cache_rate',0):.2f})")
    ax.legend(fontsize=8)
    _save_fig(fig, os.path.join(out_dir, "token_breakdown.png"))
    _csv(os.path.join(out_dir, "token_stats.csv"),
         [[c, per[c]["input_tokens"], per[c]["output_tokens"], per[c]["cache_read_tokens"],
           per[c]["reasoning_tokens"], per[c]["n_calls"]] for c in cats],
         ["category", "input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens", "n_calls"])


# ── aggregate ────────────────────────────────────────────────────────
def report_all(metrics: dict, agent: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    report_latency(metrics, out_dir)
    report_breakdown(metrics, out_dir)
    report_io(metrics, out_dir)
    report_tokens(metrics, out_dir)
    _report_markdown(metrics, agent, out_dir)
    # summary
    repo = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    lines = [f"agent: {agent}", f"output: {os.path.relpath(out_dir, repo)}", ""]
    for name in ("latency", "breakdown", "io", "tokens"):
        m = metrics[name]
        avail = m.get("available", False)
        lines.append(f"[{name}] available={avail}")
        if name == "latency" and avail:
            lines.append(f"  n={m['n']} mean={m.get('mean',0):.2f}s p50={m.get('p50',0):.2f} p90={m.get('p90',0):.2f} p95={m.get('p95',0):.2f}")
        elif name == "breakdown" and avail:
            lines.append(f"  e2e_mean={m.get('e2e_mean',0):.2f}s fractions={{ {', '.join(f'{k}:{v:.2f}' for k,v in m.get('fraction',{}).items())} }}")
            if m.get("overlap", 1.0) > 1.0:
                lines.append(f"  overlap={m['overlap']:.2f}x (concurrent sub-calls; fractions scaled to e2e)")
        elif name == "io" and avail:
            lines.append(f"  categories={list(m.get('per_category',{}).keys())} n_total={m.get('n_total',0)}")
        elif name == "tokens" and avail:
            lines.append(f"  totals={m.get('totals')} cache_rate={m.get('cache_rate',0):.3f}")
        else:
            lines.append(f"  reason={m.get('reason','n/a')}")
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return os.path.join(out_dir, "report.md")


def _fmt_bytes(n: float) -> str:
    if n < 1024:
        return f"{n:.0f} B"
    if n < 1048576:
        return f"{n/1024:.1f} KB"
    return f"{n/1048576:.1f} MB"


def _report_markdown(metrics: dict, agent: str, out_dir: str):
    """Generate a single report.md with all 4 metrics, tables, and figure refs."""
    lines = [f"# {agent} — Analysis Report", ""]

    # ── latency ──
    m = metrics["latency"]
    lines.append("## 1. E2E Request Latency")
    if m.get("available"):
        lines.append(f"- **Requests**: {m['n']}")
        lines.append(f"- **Mean**: {m.get('mean',0):.2f}s | **Std**: {m.get('std',0):.2f}s")
        lines.append(f"- **p50**: {m.get('p50',0):.2f}s | **p90**: {m.get('p90',0):.2f}s | **p95**: {m.get('p95',0):.2f}s | **p99**: {m.get('p99',0):.2f}s")
        lines.append(f"- **Min**: {m.get('min',0):.2f}s | **Max**: {m.get('max',0):.2f}s")
        lines.append("")
        lines.append("![Latency Histogram](latency_hist.png)")
    else:
        lines.append(f"⚠️ Not available: {m.get('reason','no data')}")
    lines.append("")

    # ── breakdown ──
    m = metrics["breakdown"]
    lines.append("## 2. Latency Breakdown")
    if m.get("available"):
        lines.append(f"- **E2E mean**: {m.get('e2e_mean',0):.2f}s | **Requests**: {m.get('n_requests',0)}")
        frac = m.get("fraction", {})
        if frac:
            lines.append("")
            lines.append("| Category | Fraction |")
            lines.append("|----------|----------|")
            for cat in sorted(frac, key=lambda c: -frac[c]):
                lines.append(f"| {cat} | {frac[cat]:.1%} |")
        lines.append("")
        lines.append("![Breakdown](breakdown_bar.png)")
        if m.get("examples"):
            lines.append("")
            lines.append("### Representative request breakdowns")
            lines.append("")
            lines.append("![Examples](breakdown_examples.png)")
    else:
        lines.append(f"⚠️ Not available: {m.get('reason','no data')}")
    lines.append("")

    # ── IO ──
    m = metrics["io"]
    lines.append("## 3. IO Size per Category")
    if m.get("available"):
        per = m.get("per_category", {})
        if per:
            lines.append("")
            lines.append("| Category | Input (mean) | Input (p95) | Output (mean) | Output (p95) | Calls |")
            lines.append("|----------|-------------|-------------|--------------|--------------|-------|")
            for cat in sorted(per.keys()):
                s = per[cat]
                lines.append(f"| {cat} | {_fmt_bytes(s['input_mean_bytes'])} | {_fmt_bytes(s['input_p95_bytes'])} | {_fmt_bytes(s['output_mean_bytes'])} | {_fmt_bytes(s['output_p95_bytes'])} | {s['n_calls']} |")
        lines.append("")
        lines.append(f"- **Total IO records**: {m.get('n_total',0)}")
        lines.append("")
        lines.append("![IO Size](io_scatter.png)")
    else:
        lines.append(f"⚠️ Not available: {m.get('reason','no data')}")
    lines.append("")

    # ── tokens ──
    m = metrics["tokens"]
    lines.append("## 4. Token Consumption")
    if m.get("available"):
        per = m.get("per_category", {})
        totals = m.get("totals", {})
        lines.append(f"- **Cache rate**: {m.get('cache_rate',0):.1%}")
        lines.append(f"- **Total input**: {totals.get('input_tokens',0):,} | **Total output**: {totals.get('output_tokens',0):,}")
        lines.append(f"- **Cache read**: {totals.get('cache_read_tokens',0):,} | **Reasoning**: {totals.get('reasoning_tokens',0):,}")
        if per:
            lines.append("")
            lines.append("| Category | Input | Output | Cache Read | Reasoning | Calls |")
            lines.append("|----------|-------|--------|------------|-----------|-------|")
            for cat in sorted(per.keys()):
                s = per[cat]
                lines.append(f"| {cat} | {s['input_tokens']:,} | {s['output_tokens']:,} | {s['cache_read_tokens']:,} | {s['reasoning_tokens']:,} | {s['n_calls']} |")
        lines.append("")
        lines.append("![Token Breakdown](token_breakdown.png)")
    else:
        lines.append(f"⚠️ Not available: {m.get('reason','no data')}")
    lines.append("")
    lines.append("---")
    lines.append("*Generated by `benchmark/analysis/run.py` — see [README](../README.md) for usage.*")

    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
