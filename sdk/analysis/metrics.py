"""Metric computation — agent-agnostic. Operates only on AnalysisData
records (LatencyRecord / IORecord / TokenRecord / RequestSummary).

Produces the 4 standard metrics:
  1. e2e request latency (mean/std/p50/p90/p95/histogram)
  2. e2e latency breakdown (per-category fraction + example selection)
  3. IO size per category (input/output bytes distribution)
  4. token consumption breakdown (per-category token sums)
"""

import math
from typing import Dict, List

import numpy as np
import pandas as pd

from .schema import AnalysisData


def _empty_stats() -> dict:
    return {"n": 0, "available": False, "reason": "no data"}


# ── Metric 1: e2e request latency ────────────────────────────────────
def latency_stats(data: AnalysisData) -> dict:
    """Per-request e2e latency distribution. Uses RequestSummary.e2e_latency
    if present, else top-level fast-service delay."""
    df = data.requests
    if df.empty or "e2e_latency" not in df.columns:
        return _empty_stats()
    s = df["e2e_latency"].dropna()
    s = s[s > 0]
    if s.empty:
        return {"n": 0, "available": False, "reason": "all e2e_latency <= 0"}
    arr = s.to_numpy()
    hist, edges = np.histogram(arr, bins="auto")
    return {
        "n": int(len(s)),
        "available": True,
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "max": float(s.max()),
        "p50": float(s.quantile(0.5)),
        "p90": float(s.quantile(0.9)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)) if len(s) >= 1 else 0.0,
        "hist_counts": hist.tolist(),
        "hist_edges": edges.tolist(),
    }


# ── Metric 2: latency breakdown ──────────────────────────────────────
def latency_breakdown(data: AnalysisData) -> dict:
    """Per-category mean delay as fraction of e2e mean, plus per-request
    breakdown vectors for example selection."""
    df = data.latency
    if df.empty:
        return _empty_stats()
    # exclude top-level e2e rows from the breakdown components
    sub = df[~df.get("is_top_level", False)]
    if sub.empty:
        return {"n": 0, "available": False, "reason": "only top-level rows, no sub-calls"}
    # Only leaf calls: a parent's delay already contains its children's, so
    # counting both double-counts (e.g. generation -> normal_generation).
    parent_ids = set(df["parent_call_id"].dropna().astype(str)) - {"None", ""} \
        if "parent_call_id" in df else set()
    if parent_ids and "call_id" in sub:
        sub = sub[~sub["call_id"].astype(str).isin(parent_ids)]
    top = df[df.get("is_top_level", False)]
    if top.empty:
        # fall back: sum of all delays per request
        e2e_per_req = sub.groupby("request_id")["delay"].sum()
        e2e_mean = float(e2e_per_req.mean()) if not e2e_per_req.empty else 0.0
        n_req = int(e2e_per_req.size)
    else:
        e2e_mean = float(top["delay"].mean())
        n_req = int(len(top))
    if e2e_mean <= 0 or n_req == 0:
        return {"n": 0, "available": False, "reason": "e2e mean <= 0"}
    # per-category mean delay per request
    per_cat = (sub.groupby("category")["delay"].sum() / n_req).to_dict()
    fraction = {c: t / e2e_mean for c, t in per_cat.items()}
    tracked = sum(fraction.values())
    overlap = 1.0
    if tracked > 1.0:
        # Leaf calls ran concurrently (parallel sections/searches), so their
        # summed time exceeds wall-clock. Scale to shares of e2e and report
        # the concurrency factor separately.
        overlap = tracked
        fraction = {c: v / tracked for c, v in fraction.items()}
    elif tracked < 0.99:
        # For react agents the untracked residual is the LLM call (external,
        # not in the fast-service trace); for pipeline agents it is genuinely
        # "Other" overhead.
        from . import config as _C
        meta = _C.AGENT_META.get(data.agent, {})
        residual_label = meta.get("breakdown_residual") or (
            "LLM" if meta.get("type") == _C.AGENT_TYPE_REACT else "Other")
        # ADD residual to existing category (don't overwrite — trace may
        # already have some LLM rows like model_query_call).
        if residual_label in fraction:
            fraction[residual_label] += 1.0 - tracked
        else:
            fraction[residual_label] = 1.0 - tracked
    # per-request breakdown vector (for example selection)
    req_cat = sub.groupby(["request_id", "category"])["delay"].sum().unstack(fill_value=0)
    examples = _select_breakdown_examples(req_cat)
    return {
        "available": True,
        "e2e_mean": e2e_mean,
        "fraction": fraction,
        "per_category_seconds": per_cat,
        "overlap": overlap,
        "n_requests": int(df["request_id"].nunique()),
        "examples": examples,
    }


def _select_breakdown_examples(req_cat: pd.DataFrame, k: int = 3) -> List[dict]:
    """Pick k requests whose breakdown vectors are most distinct, via simple
    k-means on the normalized breakdown matrix. Returns representative
    request_ids + their breakdown fractions."""
    if req_cat.empty:
        return []
    mat = req_cat.div(req_cat.sum(axis=1).replace(0, 1), axis=0).fillna(0)
    n = len(mat)
    if n <= k:
        return [{"request_id": str(rid), "breakdown": mat.loc[rid].to_dict()}
                for rid in mat.index]
    # simple k-means (no sklearn dependency)
    from . import _kmeans
    centers, labels = _kmeans.kmeans(mat.to_numpy(), k)
    examples = []
    for c in range(k):
        idxs = [i for i, lb in enumerate(labels) if lb == c]
        if not idxs:
            continue
        # pick the point closest to its center
        dists = [np.linalg.norm(mat.to_numpy()[i] - centers[c]) for i in idxs]
        rep = idxs[int(np.argmin(dists))]
        examples.append({"request_id": str(mat.index[rep]),
                         "breakdown": mat.iloc[rep].to_dict(),
                         "cluster_size": len(idxs)})
    return examples


# ── Metric 3: IO size per category ───────────────────────────────────
def io_stats(data: AnalysisData) -> dict:
    df = data.io
    if df.empty:
        return _empty_stats()
    out = {"available": True, "per_category": {}}
    for cat, g in df.groupby("category"):
        inp = g[g["direction"] == "input"]["bytes"]
        outp = g[g["direction"] == "output"]["bytes"]
        out["per_category"][cat] = {
            "input_mean_bytes": float(inp.mean()) if not inp.empty else 0.0,
            "input_p95_bytes": float(inp.quantile(0.95)) if not inp.empty else 0.0,
            "output_mean_bytes": float(outp.mean()) if not outp.empty else 0.0,
            "output_p95_bytes": float(outp.quantile(0.95)) if not outp.empty else 0.0,
            "n_calls": int(len(g)),
        }
    out["n_total"] = int(len(df))
    return out


# ── Metric 4: token consumption breakdown ────────────────────────────
def token_breakdown(data: AnalysisData) -> dict:
    df = data.tokens
    if df.empty:
        return _empty_stats()
    out = {"available": True, "n": int(len(df)), "per_category": {}, "totals": {}}
    for cat, g in df.groupby("category"):
        out["per_category"][cat] = {
            "input_tokens": int(g["input_tokens"].fillna(0).sum()),
            "output_tokens": int(g["output_tokens"].fillna(0).sum()),
            "cache_read_tokens": int(g["cache_read_tokens"].fillna(0).sum()),
            "reasoning_tokens": int(g["reasoning_tokens"].fillna(0).sum()),
            "n_calls": int(len(g)),
        }
    for col in ("input_tokens", "output_tokens", "cache_read_tokens", "reasoning_tokens"):
        out["totals"][col] = int(df[col].fillna(0).sum())
    total_io = out["totals"]["input_tokens"] + out["totals"]["output_tokens"]
    if total_io == 0:
        # rows exist but carry no real token counts (e.g. only aggregate cost)
        return {"available": False, "n": int(len(df)),
                "reason": "token rows present but all input/output=0 (no per-call usage)"}
    if out["totals"].get("input_tokens", 0) > 0:
        out["cache_rate"] = out["totals"]["cache_read_tokens"] / out["totals"]["input_tokens"]
    else:
        out["cache_rate"] = 0.0
    return out


# ── Aggregate ────────────────────────────────────────────────────────
def compute_all(data: AnalysisData) -> dict:
    return {
        "latency": latency_stats(data),
        "breakdown": latency_breakdown(data),
        "io": io_stats(data),
        "tokens": token_breakdown(data),
    }
