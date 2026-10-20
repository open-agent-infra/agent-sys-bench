"""Inspect one setting's run: per-request summary, then a call-by-call
timeline for one request. Also a starting point for custom analyses — it
works on the same normalised tables (requests/latency/io/tokens) that
run.py writes to out/<setting>/tables/.

Usage (from repo root):
  python benchmark/analysis/inspect_run.py swe-mini-swe-agent            # summary + slowest request
  python benchmark/analysis/inspect_run.py swe-mini-swe-agent <request>  # timeline of <request>
"""

import os
import sys

_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "sdk"))
sys.path.insert(0, os.path.join(_REPO, "sdk", "fast-service", "src"))

import pandas as pd  # noqa: E402

from analysis import config as C  # noqa: E402
from analysis import extract as E  # noqa: E402


def per_request_summary(data) -> pd.DataFrame:
    """One row per request: e2e, #LLM calls, tokens, and seconds per category.
    <cat>_s sums leaf-call durations, so it can exceed e2e when calls overlap
    (e.g. deep-research runs searches/LLM calls concurrently)."""
    req = data.requests.set_index("request_id")[
        ["e2e_latency", "success", "n_llm_calls", "total_input_tokens", "total_output_tokens"]]
    lat = data.latency
    if not lat.empty:
        leaf = lat[lat["is_top_level"] != True]  # noqa: E712
        parents = set(leaf["parent_call_id"].astype(str)) - {"None", ""}
        leaf = leaf[~leaf["call_id"].astype(str).isin(parents) | (leaf["call_id"].astype(str) == "")]
        per_cat = leaf.pivot_table(index="request_id", columns="category", values="delay",
                                   aggfunc="sum", fill_value=0.0).add_suffix("_s")
        req = req.join(per_cat, how="left")
    return req.round(2)


def timeline(data, request_id: str) -> pd.DataFrame:
    """Non-top-level calls of one request, ordered by start time."""
    lat = data.latency[(data.latency["request_id"] == request_id)
                       & (data.latency["is_top_level"] != True)]  # noqa: E712
    cols = ["function_name", "category", "delay"]
    out = lat[cols + ["start_time"]].sort_values("start_time", kind="stable")[cols]
    return out.reset_index(drop=True).round(3)


def main(argv):
    if len(argv) < 2 or argv[1] not in C.SETTING_AGENT:
        print(__doc__)
        print("settings:", ", ".join(sorted(C.SETTING_AGENT)))
        return 1
    setting = argv[1]
    data = E.extract(C.SETTING_AGENT[setting], os.path.join(C._BENCH_DATA_ROOT, setting))
    for d in data.diagnostics:
        print(f"  - {d}")
    if data.requests.empty:
        print("no requests found")
        return 1

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    summary = per_request_summary(data)
    print(f"\n## {setting}: per-request summary\n")
    print(summary.to_string())

    rid = argv[2] if len(argv) > 2 else summary["e2e_latency"].idxmax()
    tl = timeline(data, rid)
    print(f"\n## timeline of request {rid} ({len(tl)} calls)\n")
    print(tl.to_string() if not tl.empty else "(no per-call trace rows for this request)")

    tok = data.tokens[data.tokens["request_id"] == rid]
    if not tok.empty:
        print(f"\n## tokens of request {rid}\n")
        print(tok[["input_tokens", "output_tokens", "cache_read_tokens"]].describe()
              .loc[["count", "mean", "max"]].round(0).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
