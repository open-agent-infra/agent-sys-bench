"""AgentSysBench analysis runner — drives the SDK extraction layer over all
default experiment settings and prints a per-agent x per-metric problem
matrix.

Usage (from repo root):
  python -m benchmark.analysis.run            # all settings
  python -m benchmark.analysis.run swe-mini-swe-agent qa-rag   # subset

Output: a table showing, for each agent, whether each of the 4 metrics
has usable data, and the reason when it doesn't. Per-setting reports go to
benchmark/analysis/out/<setting>/, with the normalised per-row tables
(requests/latency/io/tokens.csv) under tables/.
"""

import os
import sys
import traceback

# Make the sdk/ packages + this dir importable.
_REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "sdk"))
sys.path.insert(0, os.path.join(_REPO, "sdk", "fast-service", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for `import report`

import analysis  # noqa: E402  (sdk/analysis)
from analysis import config as C  # noqa: E402
from analysis import extract as E  # noqa: E402
from analysis import metrics as M  # noqa: E402
import report as R  # noqa: E402

METRICS = ["latency", "breakdown", "io", "tokens"]


def _status(metric_result: dict) -> str:
    if metric_result.get("available"):
        n = metric_result.get("n", metric_result.get("n_requests", metric_result.get("n_total", "?")))
        return f"OK ({n})"
    return f"MISSING: {metric_result.get('reason', 'unknown')}"


def _dump_tables(data, tables_dir: str) -> None:
    """Write the normalised per-row tables so custom analyses can start from
    one schema for every agent instead of re-parsing raw trajectories."""
    os.makedirs(tables_dir, exist_ok=True)
    for name in ("requests", "latency", "io", "tokens"):
        getattr(data, name).to_csv(os.path.join(tables_dir, f"{name}.csv"), index=False)


def run_setting(setting: str) -> dict:
    agent = C.SETTING_AGENT.get(setting)
    if not agent:
        return {"setting": setting, "agent": "?", "error": "unknown setting"}
    data_dir = os.path.join(C._BENCH_DATA_ROOT, setting)
    data = E.extract(agent, data_dir)
    metrics = M.compute_all(data)
    # produce the report
    out_dir = os.path.join(_REPO, "benchmark", "analysis", "out", setting)
    _dump_tables(data, os.path.join(out_dir, "tables"))
    try:
        summary_path = R.report_all(metrics, agent, out_dir)
    except Exception as e:
        summary_path = f"report error: {type(e).__name__}: {e}"
    return {
        "setting": setting, "agent": agent, "data_dir_exists": os.path.isdir(data_dir),
        "diagnostics": data.diagnostics, "metrics": metrics,
        "n_requests": int(len(data.requests)),
        "n_latency_rows": int(len(data.latency)),
        "n_io_rows": int(len(data.io)),
        "n_token_rows": int(len(data.tokens)),
        "report_dir": out_dir, "summary": summary_path,
    }


def main(argv):
    settings = argv[1:] or [m["setting"] for m in C.AGENT_META.values()]
    results = []
    for s in settings:
        try:
            results.append(run_setting(s))
        except Exception as e:
            results.append({"setting": s, "agent": C.SETTING_AGENT.get(s, "?"),
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc()})

    # ── Print matrix ─────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"{'setting':<32} {'agent':<18} {'data?':<6} {'#req':<5} {'latency':<22} {'breakdown':<22} {'io':<22} {'tokens':<22}")
    print("-" * 110)
    for r in results:
        if "error" in r and "metrics" not in r:
            print(f"{r['setting']:<32} {r.get('agent','?'):<18} {'-':<6} {'-':<5} {'ERROR: {error}'.format(**r)[:86]}")
            continue
        m = r["metrics"]
        print(f"{r['setting']:<32} {r['agent']:<18} "
              f"{'Y' if r['data_dir_exists'] else 'N':<6} {r['n_requests']:<5} "
              f"{_status(m['latency']):<22} {_status(m['breakdown']):<22} "
              f"{_status(m['io']):<22} {_status(m['tokens']):<22}")
    print("=" * 110)

    # ── Per-agent diagnostics (data problems) ────────────────────────
    print("\n### Diagnostics (data problems) ###")
    for r in results:
        if "metrics" not in r:
            print(f"\n[{r['setting']}] RUNNER ERROR: {r.get('error')}")
            if "traceback" in r:
                print(r["traceback"])
            continue
        print(f"\n[{r['setting']}] agent={r['agent']} data_dir={'OK' if r['data_dir_exists'] else 'MISSING'} "
              f"req={r['n_requests']} lat_rows={r['n_latency_rows']} io_rows={r['n_io_rows']} tok_rows={r['n_token_rows']}")
        for d in r.get("diagnostics", []):
            print(f"  - {d}")
        # breakdown fraction if available
        br = r["metrics"]["breakdown"]
        if br.get("available"):
            frac = br.get("fraction", {})
            print(f"  breakdown fractions: { {k: round(v,3) for k,v in frac.items()} }")
        tk = r["metrics"]["tokens"]
        if tk.get("available"):
            print(f"  token totals: {tk.get('totals')} cache_rate={tk.get('cache_rate',0):.3f}")


if __name__ == "__main__":
    main(sys.argv)
