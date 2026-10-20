#!/bin/env sh

# Summarise MLE-bench results. Grading already happened during the run:
# after pi finishes (or hits the time limit), the agent service sends
# submission/submission.csv to the MLE-bench grader and records the report
# in trajectories/preds.json. This prints per-competition score / medal and
# the medal rate (MLE-bench's headline metric).
#
# Usage: bash scripts/4.evaluate.sh [path/to/preds.json]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGSBENCH_HOME="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PREDS="${1:-${AGSBENCH_HOME}/benchmark/data/auto-research-pi-autoresearch/trajectories/preds.json}"

if [ ! -f "$PREDS" ]; then
    echo "No results file at $PREDS"
    exit 1
fi

python3 - "$PREDS" << 'PY'
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

def fmt(x, spec=".5f"):
    return format(x, spec) if isinstance(x, (int, float)) else "n/a"

print(f"\n{'competition':<45} {'score':>10} {'median':>10} {'medal':>7} {'>median':>8} {'exps':>5} {'time(s)':>8}  status")
print("-" * 112)
for task_id, v in sorted(data.items()):
    g = v.get("grade_report") or {}
    status = "valid" if v.get("valid_submission") else "INVALID"
    if v.get("timed_out"):
        status += " (time limit)"
    if v.get("error"):
        status += f"  {v['error'][:60]}"
    print(f"{v.get('competition_id') or task_id:<45} {fmt(v.get('score')):>10} "
          f"{fmt(g.get('median_threshold')):>10} {v.get('medal') or '-':>7} "
          f"{'yes' if v.get('above_median') else 'no':>8} {v.get('n_experiments', 0):>5} "
          f"{fmt(v.get('duration_seconds'), '.0f'):>8}  {status}")
print("-" * 112)
n = len(data)
if n:
    valid = sum(1 for v in data.values() if v.get("valid_submission"))
    medal = sum(1 for v in data.values() if v.get("any_medal"))
    above = sum(1 for v in data.values() if v.get("above_median"))
    print(f"Valid submission: {valid}/{n} = {valid/n:.1%}")
    print(f"Any medal:        {medal}/{n} = {medal/n:.1%}")
    print(f"Above median:     {above}/{n} = {above/n:.1%}")
else:
    print("No results.")
PY
