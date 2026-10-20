#!/bin/env sh

# Summarise TerminalBench results. Scoring already happened during the run:
# after codex finishes, the agent service runs the task's run-tests.sh
# (pytest) in the task's sandbox and records the outcome in
# trajectories/results.json. This prints per-task pass/fail and accuracy.
#
# Usage: bash scripts/4.evaluate.sh [path/to/results.json]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AGSBENCH_HOME="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RESULTS="${1:-${AGSBENCH_HOME}/benchmark/data/terminal-use-codex/trajectories/results.json}"

if [ ! -f "$RESULTS" ]; then
    echo "No results file at $RESULTS"
    exit 1
fi

python3 - "$RESULTS" << 'PY'
import json, sys

with open(sys.argv[1]) as f:
    data = json.load(f)

print(f"\n{'task_id':<40} {'pass/total':>10}  {'exec(s)':>8} {'tests(s)':>8}  status")
print("-" * 85)
for task_id, v in sorted(data.items()):
    tp, tt = v.get("tests_passed", 0), v.get("tests_total", 0)
    ok = "PASS" if v.get("success") else "FAIL"
    ex, ts = v.get("codex_exec_secs"), v.get("test_runner_secs")
    ex = f"{ex:.1f}" if ex is not None else "n/a"
    ts = f"{ts:.1f}" if ts is not None else "n/a"
    print(f"{task_id:<40} {f'{tp}/{tt}':>10}  {ex:>8} {ts:>8}  {ok}  {v.get('exit_status') or ''}")
print("-" * 85)
total = len(data)
resolved = sum(1 for v in data.values() if v.get("success"))
print(f"Accuracy: {resolved}/{total} = {resolved/total:.1%}" if total else "No results.")
PY
