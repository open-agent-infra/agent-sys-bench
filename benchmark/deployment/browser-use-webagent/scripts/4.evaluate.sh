#!/bin/env sh

# browsergym's reward is built into the env, so preds.json already carries
# per-task reward. This script just aggregates resolve rate (reward > 0 ==
# resolved) into a swebench-shaped report.

set -e

PREDICTIONS_PATH=${PREDICTIONS_PATH:-trajectories/preds.json}
RUN_ID=${RUN_ID:-browsergym-$(date -u +%Y%m%d-%H%M%S)}

python3 <<PYTHON
import json
from pathlib import Path

preds = json.loads(Path("${PREDICTIONS_PATH}").read_text())
total = len(preds)
resolved = [k for k, v in preds.items() if v.get("reward", 0) > 0]
unresolved = [k for k, v in preds.items() if v.get("reward", 0) <= 0]

report = {
    "total_instances": total,
    "submitted_instances": total,
    "completed_instances": total,
    "resolved_instances": len(resolved),
    "unresolved_instances": len(unresolved),
    "empty_patch_instances": 0,
    "error_instances": 0,
    "completed_ids": sorted(preds.keys()),
    "resolved_ids": sorted(resolved),
    "unresolved_ids": sorted(unresolved),
    "error_ids": [],
}
out = Path("trajectories") / "${RUN_ID}.eval.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
pct = (100.0 * len(resolved) / total) if total else 0.0
print(f"Total: {total}, Resolved: {len(resolved)} ({pct:.1f}%)")
print(f"Report: {out}")
PYTHON
