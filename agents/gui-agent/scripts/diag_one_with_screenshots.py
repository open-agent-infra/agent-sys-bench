"""Run a single OSWorld task with screenshot dumping enabled, for the
click-grounding diagnostic. Reads the task config from the staged JSON
and writes screenshots + traj.json into a dedicated directory.

Usage (inside os-world-service container):
    OS_WORLD_DUMP_SCREENSHOTS=1 OS_WORLD_AGENT_KIND=owl15 \
        python /workspace/os_world/scripts/diag_one_with_screenshots.py \
        c59742c0-4323-4b9d-8a02-723c251deaa0
"""

import json
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    task_id = (
        sys.argv[1] if len(sys.argv) > 1 else "c59742c0-4323-4b9d-8a02-723c251deaa0"
    )
    max_steps = int(os.environ.get("DIAG_MAX_STEPS", "8"))

    jsons_dir = Path("/workspace/os_world/.cache/os_world_data_jsons")
    found = None
    for p in jsons_dir.glob("*.json"):
        try:
            with p.open() as f:
                d = json.load(f)
            if (
                d.get("task_id") == task_id
                or d.get("task_config", {}).get("id") == task_id
            ):
                found = d
                break
        except Exception:
            continue
    if found is None:
        print(f"task config for {task_id} not found in {jsons_dir}", file=sys.stderr)
        sys.exit(2)

    task_config = found["task_config"]

    out_dir = Path(f"/workspace/os_world/trajectories/diag_{task_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["OS_WORLD_DUMP_SCREENSHOTS"] = "1"
    os.environ.setdefault("OS_WORLD_AGENT_KIND", "owl15")

    from agsbench_gui_agent import runner as runner_mod

    result = runner_mod.run_one(
        task_id=task_id,
        task_config=task_config,
        output_dir=str(out_dir.parent),
        max_steps=max_steps,
        observation_type="screenshot",
        fsm_context=None,
    )
    print(
        json.dumps(
            {
                "success": result.success,
                "task_id": result.task_id,
                "reward": result.reward,
                "n_steps": result.n_steps,
                "error": result.error,
            },
            indent=2,
        )
    )
