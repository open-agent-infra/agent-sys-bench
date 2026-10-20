"""Browser-use scenario data preparation — manifest of browsergym task ids.

Task ids are builtin (no external download). The MiniWoB++ HTML is
pre-baked into the webagent Docker image, so this script only generates
the manifest of task ids.
"""

import json
import os
import random

# Pinned task list — these are stable Gym-registered MiniWoB++ task ids
# that (a) are single-page synthetic HTML, (b) solvable in 1-3 steps,
# (c) don't require credentials or external network.
DEFAULT_TASKS = [
    "browsergym/miniwob.click-test",
    "browsergym/miniwob.click-button",
    "browsergym/miniwob.click-link",
    "browsergym/miniwob.focus-text",
    "browsergym/miniwob.identify-shape",
    "browsergym/miniwob.click-checkboxes",
    "browsergym/miniwob.click-tab",
    "browsergym/miniwob.simple-arithmetic",
]

MANIFEST_NAME = "browsergym_data.txt"


def extra_args(parser):
    parser.add_argument("--num_samples", type=int, default=len(DEFAULT_TASKS),
                        help="Number of tasks to stage.")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Explicit task ids to use. Overrides DEFAULT_TASKS.")
    parser.add_argument("--seed", type=int, default=42)


def run(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not args.force:
        print(f"{manifest_path} already exists. Use --force to overwrite.")
        return

    tasks = args.tasks or DEFAULT_TASKS
    random.seed(args.seed)
    random.shuffle(tasks)
    sampled = tasks[:args.num_samples]
    print(f"Staging {len(sampled)} tasks: {sampled}")

    jsons_dir = os.path.join(output_dir, "browsergym_data_jsons")
    os.makedirs(jsons_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for idx, task_id in enumerate(sampled):
            payload = {"instance_id": f"miniwob-{idx:04d}", "task_id": task_id, "seed": idx}
            json_path = os.path.join(jsons_dir, f"{idx}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            manifest.write(f"{os.path.relpath(json_path, output_dir)}\n")
    print(f"Wrote manifest to {manifest_path}")
