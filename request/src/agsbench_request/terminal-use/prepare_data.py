"""Terminal-use scenario data preparation — TerminalBench tasks for codex.

Clones terminal-bench (pinned commit) and writes a JSON-per-line manifest,
one record per task under ``original-tasks/``. Each task directory ships
its own Dockerfile + docker-compose.yaml (the sandbox) and run-tests.sh
(the pytest-based checker); the agent service builds and scores it, so the
manifest only carries the task id and instruction.

hello-world is placed first so a 1-request run is a quick smoke test; the
remaining tasks follow in sorted order.
"""

import json
import os
import sys

import yaml

from agsbench_request._git import checkout_pinned

# Pinned commit for reproducibility — update intentionally.
TB_REPO = "https://github.com/laude-institute/terminal-bench.git"
TB_COMMIT = "d28711d0da2675d0bb1d56de45ae5df6082438a3"  # 2026-07-10, 241 tasks
TASKS_SUBDIR = "original-tasks"
MANIFEST_NAME = "codex_data.txt"
FIRST_TASK = "hello-world"


def extra_args(parser):
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Explicit task ids to include (default: all tasks).")
    parser.add_argument("--task_filter", default="",
                        help="Substring filter for task ids. Empty = all tasks.")


def run(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not args.force:
        print(f"{manifest_path} already exists. Use --force to overwrite.")
        return

    tb_dir = os.path.join(output_dir, "terminal-bench")
    checkout_pinned(TB_REPO, TB_COMMIT, tb_dir)

    tasks_dir = os.path.join(tb_dir, TASKS_SUBDIR)
    task_ids = sorted(
        d for d in os.listdir(tasks_dir)
        if os.path.isfile(os.path.join(tasks_dir, d, "task.yaml"))
    )
    if args.tasks:
        missing = sorted(set(args.tasks) - set(task_ids))
        if missing:
            print(f"ERROR: unknown task ids: {missing}", file=sys.stderr)
            sys.exit(1)
        task_ids = [t for t in task_ids if t in set(args.tasks)]
    if args.task_filter:
        task_ids = [t for t in task_ids if args.task_filter in t]
    if FIRST_TASK in task_ids:
        task_ids.remove(FIRST_TASK)
        task_ids.insert(0, FIRST_TASK)
    if not task_ids:
        print(f"ERROR: no tasks selected under {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for tid in task_ids:
            with open(os.path.join(tasks_dir, tid, "task.yaml"), encoding="utf-8") as f:
                meta = yaml.safe_load(f) or {}
            record = {
                "task_id": tid,
                "instruction": meta.get("instruction", ""),
                "difficulty": meta.get("difficulty", ""),
                "category": meta.get("category", ""),
                "max_agent_timeout_sec": meta.get("max_agent_timeout_sec"),
                "max_test_timeout_sec": meta.get("max_test_timeout_sec"),
            }
            manifest.write(json.dumps(record) + "\n")
    print(f"Wrote {len(task_ids)} tasks to {manifest_path}")
