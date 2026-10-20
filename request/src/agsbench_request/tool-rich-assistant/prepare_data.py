"""Tool-rich-assistant scenario data preparation.

Downloads PinchBench (pinned commit) and generates a manifest of
task .md files for the claude agent.

The manifest is a text file with one path per line (relative to the
manifest dir), each pointing to a task .md file inside the cloned pinchbench repo.
"""

import os
import sys

from agsbench_request._git import checkout_pinned

# Pinned commit for reproducibility — update intentionally.
PB_REPO = "https://github.com/pinchbench/skill.git"
PB_COMMIT = "819384ae830492365b8363fc26bc2602e73f216d"
MANIFEST_NAME = "claude_tasks.txt"


def extra_args(parser):
    parser.add_argument("--task_filter", default="",
                        help="Substring filter for task filenames. Empty = all tasks.")


def run(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not args.force:
        print(f"{manifest_path} already exists. Use --force to overwrite.")
        return

    # Fetch pinchbench at the pinned commit (retries on flaky GitHub access)
    pb_dir = os.path.join(output_dir, "pinchbench")
    checkout_pinned(PB_REPO, PB_COMMIT, pb_dir)

    # Scan for task .md files
    tasks_dir = os.path.join(pb_dir, "tasks")
    if not os.path.isdir(tasks_dir):
        print(f"ERROR: tasks/ directory not found in {pb_dir}", file=sys.stderr)
        sys.exit(1)

    md_files = []
    for root, dirs, files in os.walk(tasks_dir):
        dirs.sort()  # deterministic manifest order
        for f in sorted(files):
            if f.endswith(".md") and f != "TASK_TEMPLATE.md":
                full_path = os.path.join(root, f)
                if args.task_filter and args.task_filter not in full_path:
                    continue
                md_files.append(full_path)

    if not md_files:
        print(f"ERROR: no task .md files found in {tasks_dir}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for path in md_files:
            manifest.write(f"{os.path.relpath(path, output_dir)}\n")

    print(f"Wrote {len(md_files)} task paths to {manifest_path}")
    print(f"pinchbench at {pb_dir} (commit {PB_COMMIT[:8]})")
