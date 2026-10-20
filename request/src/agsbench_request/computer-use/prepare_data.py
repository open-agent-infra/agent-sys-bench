"""Computer-use scenario data preparation — OSWorld tasks."""
import json
import os
import sys

from agsbench_request._git import checkout_pinned

UPSTREAM_REPO = "https://github.com/xlang-ai/OSWorld.git"
EVAL_EXAMPLES_SUBPATH = "evaluation_examples/examples"
MANIFEST_NAME = "os_world_data.txt"

# Pinned commit for reproducibility. Same commit the gui-agent image installs
# desktop_env from (agents/gui-agent/scripts/build_image.sh), so task configs
# match the baked-in evaluators.
PINNED_COMMIT = "091f5ef1d5544bc74953c77875d5feb5bed30108"  # 2026-07-28

def extra_args(parser):
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--domain", default=None,
                        help="Restrict to a single OSWorld domain (chrome, libreoffice_writer, ...).")

def run(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not args.force:
        print(f"{manifest_path} already exists. Use --force to overwrite.")
        return

    upstream_dir = os.path.join(output_dir, "OSWorld_upstream")
    checkout_pinned(UPSTREAM_REPO, PINNED_COMMIT, upstream_dir)

    examples_dir = os.path.join(upstream_dir, EVAL_EXAMPLES_SUBPATH)
    if not os.path.isdir(examples_dir):
        print(f"ERROR: {examples_dir} not found", file=sys.stderr)
        sys.exit(1)

    # Collect task JSONs
    task_files = []
    for root, _, files in os.walk(examples_dir):
        for f in sorted(files):
            if f.endswith(".json"):
                full = os.path.join(root, f)
                if args.domain and args.domain not in full:
                    continue
                task_files.append(full)

    if not task_files:
        print(f"ERROR: no task JSONs found in {examples_dir}", file=sys.stderr)
        sys.exit(1)

    # Sample
    import random
    random.seed(42)
    random.shuffle(task_files)
    sampled = task_files[:args.num_samples]

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for path in sampled:
            manifest.write(f"{os.path.relpath(path, output_dir)}\n")

    print(f"Wrote {len(sampled)} tasks to {manifest_path}")
