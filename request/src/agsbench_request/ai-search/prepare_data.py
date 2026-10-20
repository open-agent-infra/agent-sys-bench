"""AI-search scenario data preparation — hle / gaia / frames datasets.

Downloads from HuggingFace (pinned revisions) and generates a
JSON-per-line manifest for the deep-research agent.

hle and gaia are gated: accept the terms on huggingface.co and export
HF_TOKEN before running.
"""

import json
import os
from datasets import load_dataset

# Pinned HF dataset revisions for reproducibility.
DATASETS = {
    "hle": {
        "name": "cais/hle",
        "revision": "5a81a4c7271a2a2a312b9a690f0c2fde837e4c29",
    },
    "gaia": {
        "name": "gaia-benchmark/GAIA",
        "revision": "682dd723ee1e1697e00360edccf2366dc8418dd9",
    },
    "frames": {
        "name": "google/frames-benchmark",
        "revision": "58d9fb6330f3ab1316d1eca12e5e8ef23dcc22ef",
    },
}


def extra_args(parser):
    parser.add_argument("--dataset", type=str, default="hle",
                        choices=list(DATASETS.keys()),
                        help="Dataset to prepare.")
    parser.add_argument("--split", type=str, default="test")


def run(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, f"{args.dataset}.txt")
    if os.path.exists(manifest_path) and not args.force:
        print(f"{manifest_path} already exists. Use --force to overwrite.")
        return

    ds_info = DATASETS[args.dataset]
    ds_name = ds_info["name"]
    ds_rev = ds_info["revision"]

    if ds_rev:
        print(f"Loading {ds_name} (revision {ds_rev[:8]})...")
        ds = load_dataset(ds_name, split=args.split, revision=ds_rev, trust_remote_code=True)
    else:
        print(f"Loading {ds_name} (latest — no pinned revision)...")
        ds = load_dataset(ds_name, split=args.split, trust_remote_code=True)

    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(ds):
            prompt = next((item[k] for k in ("prompt", "Prompt", "question", "Question", "query")
                           if item.get(k)), "")
            prompt = prompt.strip()
            entry = {"id": str(item.get("id", item.get("task_id", i))), "topic": prompt,
                     "language": "en", "prompt": prompt}
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(ds)} entries to {manifest_path}")
