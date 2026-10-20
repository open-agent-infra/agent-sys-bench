"""SWE scenario data preparation — SWE-bench dataset.

Downloads SWE-bench dataset (pinned HF revision), writes per-instance
JSON files + manifest for the mini-swe-agent benchmark.
"""

import json
import os
from pathlib import Path

import datasets

# Pinned HF dataset revision for reproducibility.
HF_DATASET = "princeton-nlp/SWE-bench"
HF_REVISION = "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6"
MANIFEST_NAME = "mini_swe_agent_data.txt"


def extra_args(parser):
    parser.add_argument("--data_path", default=HF_DATASET,
                        help=f"HF dataset name or local path (default: {HF_DATASET})")
    # test: swebench/sweb.eval.* instance images on Docker Hub exist only
    # for the test split (and 4.evaluate.sh scores against it).
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (default: test).")


def run(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not args.force:
        print(f"{manifest_path} already exists. Use --force to overwrite.")
        return

    data_path = args.data_path
    if Path(data_path).is_dir():
        try:
            dataset = datasets.load_from_disk(data_path)
        except FileNotFoundError:
            dataset = datasets.load_dataset(data_path)
    elif data_path == HF_DATASET:
        # Use pinned revision for the canonical HF dataset
        print(f"Loading {HF_DATASET} (revision {HF_REVISION[:8]})...")
        dataset = datasets.load_dataset(HF_DATASET, revision=HF_REVISION)
    else:
        dataset = datasets.load_dataset(data_path)

    if isinstance(dataset, dict):
        try:
            dataset = dataset[args.split]
        except KeyError:
            print(f"Invalid split: {args.split}. Valid splits: {list(dataset.keys())}")
            return

    jsons_dir = os.path.join(output_dir, "mini_swe_agent_data_jsons")
    os.makedirs(jsons_dir, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for idx, instance in enumerate(dataset):
            json_path = os.path.join(jsons_dir, f"{idx}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(instance, f)
            # Relative to the manifest dir so the loadgen can mount it anywhere.
            manifest.write(f"{os.path.relpath(json_path, output_dir)}\n")
    print(f"Wrote manifest to {manifest_path}")
