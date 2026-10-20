"""Auto-research scenario data preparation — MLE-bench competitions.

Writes a JSON-per-line manifest, one record per MLE-bench competition. The
competition data itself (Kaggle download + MLE-bench train/test split) is
prepared by the grader image afterwards (see the deployment's
scripts/1.prepare_data.sh), because it needs the mlebench package and the
private answers must stay outside the agent container.

Default is one small competition so a 1-request run is a quick smoke test;
``--lite`` selects MLE-bench Lite (the 22 low-complexity competitions).
"""

import json
import os

# MLE-bench Lite = experiments/splits/low.txt at the MLE-bench commit pinned
# in agents/pi-autoresearch/grader/Dockerfile (507f92e1).
LITE_COMPETITIONS = [
    "aerial-cactus-identification",
    "aptos2019-blindness-detection",
    "denoising-dirty-documents",
    "detecting-insults-in-social-commentary",
    "dog-breed-identification",
    "dogs-vs-cats-redux-kernels-edition",
    "histopathologic-cancer-detection",
    "jigsaw-toxic-comment-classification-challenge",
    "leaf-classification",
    "mlsp-2013-birds",
    "new-york-city-taxi-fare-prediction",
    "nomad2018-predict-transparent-conductors",
    "plant-pathology-2020-fgvc7",
    "random-acts-of-pizza",
    "ranzcr-clip-catheter-line-classification",
    "siim-isic-melanoma-classification",
    "spooky-author-identification",
    "tabular-playground-series-dec-2021",
    "tabular-playground-series-may-2022",
    "text-normalization-challenge-english-language",
    "text-normalization-challenge-russian-language",
    "the-icml-2013-whale-challenge-right-whale-redux",
]
DEFAULT_COMPETITIONS = ["spooky-author-identification"]  # 1.9 MB text classification
MANIFEST_NAME = "pi_autoresearch_data.txt"


def extra_args(parser):
    parser.add_argument("--competitions", nargs="*", default=None,
                        help="MLE-bench competition ids (default: "
                             f"{' '.join(DEFAULT_COMPETITIONS)}).")
    parser.add_argument("--lite", action="store_true",
                        help="All 22 MLE-bench Lite competitions (~158 GB of data).")


def run(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, MANIFEST_NAME)
    if os.path.exists(manifest_path) and not args.force:
        print(f"{manifest_path} already exists. Use --force to overwrite.")
        return

    if args.lite:
        competitions = list(LITE_COMPETITIONS)
    else:
        competitions = args.competitions or list(DEFAULT_COMPETITIONS)

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for cid in competitions:
            manifest.write(json.dumps({"task_id": cid, "competition_id": cid}) + "\n")

    print(f"Wrote {len(competitions)} competitions to {manifest_path}")
