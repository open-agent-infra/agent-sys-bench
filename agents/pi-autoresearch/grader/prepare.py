"""Download + prepare the MLE-bench competitions listed in the manifest.

For every competition in the manifest (one JSON object per line with a
``competition_id``):
  1. ``mlebench prepare`` into <data_dir>/<id>/prepared/{public,private}
     (needs Kaggle credentials and accepted competition rules);
  2. hard-link the public split to <public_dir>/<id>, which is the only
     tree the agent container mounts.
"""

import argparse
import json
import os
import shutil
import sys
import types
from importlib.metadata import version
from pathlib import Path

import requests

from mlebench.data import download_and_prepare_dataset, is_dataset_prepared
from mlebench.registry import registry


def _kaggle_compat():
    """mlebench catches ``kaggle.rest.ApiException`` (kaggle<1.7). kaggle 2.x,
    needed for KGAT_ tokens, has no kaggle.rest and raises requests'
    HTTPError instead, so provide that module."""
    if int(version("kaggle").split(".")[0]) >= 2 and "kaggle.rest" not in sys.modules:
        rest = types.ModuleType("kaggle.rest")
        rest.ApiException = requests.exceptions.HTTPError
        sys.modules["kaggle.rest"] = rest


def _competition_ids(manifest: str):
    with open(manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)["competition_id"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--public_dir", required=True)
    p.add_argument("--keep_raw", action="store_true")
    args = p.parse_args()
    _kaggle_compat()

    reg = registry.set_data_dir(Path(args.data_dir))
    os.makedirs(args.public_dir, exist_ok=True)
    failed = []
    for cid in _competition_ids(args.manifest):
        comp = reg.get_competition(cid)
        if is_dataset_prepared(comp):
            print(f"[{cid}] already prepared")
        else:
            print(f"[{cid}] downloading + preparing...")
            try:
                download_and_prepare_dataset(comp, keep_raw=args.keep_raw)
            except Exception as e:
                # e.g. EOFError: mlebench asks interactively when the rules
                # are not accepted; 401/403 from Kaggle on bad credentials.
                print(f"[{cid}] FAILED: {e}\n  Check KAGGLE_API_TOKEN and accept the rules at "
                      f"https://www.kaggle.com/c/{cid}/rules", file=sys.stderr)
                failed.append(cid)
                continue
        dst = Path(args.public_dir) / cid
        if not dst.exists():
            shutil.copytree(comp.public_dir, dst, copy_function=os.link)
        print(f"[{cid}] public data at {dst}")
    if failed:
        sys.exit(f"{len(failed)} competition(s) not prepared: {', '.join(failed)}")


if __name__ == "__main__":
    main()
