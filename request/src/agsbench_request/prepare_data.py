#!/usr/bin/env python
"""Unified data preparation entry point.

Usage:
  python -m request.prepare_data \\
      --scenario swe \\
      --output_dir ./.cache

Each scenario's prepare_data.py defines:
  - extra_args(parser)   — add scenario-specific CLI args
  - run(args)            — download + organize data, write manifest

Scenarios that need no data preparation (office-work, tool-rich-assistant)
can omit prepare_data.py — the entry point will report "no preparation needed".
"""

import argparse
import sys
from pathlib import Path

from agsbench_request import REQUEST_DIR


def main():
    # Phase 1: parse --scenario
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--scenario", required=True, help="Scenario name (e.g. swe)")
    pre_args, remaining = pre_parser.parse_known_args()

    scenario_dir = REQUEST_DIR / pre_args.scenario
    prepare_file = scenario_dir / "prepare_data.py"

    if not prepare_file.exists():
        print(f"No data preparation needed for scenario '{pre_args.scenario}'.")
        return

    # Load the scenario's prepare_data module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"request.{pre_args.scenario}.prepare_data", prepare_file
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, "run"):
        print(f"ERROR: scenario '{pre_args.scenario}' prepare_data.py missing required function: run(args)", file=sys.stderr)
        sys.exit(1)

    # Phase 2: full parser with common args + scenario-specific args
    parser = argparse.ArgumentParser(
        description=f"Prepare data for {pre_args.scenario} scenario."
    )
    parser.add_argument("--scenario", required=True, help=argparse.SUPPRESS)
    parser.add_argument("--output_dir", default="./.cache", help="Output directory for prepared data")
    parser.add_argument("--force", action="store_true", help="Force re-download/overwrite existing data")

    # Let the scenario add its own args
    if hasattr(mod, "extra_args"):
        mod.extra_args(parser)

    args = parser.parse_args()

    # Run the scenario's data preparation
    print(f"Preparing data for scenario: {pre_args.scenario}", flush=True)
    mod.run(args)
    print("Data preparation complete.", flush=True)


if __name__ == "__main__":
    main()
