"""Launcher for the openclaw LLM proxy.

Run this as the entry point — NOT `python -m agsbench_openclaw.llm_proxy`. The reason:
when invoked with -m, Python runs the file as __main__ and decorated
functions get __module__ == "__main__", which won't match the
`module_name: agsbench_openclaw.llm_proxy` entries in the fast-service YAML config.
The rewriter would silently drop them and you'd see no telemetry CSVs.

Usage:

    DEEPSEEK_API_KEY=<key> UPSTREAM_BASE_URL=<url> \\
    python scripts/serve_proxy.py --port 1801
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Make the openclaw package importable when running this script directly.

from agsbench_openclaw.llm_proxy import serve


def main() -> None:
    here = Path(__file__).parent.parent  # openclaw/
    ap = argparse.ArgumentParser(
        description=(
            "Run the openclaw LLM proxy. Conventional cwd is the openclaw/ "
            "directory (matching swe_agent/). All telemetry output paths in "
            "the YAML config and the audit log are resolved relative to that "
            "cwd, so everything lands under openclaw/.cache/."
        )
    )
    ap.add_argument(
        "--fast-service-config",
        default=str(here / "config" / "llm_proxy-server.yml"),
        help="Path to the fast-service YAML config (controls monitor output dir).",
    )
    ap.add_argument(
        "--upstream-base-url",
        default=os.environ.get("UPSTREAM_BASE_URL", "https://api.deepseek.com"),
    )
    ap.add_argument(
        "--usage-log",
        default="./.cache/openclaw/llm_proxy/llm_calls.jsonl",
        help=(
            "Where the per-call JSONL audit log is appended. Resolved against "
            "cwd, so by default lands under openclaw/.cache/openclaw/llm_proxy/ "
            "when launched from the openclaw/ directory."
        ),
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=1801)
    args = ap.parse_args()

    # Auto-cd into openclaw/ if launched from somewhere else, so YAML
    # `./.cache/...` paths resolve uniformly regardless of where the caller
    # was. Keeps all telemetry inside the openclaw subtree.
    if Path.cwd() != here.resolve():
        os.chdir(here)

    # Rotate the per-call audit log so each launch starts fresh and prior
    # audits are preserved alongside (rather than being clobbered or having
    # to be manually truncated between runs). Renames to a sibling file
    # `<stem>.<launch_ts>.<ext>` so timestamps sort naturally.
    log = Path(args.usage_log)
    if log.exists() and log.stat().st_size > 0:
        ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        log.rename(log.with_name(f"{log.stem}.{ts}{log.suffix}"))

    serve(
        fast_service_config_path=args.fast_service_config,
        upstream_base_url=args.upstream_base_url,
        usage_log_path=args.usage_log,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
