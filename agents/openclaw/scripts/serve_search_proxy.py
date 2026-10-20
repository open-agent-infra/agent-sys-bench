"""Launcher for the Firecrawl-backed kimi-shape search proxy.

Run this in addition to serve_proxy.py (the LLM proxy). They run on
different ports — 1801 for LLM, 1802 for search — and have separate
fast-service configs, so the telemetry for the two flows stays cleanly
separated under .cache/openclaw/{llm_proxy,search_proxy}/.

Why not python -m agsbench_openclaw.search_proxy: same PEP-563 / __main__-module
trap as the LLM proxy. We import the package so decorated functions get
__module__ == "agsbench_openclaw.search_proxy" matching the YAML entries.

Usage:

    FIRECRAWL_API_KEY=$(cat /data/agent-sys-bench \\
    python scripts/serve_search_proxy.py --port 1802
"""

import argparse
import os
import sys
from pathlib import Path


from agsbench_openclaw.search_proxy import serve


def main() -> None:
    here = Path(__file__).parent.parent  # openclaw/
    ap = argparse.ArgumentParser(
        description=(
            "Firecrawl-backed proxy mimicking openclaw's kimi web_search "
            "wire protocol. Auto-cd's into openclaw/ on startup so all "
            "telemetry lands under openclaw/.cache/openclaw/search_proxy/."
        )
    )
    ap.add_argument(
        "--fast-service-config",
        default=str(here / "config" / "search_proxy-server.yml"),
    )
    ap.add_argument(
        "--firecrawl-base-url",
        default=os.environ.get("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev"),
    )
    ap.add_argument(
        "--firecrawl-api-key",
        default=os.environ.get("FIRECRAWL_API_KEY", ""),
        help="Firecrawl API key. Default from FIRECRAWL_API_KEY env var.",
    )
    ap.add_argument(
        "--firecrawl-api-key-file",
        default="/data/agent-sys-bench",
        help="Fallback file to read the Firecrawl key from if --firecrawl-api-key not provided.",
    )
    ap.add_argument(
        "--usage-log",
        default="./.cache/openclaw/search_proxy/search_calls.jsonl",
    )
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=1802)
    args = ap.parse_args()

    if Path.cwd() != here.resolve():
        os.chdir(here)

    api_key = args.firecrawl_api_key
    if not api_key and args.firecrawl_api_key_file:
        p = Path(args.firecrawl_api_key_file)
        if p.exists():
            api_key = p.read_text().strip()
    if not api_key:
        raise SystemExit(
            "No Firecrawl API key: set FIRECRAWL_API_KEY env, pass "
            "--firecrawl-api-key, or place the key at "
            f"{args.firecrawl_api_key_file}"
        )

    serve(
        fast_service_config_path=args.fast_service_config,
        firecrawl_base_url=args.firecrawl_base_url,
        firecrawl_api_key=api_key,
        usage_log_path=args.usage_log,
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
