"""End-to-end PoC runner for one WildClawBench task through openclaw + the
fast-service-instrumented LLM proxy.

Usage:

    python openclaw/scripts/run.py \
        --task /data/agent-sys-bench \
        --model deepseek-instrumented/deepseek-v4-flash \
        --api-key-file /data/agent-sys-bench \
        --output-root ./output/openclaw_poc

    # With Docker sandbox:
    python openclaw/scripts/run.py \
        --task /path/to/task.md \
        --model deepseek-instrumented/deepseek-v4-flash \
        --sandbox

Assumes the LLM proxy is already running on port 1801. The runner does NOT
start the proxy itself — that's a separate long-running process (see
openclaw/scripts/serve_proxy.py).
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path

# Make the openclaw package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fast_service import FastServiceConfig
from agsbench_openclaw.run_task import (
    OCLAW_RUN_FSM,
    collect_outputs,
    launch_openclaw,
    load_task,
    prepare_workspace,
)
from agsbench_openclaw.postprocess import postprocess


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
    )


def verify_proxy_reachable(proxy_url: str) -> None:
    """Sanity-check before kicking off an expensive run."""
    import urllib.request, urllib.error

    try:
        with urllib.request.urlopen(f"{proxy_url}/health", timeout=3) as r:
            body = json.loads(r.read())
        if body.get("status") != "ok":
            raise RuntimeError(f"Proxy health returned {body}")
    except (urllib.error.URLError, ValueError) as exc:
        raise SystemExit(
            f"Proxy not reachable at {proxy_url}/health: {exc}\n"
            f"Start it with: python openclaw/scripts/serve_proxy.py --port 1801"
        )


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------
def ensure_sandbox_image(image: str = "openclaw-sandbox:bookworm-slim") -> None:
    """Check if the Docker sandbox image exists; build it if missing."""
    log = logging.getLogger("agsbench_openclaw.run")
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        log.info("Sandbox image %s exists.", image)
        return
    log.info("Sandbox image %s not found. Building...", image)
    dockerfile = textwrap.dedent(
        """\
        FROM debian:bookworm-slim
        ENV DEBIAN_FRONTEND=noninteractive
        RUN apt-get update && apt-get install -y --no-install-recommends \
          bash ca-certificates curl git jq python3 ripgrep \
          && rm -rf /var/lib/apt/lists/*
        RUN useradd --create-home --shell /bin/bash sandbox
        USER sandbox
        WORKDIR /home/sandbox
        CMD ["sleep", "infinity"]
    """
    )
    subprocess.run(
        ["docker", "build", "-t", image, "-"],
        input=dockerfile.encode(),
        check=True,
    )
    log.info("Built sandbox image %s.", image)


def ensure_sandbox_config(config_yaml_path: Path) -> None:
    """Apply the sandbox config patch if the main-sandbox agent doesn't exist yet.

    openclaw config patch replaces arrays by default, so we must merge the
    new agent entry into the existing agents.list ourselves.
    """
    log = logging.getLogger("agsbench_openclaw.run")
    import yaml as _yaml

    openclaw_json = Path.home() / ".openclaw" / "agsbench_openclaw.json"
    current = json.loads(openclaw_json.read_text()) if openclaw_json.exists() else {}
    existing_agents = current.get("agents", {}).get("list", [])
    if any(a.get("id") == "main-sandbox" for a in existing_agents):
        log.info("main-sandbox agent config already present.")
        return

    log.info("Applying sandbox config patch from %s", config_yaml_path)
    sandbox_cfg = _yaml.safe_load(config_yaml_path.read_text())
    new_agent = sandbox_cfg["agents"]["list"][0]
    existing_agents.append(new_agent)
    patch = {"agents": {"list": existing_agents}}
    proc = subprocess.run(
        ["openclaw", "config", "patch", "--stdin"],
        input=json.dumps(patch),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"openclaw config patch failed (rc={proc.returncode}):\n{proc.stderr}"
        )
    log.info("Sandbox config patch applied successfully.")


def main():
    here = Path(__file__).parent.parent  # openclaw/

    ap = argparse.ArgumentParser(
        description=(
            "Run one WildClawBench task end-to-end through openclaw + the "
            "fast-service-instrumented LLM proxy. Conventional cwd is "
            "openclaw/ (the script auto-cds there if launched from elsewhere)."
        )
    )
    ap.add_argument(
        "--task", required=True, help="Path to the WildClawBench task .md file"
    )
    ap.add_argument("--model", default="deepseek-instrumented/deepseek-v4-flash")
    ap.add_argument("--output-root", default="./output")
    ap.add_argument(
        "--fast-service-config", default=str(here / "config" / "run_task-server.yml")
    )
    ap.add_argument("--proxy-url", default="http://127.0.0.1:1801")
    ap.add_argument(
        "--proxy-audit", default="./.cache/openclaw/llm_proxy/llm_calls.jsonl"
    )
    ap.add_argument(
        "--sessions-root",
        default=str(Path.home() / ".openclaw" / "agents" / "main" / "sessions"),
    )
    ap.add_argument(
        "--results-md",
        default="",
        help="Override the path to look for results.md. Empty (default) uses "
        "<output_dir>/task_workspace/results/results.md (the path the "
        "prompt is auto-rewritten to use).",
    )
    ap.add_argument(
        "--timeout-override",
        type=int,
        default=0,
        help="Override the task's timeout_seconds (0 = use task default).",
    )
    ap.add_argument(
        "--thinking",
        default="",
        help=(
            "Forwarded as `openclaw agent --thinking <level>`. Levels: off | "
            "minimal | low | medium | high | xhigh | adaptive | max. Empty "
            "(default) uses openclaw's per-model default."
        ),
    )
    ap.add_argument(
        "--moonshot-api-key-file",
        default="/data/agent-sys-bench",
        help=(
            "Path to a file containing a Moonshot/Kimi API key. If the file "
            "exists, MOONSHOT_API_KEY and KIMI_API_KEY are set in the openclaw "
            "subprocess's env so its built-in web_search tool works."
        ),
    )
    # --- Sandbox options ---
    ap.add_argument(
        "--sandbox",
        action="store_true",
        default=False,
        help="Run agent tool calls inside a Docker sandbox instead of on the host.",
    )
    ap.add_argument(
        "--sandbox-workspace",
        default=str(here / "sandbox_workspace"),
        help="Fixed host directory used as the agent workspace for sandbox runs. "
        "Mounted at /workspace inside the container.",
    )
    ap.add_argument(
        "--sandbox-image",
        default="openclaw-sandbox:bookworm-slim",
        help="Docker image for the sandbox container.",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    setup_logging(args.log_level)
    log = logging.getLogger("agsbench_openclaw.run")

    # Auto-cd into openclaw/ so all './.cache' and './output' paths land
    # under this subtree regardless of where the user launched from.
    if Path.cwd() != here.resolve():
        log.info(
            "chdir %s -> %s (so ./output and ./.cache land under openclaw/)",
            Path.cwd(),
            here,
        )
        os.chdir(here)

    verify_proxy_reachable(args.proxy_url)

    # --- Sandbox setup (one-time checks) ---
    if args.sandbox:
        ensure_sandbox_image(args.sandbox_image)
        ensure_sandbox_config(here / "config" / "sandbox.yaml")
        Path(args.sandbox_workspace).mkdir(parents=True, exist_ok=True)

    # Surface the Moonshot/Kimi key to the openclaw subprocess so its
    # built-in web_search tool can dispatch.
    if args.moonshot_api_key_file:
        key_path = Path(args.moonshot_api_key_file)
        if key_path.exists():
            key = key_path.read_text().strip()
            os.environ["MOONSHOT_API_KEY"] = key
            os.environ["KIMI_API_KEY"] = key
            log.info(
                "loaded Moonshot/Kimi key from %s (len=%d) into env", key_path, len(key)
            )
        else:
            log.warning(
                "--moonshot-api-key-file %s does not exist; web_search will fail",
                key_path,
            )

    # Truncate the proxy's per-call audit JSONL so this run's audit data is
    # the only thing in there.
    audit_path = Path(args.proxy_audit)
    if audit_path.exists():
        audit_path.write_text("")

    # Initialize fast-service for the run_task module
    cfg = FastServiceConfig.load_from_file(args.fast_service_config)
    OCLAW_RUN_FSM.execute(cfg)
    log.info("fast-service ready (run_task)")

    # 1. load task
    task = load_task(task_md_path=args.task)
    log.info(
        "Loaded task %s (timeout=%ds, modality=%s)",
        task.task_id,
        task.timeout_seconds,
        task.modality,
    )

    # 2. prepare per-run output directory
    output_dir = prepare_workspace(
        output_root=args.output_root,
        task_id=task.task_id,
        sandbox=args.sandbox,
        sandbox_workspace=args.sandbox_workspace,
    )
    log.info("Output dir: %s", output_dir)

    # 3. launch openclaw
    t0 = time.perf_counter()
    run = launch_openclaw(
        task=task,
        model=args.model,
        output_dir=output_dir,
        timeout_override_s=args.timeout_override,
        thinking_level=args.thinking,
        sandbox=args.sandbox,
    )
    log.info(
        "openclaw exit=%s wall=%.1fs timed_out=%s",
        run.exit_code,
        run.wall_s,
        run.timed_out,
    )

    # 4. collect chat.jsonl + results.md
    if args.sandbox:
        sessions_root = str(
            Path.home() / ".openclaw" / "agents" / "main-sandbox" / "sessions"
        )
        results_md_path = args.results_md or str(
            Path(args.sandbox_workspace) / "results" / "results.md"
        )
    else:
        sessions_root = args.sessions_root
        results_md_path = args.results_md or str(
            Path(output_dir) / "task_workspace" / "results" / "results.md"
        )

    run = collect_outputs(
        run=run,
        sessions_root=sessions_root,
        results_md_path=results_md_path,
    )
    log.info("chat.jsonl=%s results.md=%s", run.chat_jsonl_path, run.results_md_path)

    # 5. post-process telemetry
    if run.chat_jsonl_path:
        post = postprocess(
            chat_jsonl_path=Path(run.chat_jsonl_path),
            proxy_audit_jsonl_path=Path(args.proxy_audit),
            output_dir=Path(output_dir) / "postprocessed",
            extra_context={
                "model": args.model,
                "task_id": task.task_id,
                "category": task.category,
                "run_id": run.run_id,
            },
        )
    else:
        post = {"n_llm_calls": 0, "n_tool_calls": 0, "n_proxy_rows": 0}

    summary = {
        "task_id": task.task_id,
        "category": task.category,
        "model": args.model,
        "run_id": run.run_id,
        "output_dir": output_dir,
        "exit_code": run.exit_code,
        "timed_out": run.timed_out,
        "error": run.error,
        "wall_s": round(run.wall_s, 2),
        "total_wall_s": round(time.perf_counter() - t0, 2),
        "results_md_produced": bool(run.results_md_path),
        "sandbox": args.sandbox,
        **post,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
