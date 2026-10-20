"""Fast-service RPC module for mcp-atlas task execution.

Wraps run_mcp_atlas.run_task() as a fast-service endpoint so it can be
dispatched by the loadgen via the compose stack.

NB: do NOT add ``from __future__ import annotations``.
"""

import importlib.util
import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from fast_service import FastServiceManager, RequestContext

CLAUDE_MCP_RUN_FSM = FastServiceManager()

# Import run_mcp_atlas from scripts/ (not a package)
_SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
_spec = importlib.util.spec_from_file_location(
    "run_mcp_atlas", str(_SCRIPTS_DIR / "run_mcp_atlas.py")
)
_run_mcp_atlas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_run_mcp_atlas)


class McpTaskRequest(BaseModel):
    task_id: str
    prompt: str
    enabled_tools: list
    servers: list = []
    output_dir: str = "./output/mcp-atlas"
    model: str = "deepseek-v4-flash"
    timeout: int = 300

    class Config:
        arbitrary_types_allowed = True


class McpTaskResponse(BaseModel):
    task_id: str
    exit_code: Optional[int] = None
    wall_s: float
    timed_out: bool
    error: Optional[str] = None
    num_turns: Optional[int] = None
    duration_api_ms: Optional[float] = None
    output_dir: str = ""
    session_jsonl_path: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


@CLAUDE_MCP_RUN_FSM.fast_service
def run_mcp_task(
    request: McpTaskRequest,
    fsm_context: RequestContext = None,
) -> McpTaskResponse:
    """Run a single mcp-atlas task through claude CLI."""
    agent_env_url = os.environ.get("AGENT_ENV_URL", "http://agent-environment:1984")

    task = {
        "task_id": request.task_id,
        "prompt": request.prompt,
        "enabled_tools": request.enabled_tools,
        "servers": request.servers,
    }

    result = _run_mcp_atlas.run_task(
        task=task,
        model=request.model,
        output_dir=request.output_dir,
        timeout=request.timeout,
        agent_env_url=agent_env_url,
    )

    return McpTaskResponse(
        task_id=result["task_id"],
        exit_code=result.get("exit_code"),
        wall_s=result["wall_s"],
        timed_out=result["timed_out"],
        error=result.get("error"),
        num_turns=result.get("num_turns"),
        duration_api_ms=result.get("duration_api_ms"),
        output_dir=result.get(
            "output_dir", str(Path(request.output_dir) / request.task_id)
        ),
        session_jsonl_path=result.get("session_jsonl_path"),
    )
