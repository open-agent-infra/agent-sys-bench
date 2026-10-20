"""Fast-service entrypoint for the OSWorld benchmark service.

Exposes ``solve_task`` which drives one OSWorld task end-to-end:
``DesktopEnv.reset(task_config)`` (reverts the per-task VM snapshot,
runs the task's setup) -> loop of LLM-picked pyautogui actions ->
final reward from the task's built-in evaluator.

Each task runs in its OWN subprocess (``multiprocessing.get_context("spawn")``).
Upstream's `run_multienv_kimi_k25.py` does the same via process-per-env
multiprocessing — OSWorld's docker provider has a process-global port-
allocation lock that contends on concurrent env.reset() calls in a
single Python process. Running each env in a fresh process gives every
worker its own lock instance.

NB: no ``from __future__ import annotations`` (see fast-service-gotchas).
"""

import multiprocessing as mp
import os
import traceback
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from fast_service import RequestContext

from agsbench_gui_agent.fsm import agsbench_gui_agent_fsm


class SolveTaskRequest(BaseModel):
    task_id: str
    task_config: Dict[str, Any] = Field(
        description="The full upstream OSWorld task JSON (instruction, config, evaluator, ...).",
    )
    output_dir: str = Field(default="/workspace/os_world/trajectories")
    max_steps: int = Field(default=15)
    observation_type: str = Field(
        default="a11y_tree",
        description="One of: 'a11y_tree' (text-only, works with non-VLM models), 'screenshot' (needs VLM), 'screenshot_a11y_tree' (both).",
    )


class SolveTaskResponse(BaseModel):
    success: bool
    task_id: str
    reward: float = 0.0
    n_steps: int = 0
    error: Optional[str] = None


def _run_one_in_subprocess(
    task_id: str,
    task_config: Dict[str, Any],
    output_dir: str,
    max_steps: int,
    observation_type: str,
    q: "mp.Queue",
) -> None:
    """Run the OSWorld agent loop for one task in a fresh subprocess.

    Imports happen INSIDE this function so the child doesn't inherit any
    OSWorld / DesktopEnv state from the parent (most importantly, the
    in-process port-allocation lock and any cached docker.SDK clients).
    Writes telemetry into the same bind-mounted
    .cache/os-world/telemetry/usage.*.jsonl that the parent process
    reads — each process gets its own file (filename includes pid).
    """
    try:
        from agsbench_gui_agent import runner as runner_mod  # noqa: WPS433  intentional

        result = runner_mod.run_one(
            task_id=task_id,
            task_config=task_config,
            output_dir=output_dir,
            max_steps=max_steps,
            observation_type=observation_type,
            fsm_context=None,  # cannot pickle RequestContext into child
        )
        q.put(
            {
                "success": result.success,
                "task_id": result.task_id,
                "reward": result.reward,
                "n_steps": result.n_steps,
                "error": result.error,
            }
        )
    except Exception as exc:  # noqa: BLE001 - we want to surface anything
        q.put(
            {
                "success": False,
                "task_id": task_id,
                "reward": 0.0,
                "n_steps": 0,
                "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}",
            }
        )


# `spawn` context so the child does NOT inherit module imports from the
# parent — DesktopEnv / docker SDK state would otherwise carry over and
# defeat the point of process-per-task isolation.
_MP_CTX = mp.get_context("spawn")


@agsbench_gui_agent_fsm.fast_service
def solve_task(
    req: SolveTaskRequest,
    fsm_context: RequestContext = None,
) -> SolveTaskResponse:
    q: "mp.Queue" = _MP_CTX.Queue()
    proc = _MP_CTX.Process(
        target=_run_one_in_subprocess,
        kwargs=dict(
            task_id=req.task_id,
            task_config=req.task_config,
            output_dir=req.output_dir,
            max_steps=req.max_steps,
            observation_type=req.observation_type,
            q=q,
        ),
        name=f"os_world_worker_{req.task_id[:8]}",
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0 and q.empty():
        return SolveTaskResponse(
            success=False,
            task_id=req.task_id,
            error=f"worker subprocess died (exitcode={proc.exitcode})",
        )

    result_dict = q.get_nowait() if not q.empty() else {}
    return SolveTaskResponse(
        success=result_dict.get("success", False),
        task_id=result_dict.get("task_id", req.task_id),
        reward=result_dict.get("reward", 0.0),
        n_steps=result_dict.get("n_steps", 0),
        error=result_dict.get("error"),
    )
