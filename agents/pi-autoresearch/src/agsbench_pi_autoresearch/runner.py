"""Single-task runner for pi-autoresearch on MLE-bench competitions.

pi-autoresearch is a Node.js extension for the pi coding agent that runs
autonomous experiment loops (init_experiment / run_experiment /
log_experiment). For an AutoML research task this runner:

1. Builds a fresh workspace per competition: the public competition data
   (read-only, ``data/``), an empty ``submission/``, the MLE-bench
   instructions adapted to that layout, and a validate_submission.sh helper
   that talks to the grading sidecar. The workspace is a git repo, since
   autoresearch commits/reverts experiments.
2. Invokes pi in print mode with the autoresearch setup skill and the
   MLE-bench prompt, under the task time limit.
3. Grades ``submission/submission.csv`` with the MLE-bench grader (the
   sidecar holds the private answers; the agent never sees them).
4. Returns the grade plus the agent's experiment log and token usage.

Because pi is a Node.js CLI tool, we shell out to it rather than calling
a Python library. Each task is fully isolated.
"""

import json
import os
import shutil
import signal
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

import requests

from fast_service import RequestContext

from agsbench_pi_autoresearch.fsm import agsbench_pi_autoresearch_fsm


_PREDS_LOCK = Lock()

# Grading sidecar (agents/pi-autoresearch/grader). /validate is agent-facing;
# /grade needs GRADER_TOKEN, which is stripped from the agent's environment.
DEFAULT_GRADER_URL = "http://127.0.0.1:5005"
# Python env for the agent's training code (see agent-env-requirements.txt).
DEFAULT_AGENT_PYTHON_ENV = "/opt/agent-env"


@dataclass
class RunResult:
    success: bool
    task_id: str
    competition_id: Optional[str] = None
    score: Optional[float] = None
    valid_submission: bool = False
    any_medal: bool = False
    medal: Optional[str] = None
    above_median: bool = False
    grade_report: Optional[Dict[str, Any]] = None
    final_metric: Optional[float] = None
    baseline_metric: Optional[float] = None
    improvement_pct: Optional[float] = None
    n_experiments: int = 0
    n_kept: int = 0
    agent_output: str = ""
    duration_seconds: float = 0.0
    timed_out: bool = False
    error: Optional[str] = None
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    token_usage: Optional[Dict[str, Any]] = None


@agsbench_pi_autoresearch_fsm.fast_service(monitor_only=True)
def pi_agent_call(
    task_id: str,
    project_dir: str,
    prompt: str,
    timeout: int,
    pi_binary: str,
    env: dict,
    fsm_context: RequestContext = None,
) -> dict:
    """Run the pi agent in print mode with the autoresearch extension.

    Uses ``pi -p "<prompt>"`` which triggers print mode (single-shot).
    Session JSONL files (with per-turn token usage) are written to the
    directory specified by PI_CODING_AGENT_SESSION_DIR in env.

    pi runs in its own process group so that on timeout the training
    processes it spawned are killed with it.
    """
    # Build the command. pi -p triggers print mode automatically.
    # If pi_binary is a .js file, invoke with node; otherwise call directly.
    if pi_binary.endswith(".js"):
        cmd = ["node", pi_binary]
    else:
        cmd = [pi_binary]
    cmd.extend(["-p", prompt])

    # Add provider/model/key from env if available
    provider = env.get("PI_PROVIDER", "")
    model = env.get("PI_MODEL", "")
    api_key = env.get("PI_API_KEY", "") or env.get("DEEPSEEK_API_KEY", "")
    thinking = env.get("PI_THINKING", "")

    if provider:
        cmd.extend(["--provider", provider])
    if model:
        cmd.extend(["--model", model])
    if api_key:
        cmd.extend(["--api-key", api_key])
    if thinking:
        cmd.extend(["--thinking", thinking])

    proc = subprocess.Popen(
        cmd,
        cwd=project_dir,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout if timeout > 0 else None)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()

    return {
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "stdout": stdout[-10000:] if stdout else "",
        "stderr": stderr[-5000:] if stderr else "",
    }


@agsbench_pi_autoresearch_fsm.fast_service(monitor_only=True)
def env_execute_grade(
    competition_id: str,
    submission_path: str,
    grader_url: str,
    grader_token: str,
    fsm_context: RequestContext = None,
) -> dict:
    """Grade the final submission with the MLE-bench grader sidecar."""
    with open(submission_path, "rb") as f:
        resp = requests.post(
            f"{grader_url}/grade",
            params={"competition_id": competition_id},
            headers={"X-Grader-Token": grader_token} if grader_token else {},
            files={"file": ("submission.csv", f, "text/csv")},
            timeout=1800,
        )
    resp.raise_for_status()
    return resp.json()


def _parse_session_tokens(session_dir: str) -> Optional[Dict[str, Any]]:
    """Parse pi session JSONL files to extract per-turn token usage.

    Pi writes one JSONL file per session to the session directory. Each
    assistant message entry contains a usage object with input, output,
    cacheRead, cacheWrite, totalTokens, and cost breakdowns.
    """
    import glob

    jsonl_files = sorted(
        glob.glob(os.path.join(session_dir, "**", "*.jsonl"), recursive=True)
    )
    if not jsonl_files:
        return None

    turns = []
    total_input = 0
    total_output = 0
    total_cache_read = 0
    total_cache_write = 0
    total_cost = 0.0

    for jsonl_path in jsonl_files:
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if entry.get("type") != "message":
                    continue
                msg = entry.get("message", {})
                if msg.get("role") != "assistant" or "usage" not in msg:
                    continue

                usage = msg["usage"]
                turn_data = {
                    "input": usage.get("input", 0),
                    "output": usage.get("output", 0),
                    "cacheRead": usage.get("cacheRead", 0),
                    "cacheWrite": usage.get("cacheWrite", 0),
                    "totalTokens": usage.get("totalTokens", 0),
                    "cost": usage.get("cost", {}),
                }
                turns.append(turn_data)
                total_input += turn_data["input"]
                total_output += turn_data["output"]
                total_cache_read += turn_data["cacheRead"]
                total_cache_write += turn_data["cacheWrite"]
                cost = turn_data["cost"]
                if isinstance(cost, dict):
                    total_cost += cost.get("total", 0.0)

    if not turns:
        return None

    return {
        "n_turns": len(turns),
        "total_input": total_input,
        "total_output": total_output,
        "total_cache_read": total_cache_read,
        "total_cache_write": total_cache_write,
        "total_cost": total_cost,
        "turns": turns,
    }


AUTO_DIR = ".auto"
# Print mode cannot run the /autoresearch command (it returns before its
# kickoff message is sent), so load the setup skill directly. The extension's
# tools are only active in autoresearch mode, which it enables at session
# start when .auto/log.jsonl exists -- run_one pre-creates that file.
KICKOFF_PREFIX = "/skill:autoresearch-create"
KICKOFF_SUFFIX = "Run autonomously; do not ask questions."


def _setup_autoresearch_session(project_dir: str, max_iterations: int) -> None:
    auto_dir = os.path.join(project_dir, AUTO_DIR)
    os.makedirs(auto_dir, exist_ok=True)
    open(os.path.join(auto_dir, "log.jsonl"), "a").close()
    if max_iterations > 0:
        with open(os.path.join(auto_dir, "config.json"), "w") as f:
            json.dump({"maxIterations": max_iterations}, f)


def _parse_autoresearch_jsonl(project_dir: str) -> List[Dict[str, Any]]:
    """Parse the session log (.auto/log.jsonl) to extract experiment results."""
    jsonl_path = os.path.join(project_dir, AUTO_DIR, "log.jsonl")
    if not os.path.exists(jsonl_path):
        jsonl_path = os.path.join(project_dir, "autoresearch.jsonl")  # legacy layout
    if not os.path.exists(jsonl_path):
        return []
    entries = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    return entries


def _compute_metrics(experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary metrics from experiment entries."""
    if not experiments:
        return {}

    run_entries = [e for e in experiments if e.get("type") != "hook" and "metric" in e]
    if not run_entries:
        return {}

    baseline = run_entries[0].get("metric") if run_entries else None
    kept = [e for e in run_entries if e.get("status") == "keep"]
    best = None
    if kept:
        # Determine direction from config entries or first run entry
        direction = "lower"
        for e in experiments:
            if e.get("type") == "config" and "bestDirection" in e:
                direction = e["bestDirection"]
                break
        if direction == "lower":
            best = min(e["metric"] for e in kept)
        else:
            best = max(e["metric"] for e in kept)
    elif baseline is not None:
        best = baseline

    improvement_pct = None
    if baseline is not None and best is not None and baseline != 0:
        improvement_pct = ((baseline - best) / abs(baseline)) * 100

    return {
        "baseline": baseline,
        "best": best,
        "improvement_pct": improvement_pct,
        "n_total": len(run_entries),
        "n_kept": len(kept),
    }


def _save_trajectory(
    instance_dir: Path,
    task_id: str,
    agent_output: str,
    experiments: List[Dict[str, Any]],
    extra: Dict[str, Any],
) -> None:
    """Write per-task trajectory to <instance_dir>."""
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / f"{task_id}.traj.json").write_text(
        json.dumps(
            {
                "task_id": task_id,
                "agent_output": agent_output[-50000:],
                "experiments": experiments,
                "info": extra,
            },
            indent=2,
        )
    )
    if agent_output:
        (instance_dir / f"{task_id}.output.md").write_text(agent_output)


def _update_preds(output_dir: Path, task_id: str, result: "RunResult"):
    """Append (or replace) a row in trajectories/preds.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "preds.json"
    with _PREDS_LOCK:
        data: Dict[str, Any] = {}
        if preds_path.exists():
            try:
                data = json.loads(preds_path.read_text())
            except json.JSONDecodeError:
                data = {}
        data[task_id] = {
            "task_id": task_id,
            "competition_id": result.competition_id,
            "success": result.success,
            "score": result.score,
            "valid_submission": result.valid_submission,
            "any_medal": result.any_medal,
            "medal": result.medal,
            "above_median": result.above_median,
            "grade_report": result.grade_report,
            "final_metric": result.final_metric,
            "baseline_metric": result.baseline_metric,
            "improvement_pct": result.improvement_pct,
            "n_experiments": result.n_experiments,
            "n_kept": result.n_kept,
            "duration_seconds": result.duration_seconds,
            "timed_out": result.timed_out,
            "error": result.error,
            "token_usage": (
                {
                    "n_turns": result.token_usage["n_turns"],
                    "total_input": result.token_usage["total_input"],
                    "total_output": result.token_usage["total_output"],
                    "total_cache_read": result.token_usage["total_cache_read"],
                    "total_cost": result.token_usage["total_cost"],
                }
                if result.token_usage
                else None
            ),
        }
        preds_path.write_text(json.dumps(data, indent=2))


# MLE-bench environment/instructions.txt (@ 507f92e1), with /home/data,
# /home/submission and the validation server adapted to this workspace.
MLEBENCH_INSTRUCTIONS = """\
You are participating in *MLE-bench*, an offline version of Kaggle competitions adapted to test the coding ability of agents like yourself.

You will be given a machine learning task. You must solve the task by training a model and running the model on the test set to produce a submission file. You may not hand-label the data to produce a submission file.

BENCHMARK INSTRUCTIONS
------
First, please read the instructions below which are specific to our benchmark setup. Note, these instructions can be found again in `instructions.md` in your working directory.

- **Competition-specific instructions**: The instructions for the competition are available in `data/description.md`. Read this file to understand the problem setup for the competition you are currently solving.
- **Dataset**: The dataset files for the competition are available in `data/` (read-only).
- **Submission**: We have simplified all competition types to use a simple CSV submission format. Regardless of what the competition instructions say, you MUST produce a submission file at `submission/submission.csv`. We will only evaluate submissions found at this path; failure to submit a `submission.csv` at this path will result in a failed attempt.
- **Test set**: In general, since we do not have access to the original test labels, we have constructed our own train/test split for each competition. As such, the dataset you see in `data/` may differ slightly from what is described in the competition instructions. Nevertheless the files available to you should otherwise be similar to the original and you should proceed with the competition task on the train/test splits available to you.
- **Submission checker**: We provide a tool for checking the validity of your submission file, in the form of a server endpoint at `{validate_url}`. If you post a submission file to the server (e.g. `curl -X POST -F "file=@submission/submission.csv" "{validate_url}"`), you will receive a response message telling you if your submission is valid or not (note that this does NOT give you a score). For your convenience, we provide a script `validate_submission.sh` which can help you make this submission to the validation server.
- **Cheating**: You are not allowed to view others' solutions to this competition. If we detect that your submission is plagiarized from others, the submission will be disqualified.

Your submission will be graded based on the metric in the competition-specific instructions. Your goal is to achieve the best score possible on the competition task.

Wherever there is a conflict between the points above and the competition-specific instructions in `data/description.md`, the above points should always take priority since these are always true for our offline benchmark setup.

ADDITIONAL NOTES
------

- **Compute**: You have access to {hardware} with the appropriate drivers installed. Use `python` (it has numpy, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, PyTorch, transformers, ...).
- **Total Runtime**: You have a maximum of {time_limit} to attempt this task.
- **Approach**: Use the autoresearch tools (init_experiment, run_experiment, log_experiment) to iteratively improve your ML solution. The autoresearch metric should be your cross-validation score or a proxy for the competition metric; its direction (lower/higher is better) depends on the competition.
- **Submission**: Keep your best submission at `submission/submission.csv` at all times.
"""

# Same workflow as the reference pi-autoresearch MLE-bench agent.
MLEBENCH_WORKFLOW = """\
IMPORTANT WORKFLOW:
1. First read and understand the competition task from the instructions above.
2. Explore the data in data/ to understand its structure.
3. Build a baseline model that produces a valid submission at submission/submission.csv.
4. Validate your submission: ./validate_submission.sh
5. Once you have a working baseline, use the autoresearch tools (init_experiment, run_experiment, log_experiment) to iteratively improve your score.
6. For each iteration: modify code, train, predict, save submission.csv, validate, log the result.
7. Keep your best submission.csv at submission/submission.csv at all times.
8. NEVER stop early. Use all available time to improve."""

VALIDATE_SCRIPT = """\
#!/bin/bash
# Usage: ./validate_submission.sh [submission.csv]   (checks the format only; no score)
curl -s -X POST -F "file=@${{1:-submission/submission.csv}}" "{validate_url}"
echo
"""


def _detect_hardware() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
        names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
    except (OSError, subprocess.SubprocessError):
        names = []
    if not names:
        return "a CPU"
    counts: Dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return ", ".join(f"{c} {n}" for n, c in sorted(counts.items()))


def _format_time(seconds: int) -> str:
    if seconds <= 0:
        return "no fixed time limit"
    return f"{seconds // 3600}hrs {(seconds % 3600) // 60}mins"


def _build_workspace(
    workspace: Path,
    data_dir: Path,
    competition_id: str,
    grader_url: str,
    timeout: int,
    max_iterations: int,
) -> str:
    """Lay out the agent workspace and return the instructions text."""
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True)
    (workspace / "data").symlink_to(data_dir.resolve(), target_is_directory=True)
    (workspace / "submission").mkdir()

    validate_url = f"{grader_url}/validate?competition_id={competition_id}"
    instructions = MLEBENCH_INSTRUCTIONS.format(
        validate_url=validate_url,
        hardware=_detect_hardware(),
        time_limit=_format_time(timeout),
    )
    description = (data_dir / "description.md").read_text(errors="replace")
    instructions += "\nCOMPETITION INSTRUCTIONS\n------\n\n" + description
    (workspace / "instructions.md").write_text(instructions)

    script = workspace / "validate_submission.sh"
    script.write_text(VALIDATE_SCRIPT.format(validate_url=validate_url))
    script.chmod(0o755)
    # submission/ stays tracked: on discard, autoresearch reverts it to the
    # last kept experiment, which keeps the best submission in place.
    (workspace / ".gitignore").write_text(f"data\n{AUTO_DIR}/\n__pycache__/\n")

    _setup_autoresearch_session(str(workspace), max_iterations)
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=workspace, check=True)
    return instructions


def _agent_env(env_overrides: Optional[Dict[str, str]]) -> Dict[str, str]:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    env.pop("GRADER_TOKEN", None)  # the agent may validate, never grade
    venv = env.get("AGENT_PYTHON_ENV", DEFAULT_AGENT_PYTHON_ENV)
    if os.path.isdir(os.path.join(venv, "bin")):
        env["VIRTUAL_ENV"] = venv
        env["PATH"] = os.path.join(venv, "bin") + os.pathsep + env.get("PATH", "")
    return env


def run_one(
    task_id: str,
    competition_id: str,
    data_root: str,
    max_iterations: int,
    timeout: int,
    output_dir: str,
    pi_binary: str,
    env_overrides: Optional[Dict[str, str]],
    fsm_context: Optional[RequestContext],
) -> RunResult:
    """Execute one MLE-bench competition end-to-end."""
    out_root = Path(output_dir)
    instance_dir = out_root / task_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    workspace = instance_dir / "workspace"
    grader_url = os.environ.get("GRADER_URL", DEFAULT_GRADER_URL).rstrip("/")
    grader_token = os.environ.get("GRADER_TOKEN", "")

    # Set up pi session directory so session JSONL files are materialized
    session_dir = os.path.join(str(instance_dir), "pi-sessions")
    shutil.rmtree(session_dir, ignore_errors=True)
    os.makedirs(session_dir, exist_ok=True)

    start_time = time.time()
    agent_output = ""
    error = None
    timed_out = False
    report: Optional[Dict[str, Any]] = None

    try:
        data_dir = Path(data_root) / competition_id
        if not (data_dir / "description.md").exists():
            raise FileNotFoundError(
                f"competition data not prepared: {data_dir} "
                "(run scripts/1.prepare_data.sh)")
        instructions = _build_workspace(
            workspace, data_dir, competition_id, grader_url, timeout, max_iterations)

        env = _agent_env(env_overrides)
        env["PI_CODING_AGENT_SESSION_DIR"] = session_dir
        prompt = (
            f"{KICKOFF_PREFIX} You are an ML engineer solving a Kaggle competition. "
            f"Read the instructions below carefully.\n\n{instructions}\n\n"
            f"{MLEBENCH_WORKFLOW}\n\n{KICKOFF_SUFFIX} Start now."
        )
        result = pi_agent_call(
            task_id=task_id,
            project_dir=str(workspace),
            prompt=prompt,
            timeout=timeout,
            pi_binary=pi_binary,
            env=env,
            fsm_context=fsm_context,
        )
        agent_output = result.get("stdout", "")
        timed_out = bool(result.get("timed_out"))
        # The time limit is the task budget ("never stop early"), so running
        # into it is a normal finish; the submission is graded either way.
        if not timed_out and result.get("returncode", 0) != 0:
            stderr = result.get("stderr", "")
            error = f"pi exited with code {result['returncode']}: {stderr[-2000:]}"
    except Exception as e:
        error = str(e)
        traceback.print_exc()

    duration = time.time() - start_time

    experiments = _parse_autoresearch_jsonl(str(workspace))
    metrics = _compute_metrics(experiments)

    # Reaching maxIterations makes the extension abort the agent, so pi
    # exits non-zero ("This operation was aborted"); that is a normal finish.
    if (error and error.startswith("pi exited") and max_iterations > 0
            and metrics.get("n_total", 0) >= max_iterations):
        error = None

    submission = workspace / "submission" / "submission.csv"
    if submission.is_file():
        try:
            report = env_execute_grade(
                competition_id=competition_id,
                submission_path=str(submission),
                grader_url=grader_url,
                grader_token=grader_token,
                fsm_context=fsm_context,
            )
        except Exception as e:
            error = error or f"grading failed: {e}"
            traceback.print_exc()
    elif error is None:
        error = "no submission/submission.csv produced"

    report = report or {}
    medal = next((m for m in ("gold", "silver", "bronze") if report.get(f"{m}_medal")), None)
    valid = bool(report.get("valid_submission"))

    # Parse token usage from pi session JSONL files
    token_usage = _parse_session_tokens(session_dir)

    run_result = RunResult(
        success=valid and error is None,
        task_id=task_id,
        competition_id=competition_id,
        score=report.get("score"),
        valid_submission=valid,
        any_medal=bool(report.get("any_medal")),
        medal=medal,
        above_median=bool(report.get("above_median")),
        grade_report=report or None,
        final_metric=metrics.get("best"),
        baseline_metric=metrics.get("baseline"),
        improvement_pct=metrics.get("improvement_pct"),
        n_experiments=metrics.get("n_total", 0),
        n_kept=metrics.get("n_kept", 0),
        agent_output=agent_output,
        duration_seconds=duration,
        timed_out=timed_out,
        error=error,
        experiments=experiments,
        token_usage=token_usage,
    )

    _save_trajectory(
        instance_dir,
        task_id,
        agent_output,
        experiments,
        {
            "competition_id": competition_id,
            "success": run_result.success,
            "grade": report or None,
            "error": error,
            "timed_out": timed_out,
            "duration_seconds": duration,
            "metrics": metrics,
            "token_usage": token_usage,
        },
    )
    _update_preds(out_root, task_id, run_result)

    return run_result
