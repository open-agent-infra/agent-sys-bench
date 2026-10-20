"""Single-task runner for OSWorld.

OSWorld runs each task in a full Ubuntu VM (QEMU-in-Docker via the host's
/dev/kvm). One ``DesktopEnv`` instance per task; env.reset reverts to a
clean snapshot before running the task's setup_config. Per-task isolation
is BUILT IN to OSWorld (snapshot revert), so we don't add a cpuset pool
here — the VM provider already pins resources via DISK_SIZE / RAM_SIZE /
CPU_CORES on its container (see desktop_env/providers/docker/provider.py).

The agent: a tiny LLM-driven mini-agent that reads the accessibility tree
(or screenshot if the configured LLM is a VLM) and emits a pyautogui
string per step. Mirrors the bundled mm_agents/prompt_agent.py but
DeepSeek-friendly.

NB: no ``from __future__ import annotations`` (see fast-service-gotchas).
"""

import datetime as _dt
import json
import logging
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from fast_service import RequestContext

from agsbench_gui_agent.fsm import agsbench_gui_agent_fsm


_PREDS_LOCK = Lock()

# === Session logger (per-task LLM call transcript) ==========================


class SessionLogger:
    """Writes a per-task JSONL file capturing every LLM call and env step.

    Each line is a JSON object with an ``event`` key:
    - session_start  — emitted when the session opens
    - llm_call       — one record per llm_pick_action_call invocation
    - env_step       — one record per env_step_call invocation
    - session_end    — emitted in the finally block

    Files land at ``<output_dir>/<task_id>/<task_id>.session.jsonl`` so
    they coexist with the existing traj.json in the per-task directory.
    """

    def __init__(self, output_dir, task_id):
        self._path = Path(output_dir) / task_id / f"{task_id}.session.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._task_id = task_id
        self._write(
            {
                "event": "session_start",
                "task_id": task_id,
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            }
        )

    def _write(self, record):
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _sanitize_messages(messages):
        """Return a copy of messages with base64 image data removed."""
        if not messages:
            return messages
        out = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        url = (part.get("image_url") or {}).get("url", "")
                        if url.startswith("data:"):
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "<base64-omitted>"},
                                }
                            )
                        else:
                            parts.append(part)
                    else:
                        parts.append(part)
                out.append({**msg, "content": parts})
            else:
                out.append(msg)
        return out

    def log_llm_call(
        self,
        *,
        model,
        messages,
        response_text,
        actions,
        usage=None,
        duration_s=None,
        step=None,
        extra=None,
    ):
        sanitized_msgs = self._sanitize_messages(messages)
        record = {
            "event": "llm_call",
            "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
            "step": step if step is not None else self._step,
            "model": model,
            "messages": sanitized_msgs,
            "response": (
                (
                    response_text
                    if isinstance(response_text, str)
                    else json.dumps(response_text, ensure_ascii=False)
                )[:4000]
                if response_text
                else None
            ),
            "actions": actions,
            "usage": usage,
            "duration_s": round(duration_s, 4) if duration_s is not None else None,
        }
        if extra:
            record.update(extra)
        self._write(record)
        self._step += 1

    def log_env_step(self, *, step, action, reward, done):
        self._write(
            {
                "event": "env_step",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
                "step": step,
                "action": action,
                "reward": reward,
                "done": done,
            }
        )

    def log_session_end(self, *, reward, n_steps, error=None):
        self._write(
            {
                "event": "session_end",
                "timestamp": _dt.datetime.utcnow().isoformat() + "Z",
                "reward": reward,
                "n_steps": n_steps,
                "error": error,
            }
        )


@dataclass
class RunResult:
    success: bool
    task_id: str
    reward: float = 0.0
    n_steps: int = 0
    actions: List[str] = None
    error: Optional[str] = None


# === Timing helpers (monitor_only) ==========================================


_USAGE_LOG = logging.getLogger("os_world.telemetry")
# fast-service leaves the root logger at WARNING with no handlers, so an
# `info()` from a sibling logger would be silently dropped. Attach a
# stderr handler at INFO directly, and don't propagate (we don't want
# fast-service's monitor middleware to swallow these lines either).
#
# We ALSO attach a FileHandler writing to .cache/os-world/telemetry/
# usage.jsonl so the structured token / action records survive container
# teardown — `docker logs` is fragile (returns empty after `compose down`).
# The .cache/ tree is bind-mounted from the host so this file persists.
if not _USAGE_LOG.handlers:
    _USAGE_LOG.setLevel(logging.INFO)
    _USAGE_LOG.propagate = False

    _stderr_h = logging.StreamHandler()
    _stderr_h.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    _USAGE_LOG.addHandler(_stderr_h)

    # Per-process telemetry file so child workers don't race on a single
    # FileHandler's writes. Postprocess script globs `usage.*.jsonl` and
    # concatenates.
    _file_dir = os.environ.get(
        "OS_WORLD_TELEMETRY_DIR",
        "/workspace/os_world/.cache/os-world/telemetry",
    )
    _file_path = os.path.join(_file_dir, f"usage.{os.getpid()}.jsonl")
    try:
        os.makedirs(_file_dir, exist_ok=True)
        _file_h = logging.FileHandler(_file_path, mode="a", encoding="utf-8")
        _file_h.setFormatter(
            logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
        )
        _USAGE_LOG.addHandler(_file_h)
    except Exception as _e:  # noqa: BLE001 - non-fatal if telemetry path unwritable
        _USAGE_LOG.warning("could not attach FileHandler to %s: %s", _file_path, _e)


@agsbench_gui_agent_fsm.fast_service(monitor_only=True)
def env_step_call(
    env: Any,
    action: str,
    task_id: str = "-",
    fsm_context: RequestContext = None,
) -> dict:
    """Timed wrapper around ``DesktopEnv.step``. Action is a pyautogui string
    (or 'WAIT' / 'FAIL' / 'DONE' sentinels).

    Passes `sleep_after_execution=5.0` (upstream Kimi launcher default,
    `run_multienv_kimi_k25.py:63`) so the desktop has time to render
    before the next screenshot is captured. Skipping this — env.step's
    own default is 0 — was causing the agent to act on stale state.

    Emits a structured `OS_WORLD_TOOL` log line so a post-processor can
    build a fast-service-shaped tool_calls.csv.
    """
    t0 = time.time()
    sleep_after = float(os.environ.get("OS_WORLD_SLEEP_AFTER", "5.0"))
    obs, reward, done, info = env.step(action, sleep_after)
    env.last_obs = obs
    dur = time.time() - t0
    _USAGE_LOG.info(
        "OS_WORLD_TOOL task=%s action=%r reward=%.4f done=%s dur=%.4f",
        task_id,
        (action or "")[:200],
        float(reward),
        bool(done),
        dur,
    )
    return {
        "reward": float(reward),
        "done": bool(done),
        "obs_keys": list(obs.keys()) if isinstance(obs, dict) else [],
    }


@agsbench_gui_agent_fsm.fast_service(monitor_only=True)
def llm_pick_action_call(
    agent: Any,
    instruction: str,
    obs: Any,
    task_id: str = "-",
    fsm_context: RequestContext = None,
) -> dict:
    """Timed wrapper around `KimiPromptAgent.predict()`. The agent
    instance is held by `run_one` for the full task — its `predict()`
    builds the OpenAI-shape multi-turn messages from accumulated
    `self.observations + self.actions + self.thoughts`, so the
    prompt prefix is stable and Moonshot/DeepSeek prompt caches hit.

    Emits a structured `OS_WORLD_USAGE` log line per call so the
    post-processor can build a fast-service-shaped llm_calls.csv.
    """
    t0 = time.time()
    # KimiAgent.predict → (response, actions, info_dict)  — 3-tuple
    # Owl15Agent.predict → (response, actions)            — 2-tuple
    result = agent.predict(instruction, obs)
    if len(result) == 3:
        response, actions, _info_dict = result
    else:
        response, actions = result
    dur = time.time() - t0

    # Token telemetry. Available for KimiAgent (raw HTTP response stashed
    # by our shim). Owl15Agent uses an OpenAI SDK client we don't intercept;
    # local vLLM has no per-token cost so we just record what we can.
    raw = getattr(agent, "_last_response", None) or {}
    usage = raw.get("usage") or {}
    prompt_t = usage.get("prompt_tokens", 0) or 0
    completion_t = usage.get("completion_tokens", 0) or 0
    total_t = usage.get("total_tokens", 0) or 0
    cdet = usage.get("completion_tokens_details") or {}
    pdet = usage.get("prompt_tokens_details") or {}
    reasoning_t = cdet.get("reasoning_tokens", 0) or 0
    cached_t = pdet.get("cached_tokens", 0) or 0
    msg = (raw.get("choices") or [{}])[0].get("message") or {}
    rc = msg.get("reasoning_content") or ""
    reasoning_chars = len(rc)

    _USAGE_LOG.info(
        "OS_WORLD_USAGE task=%s model=%s prompt=%d completion=%d reasoning=%d cached=%d total=%d reasoning_chars=%d dur=%.4f",
        task_id,
        agent.model,
        prompt_t,
        completion_t,
        reasoning_t,
        cached_t,
        total_t,
        reasoning_chars,
        dur,
    )
    return {"response": response, "actions": actions}


# === Agent =================================================================
#
# Use OSWorld upstream's OFFICIAL `KimiAgent` (mm_agents/kimi/) — that's
# the agent class used to produce the leaderboard numbers for Kimi K2.5
# and K2.6. The official settings (from mm_agents/kimi/README.md and
# scripts/python/run_multienv_kimi_k25.py):
#
#   --observation_type screenshot           # screenshot ONLY; no a11y tree
#   --max_image_history_length 8            # K2.6 default (3 for K2.5)
#   --temperature 1                         # Moonshot hosted requires 1
#   --thinking                              # use THINKING system prompt
#   --max_steps 100
#   --coordinate_type relative
#   --screen_size 1920x1080
#
# KimiAgent has its own predict() (returns 3-tuple: response, actions,
# info_dict), its own response parser, and its own Moonshot HTTP call
# (reads KIMI_API_KEY env var, endpoint hard-coded to
# https://api.moonshot.ai/v1/chat/completions).


def _make_agent(env: Any) -> Any:
    """Instantiate the official OSWorld agent for one task. Switched by
    `OS_WORLD_AGENT_KIND` env var:

    - "kimi" → upstream KimiAgent (`mm_agents.kimi.kimi_agent`) for
      Moonshot's hosted Kimi K2.5/K2.6.
    - "owl15" → upstream Owl15Agent (`mm_agents.owl15_agent`) for
      GUI-Owl-1.5-* served via any OpenAI-compatible endpoint (we use
      a locally-hosted vLLM serving `gui-owl-1.5-32b-instruct`).

    Imported lazily so fast-service registration in request.py runs first.
    """
    kind = os.environ.get("OS_WORLD_AGENT_KIND", "owl15").lower()
    if kind == "kimi":
        from agsbench_gui_agent.upstream_agent import KimiAgent

        return KimiAgent(
            env=env,
            model=os.environ.get("OS_WORLD_MODEL", "kimi-k2.6"),
            max_tokens=int(os.environ.get("OS_WORLD_MAX_TOKENS", "4096")),
            top_p=float(os.environ.get("OS_WORLD_TOP_P", "0.95")),
            temperature=float(os.environ.get("OS_WORLD_TEMPERATURE", "1.0")),
            action_space="pyautogui",
            observation_type="screenshot",
            screen_size=(1920, 1080),
            coordinate_type="relative",
            max_image_history_length=int(
                os.environ.get("OS_WORLD_MAX_IMAGE_HISTORY", "8")
            ),
            max_steps=int(os.environ.get("OS_WORLD_MAX_STEPS", "100")),
            thinking=os.environ.get("OS_WORLD_THINKING", "1") not in ("0", "false", ""),
        )
    if kind == "owl15":
        from agsbench_gui_agent.upstream_agent import Owl15Agent

        # Owl15Agent talks to any OpenAI-compatible endpoint (engine="openai").
        # Point it at our local vLLM via env vars; default to the SSH-
        # tunneled vLLM on host.docker.internal:14243 (see scripts/
        # ssh_tunnel.sh on the host side).
        api_url = (
            os.environ.get("OS_WORLD_API_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://host.docker.internal:14243/v1"
        )
        api_key = (
            os.environ.get("OS_WORLD_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "vllm-local"
        )
        return Owl15Agent(
            model=os.environ.get("OS_WORLD_MODEL", "gui-owl-1.5-32b-instruct"),
            api_url=api_url,
            api_key=api_key,
            engine="openai",
            action_space="pyautogui",
            observation_type="screenshot",
            history_n=int(os.environ.get("OS_WORLD_HISTORY_N", "5")),
            max_tokens=int(os.environ.get("OS_WORLD_MAX_TOKENS", "1000")),
            top_p=float(os.environ.get("OS_WORLD_TOP_P", "0.9")),
            top_k=int(os.environ.get("OS_WORLD_TOP_K", "1")),
            temperature=float(os.environ.get("OS_WORLD_TEMPERATURE", "0.0")),
            max_trajectory_length=int(os.environ.get("OS_WORLD_MAX_TRAJ", "50")),
            keep_full_response=int(os.environ.get("OS_WORLD_KEEP_FULL_RESPONSE", "0")),
        )
    if kind == "holo3":
        from openai import OpenAI
        import json as _json

        api_urls_str = os.environ.get("OS_WORLD_API_URLS", "")
        if api_urls_str:
            api_urls = [u.strip() for u in api_urls_str.split(",") if u.strip()]
        else:
            api_urls = [
                os.environ.get("OS_WORLD_API_URL")
                or os.environ.get("OPENAI_API_BASE")
                or os.environ.get("OPENAI_BASE_URL")
                or "http://host.docker.internal:14243/v1"
            ]
        api_key = (
            os.environ.get("OS_WORLD_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "vllm-local"
        )
        model = os.environ.get("OS_WORLD_MODEL", "holo3-35b-a3b")
        _selected_url = api_urls[os.getpid() % len(api_urls)]
        _holo3_client = OpenAI(base_url=_selected_url, api_key=api_key, timeout=120)

        # Holo3 system prompt with tool definitions
        _HOLO3_TOOLS = [
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "Click at (x, y) coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the target UI element to click on",
                            },
                            "x": {
                                "type": "integer",
                                "description": "X coordinate as integer in [0, 1000]",
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y coordinate as integer in [0, 1000]",
                            },
                            "button": {
                                "type": "string",
                                "enum": ["left", "right", "middle"],
                                "description": "Mouse button",
                            },
                        },
                        "required": ["description", "x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "double_click",
                    "description": "Double-click at (x, y) coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "x": {
                                "type": "integer",
                                "description": "X coordinate in [0, 1000]",
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y coordinate in [0, 1000]",
                            },
                        },
                        "required": ["description", "x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "right_click",
                    "description": "Right-click at (x, y) coordinates",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "x": {
                                "type": "integer",
                                "description": "X coordinate in [0, 1000]",
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y coordinate in [0, 1000]",
                            },
                        },
                        "required": ["description", "x", "y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "description": "Type text into the currently focused element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "Text to type"},
                            "clear_before": {
                                "type": "boolean",
                                "description": "Clear the field before typing",
                            },
                            "press_enter": {
                                "type": "boolean",
                                "description": "Press Enter after typing",
                            },
                        },
                        "required": ["text"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "hotkey",
                    "description": "Press a keyboard shortcut",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keys": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Keys to press simultaneously",
                            },
                        },
                        "required": ["keys"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "scroll",
                    "description": "Scroll at the current position",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "x": {
                                "type": "integer",
                                "description": "X coordinate in [0, 1000]",
                            },
                            "y": {
                                "type": "integer",
                                "description": "Y coordinate in [0, 1000]",
                            },
                            "direction": {
                                "type": "string",
                                "enum": ["up", "down", "left", "right"],
                            },
                            "amount": {
                                "type": "integer",
                                "description": "Scroll amount (1-10)",
                            },
                        },
                        "required": ["x", "y", "direction", "amount"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "drag",
                    "description": "Drag from one point to another",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_x": {"type": "integer"},
                            "start_y": {"type": "integer"},
                            "end_x": {"type": "integer"},
                            "end_y": {"type": "integer"},
                        },
                        "required": ["start_x", "start_y", "end_x", "end_y"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "wait",
                    "description": "Wait for the screen to update",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "seconds": {
                                "type": "integer",
                                "description": "Seconds to wait (1-10)",
                            },
                        },
                        "required": ["seconds"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "done",
                    "description": "Task is complete",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "success": {"type": "boolean"},
                        },
                        "required": ["success"],
                    },
                },
            },
        ]

        class Holo3Agent:
            """Holo3 agent: outputs {note, thought, tool_call} JSON per step.
            Coordinates are in [0, 1000] range, mapped to screen pixels.
            """

            def __init__(self):
                self.messages = []
                self.screen_w = 1920
                self.screen_h = 1080
                self.model = model
                self.system_prompt = """You are a computer use agent. You observe screenshots and execute actions to complete tasks.

Each step you MUST output a JSON object:
{"note": "...", "thought": "...", "tool_call": {"tool_name": "...", ...}}

Tools (coordinates x, y are integers in [0, 1000]):
- click: {"tool_name": "click", "description": "target element", "x": 500, "y": 500, "button": "left"}
- double_click: {"tool_name": "double_click", "description": "target", "x": 500, "y": 500}
- right_click: {"tool_name": "right_click", "description": "target", "x": 500, "y": 500}
- write: {"tool_name": "write", "text": "hello", "clear_before": false, "press_enter": false}
- hotkey: {"tool_name": "hotkey", "keys": ["ctrl", "s"]}
- scroll: {"tool_name": "scroll", "x": 500, "y": 500, "direction": "down", "amount": 3}
- drag: {"tool_name": "drag", "start_x": 100, "start_y": 100, "end_x": 500, "end_y": 500}
- wait: {"tool_name": "wait", "seconds": 2}
- done: {"tool_name": "done", "success": true}"""

            def reset(self):
                self.messages = []

            def _tool_call_to_pyautogui(self, tool_call):
                name = tool_call.get("tool_name", tool_call.get("name", ""))
                # Args are flat in tool_call: {tool_name, x, y, ...}
                args = {
                    k: v for k, v in tool_call.items() if k not in ("tool_name", "name")
                }
                # Also check nested "arguments" key (OpenAI tool call format)
                if "arguments" in tool_call:
                    nested = tool_call["arguments"]
                    if isinstance(nested, str):
                        nested = _json.loads(nested)
                    args.update(nested)

                if name in ("click", "left_click"):
                    x = int(args.get("x", 0)) * self.screen_w // 1000
                    y = int(args.get("y", 0)) * self.screen_h // 1000
                    btn = args.get("button", "left")
                    return (
                        f"import pyautogui; pyautogui.click({x}, {y}, button='{btn}')"
                    )
                elif name == "double_click":
                    x = int(args.get("x", 0)) * self.screen_w // 1000
                    y = int(args.get("y", 0)) * self.screen_h // 1000
                    return f"import pyautogui; pyautogui.doubleClick({x}, {y})"
                elif name == "right_click":
                    x = int(args.get("x", 0)) * self.screen_w // 1000
                    y = int(args.get("y", 0)) * self.screen_h // 1000
                    return (
                        f"import pyautogui; pyautogui.click({x}, {y}, button='right')"
                    )
                elif name == "write":
                    text = args.get("text", "")
                    clear = args.get("clear_before", False)
                    enter = args.get("press_enter", False)
                    code = "import pyautogui\n"
                    if clear:
                        code += (
                            "pyautogui.hotkey('ctrl', 'a'); pyautogui.press('delete')\n"
                        )
                    code += f"pyautogui.typewrite({repr(text)}, interval=0.05)"
                    if enter:
                        code += "; pyautogui.press('enter')"
                    return code
                elif name == "hotkey":
                    keys = args.get("keys", [])
                    return f"import pyautogui; pyautogui.hotkey({', '.join(repr(k) for k in keys)})"
                elif name == "scroll":
                    x = int(args.get("x", 500)) * self.screen_w // 1000
                    y = int(args.get("y", 500)) * self.screen_h // 1000
                    direction = args.get("direction", "down")
                    amount = int(args.get("amount", 3))
                    clicks = amount if direction == "up" else -amount
                    return f"import pyautogui; pyautogui.moveTo({x}, {y}); pyautogui.scroll({clicks})"
                elif name == "drag":
                    sx = int(args.get("start_x", 0)) * self.screen_w // 1000
                    sy = int(args.get("start_y", 0)) * self.screen_h // 1000
                    ex = int(args.get("end_x", 0)) * self.screen_w // 1000
                    ey = int(args.get("end_y", 0)) * self.screen_h // 1000
                    return f"import pyautogui; pyautogui.moveTo({sx}, {sy}); pyautogui.dragTo({ex}, {ey}, duration=1.0)"
                elif name == "wait":
                    secs = int(args.get("seconds", 2))
                    return f"import time; time.sleep({secs})"
                elif name == "done":
                    success = args.get("success", False)
                    return "DONE" if success else "FAIL"
                return "WAIT"

            def predict(self, instruction, obs):
                import base64 as _b64

                screenshot = obs.get("screenshot", b"")
                if isinstance(screenshot, bytes):
                    screenshot_b64 = _b64.b64encode(screenshot).decode()
                else:
                    screenshot_b64 = screenshot

                # Build messages
                if not self.messages:
                    self.messages.append(
                        {"role": "system", "content": self.system_prompt}
                    )
                    self.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Task: {instruction}"},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{screenshot_b64}"
                                    },
                                },
                            ],
                        }
                    )
                else:
                    # Add screenshot as tool result / new observation
                    self.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{screenshot_b64}"
                                    },
                                },
                            ],
                        }
                    )

                # Keep message history manageable (last N screenshots)
                max_images = int(os.environ.get("OS_WORLD_HISTORY_N", "5"))
                # Count image messages and trim old ones
                img_indices = [
                    i
                    for i, m in enumerate(self.messages)
                    if isinstance(m.get("content"), list)
                    and any(
                        p.get("type") == "image_url"
                        for p in m["content"]
                        if isinstance(p, dict)
                    )
                ]
                if len(img_indices) > max_images:
                    # Remove oldest image messages (keep first instruction + last N)
                    to_remove = img_indices[
                        1:-max_images
                    ]  # keep first (has instruction) and last N
                    self.messages = [
                        m for i, m in enumerate(self.messages) if i not in to_remove
                    ]

                try:
                    resp = _holo3_client.chat.completions.create(
                        model=model,
                        messages=self.messages,
                        max_tokens=2048,
                        temperature=float(
                            os.environ.get("OS_WORLD_TEMPERATURE", "0.6")
                        ),
                        top_p=float(os.environ.get("OS_WORLD_TOP_P", "0.95")),
                    )
                except Exception as e:
                    return str(e), ["FAIL"]

                choice = resp.choices[0]
                response_text = choice.message.content or ""

                # Parse tool calls - first try native tool_calls, then extract JSON
                if choice.message.tool_calls:
                    tc = choice.message.tool_calls[0]
                    tool_call = {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    self.messages.append(
                        {"role": "assistant", "content": response_text or ""}
                    )
                    pyautogui_code = self._tool_call_to_pyautogui(tool_call)
                    return response_text, [pyautogui_code]

                # Holo3 outputs reasoning text followed by JSON: extract the JSON part
                import re as _re

                json_match = _re.search(
                    r"\{[^{}]*tool_call[^{}]*\{.*?\}[^{}]*\}", response_text, _re.DOTALL
                )
                if not json_match:
                    json_match = _re.search(r"(\{.*\})", response_text, _re.DOTALL)
                if json_match:
                    try:
                        parsed = _json.loads(json_match.group())
                        if "tool_call" in parsed:
                            tc = parsed["tool_call"]
                            # Store only the JSON part in history (not the reasoning)
                            self.messages.append(
                                {"role": "assistant", "content": _json.dumps(parsed)}
                            )
                            pyautogui_code = self._tool_call_to_pyautogui(tc)
                            return response_text, [pyautogui_code]
                    except _json.JSONDecodeError:
                        pass

                self.messages.append({"role": "assistant", "content": response_text})
                return response_text, ["DONE"]

        return Holo3Agent()

    if kind == "uitars":
        from mm_agents.uitars15_v2 import UITarsAgent
        from openai import OpenAI

        api_urls_str = os.environ.get("OS_WORLD_API_URLS", "")
        if api_urls_str:
            api_urls = [u.strip() for u in api_urls_str.split(",") if u.strip()]
        else:
            api_urls = [
                os.environ.get("OS_WORLD_API_URL")
                or os.environ.get("OPENAI_API_BASE")
                or os.environ.get("OPENAI_BASE_URL")
                or "http://host.docker.internal:14243/v1"
            ]
        api_key = (
            os.environ.get("OS_WORLD_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "vllm-local"
        )
        model = os.environ.get("OS_WORLD_MODEL", "ui-tars-1.5-7b")

        agent = UITarsAgent(
            model=model,
            model_type="qwen25vl",
            max_tokens=int(os.environ.get("OS_WORLD_MAX_TOKENS", "4096")),
            top_p=float(os.environ.get("OS_WORLD_TOP_P", "0.9")),
            temperature=float(os.environ.get("OS_WORLD_TEMPERATURE", "0.0")),
            max_trajectory_length=int(os.environ.get("OS_WORLD_MAX_TRAJ", "50")),
            max_image_history_length=int(os.environ.get("OS_WORLD_HISTORY_N", "5")),
            max_steps=int(os.environ.get("OS_WORLD_MAX_STEPS", "100")),
            use_thinking=False,
            language="English",
        )
        _selected_url = api_urls[os.getpid() % len(api_urls)]
        _uitars_client = OpenAI(base_url=_selected_url, api_key=api_key, timeout=120)

        def _uitars_inference(messages):
            resp = _uitars_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=4096,
                temperature=float(os.environ.get("OS_WORLD_TEMPERATURE", "0.0")),
                top_p=float(os.environ.get("OS_WORLD_TOP_P", "0.9")),
            )
            return resp.choices[0].message.content

        agent.inference_func = _uitars_inference
        return agent

    if kind == "mobileagent_v3":
        from mm_agents.mobileagent_v3.mobile_agent import MobileAgentV3

        api_url = (
            os.environ.get("OS_WORLD_API_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("OPENAI_BASE_URL")
            or "http://host.docker.internal:14243/v1"
        )
        api_key = (
            os.environ.get("OS_WORLD_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "vllm-local"
        )
        model = os.environ.get("OS_WORLD_MODEL", "gui-owl-1.5-32b-instruct")
        engine_params = {
            "engine_type": "openai",
            "api_key": api_key,
            "base_url": api_url,
            "model": model,
        }
        return MobileAgentV3(
            manager_engine_params=engine_params,
            operator_engine_params=engine_params,
            reflector_engine_params=engine_params,
            grounding_enging_params=engine_params,
            wait_after_action_seconds=float(
                os.environ.get("OS_WORLD_SLEEP_AFTER", "5.0")
            ),
        )
    raise ValueError(
        f"unknown OS_WORLD_AGENT_KIND={kind!r}; use 'kimi', 'owl15', 'uitars', 'holo3', or 'mobileagent_v3'"
    )


# === Per-task run ==========================================================


def _update_preds(
    output_dir: Path,
    task_id: str,
    reward: float,
    n_steps: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "preds.json"
    with _PREDS_LOCK:
        data = {}
        if preds_path.exists():
            try:
                data = json.loads(preds_path.read_text())
            except json.JSONDecodeError:
                data = {}
        data[task_id] = {
            "task_id": task_id,
            "reward": reward,
            "n_steps": n_steps,
        }
        preds_path.write_text(json.dumps(data, indent=2))


def _obs_to_text(obs: Any, observation_type: str) -> str:
    if isinstance(obs, dict):
        for k in ("accessibility_tree", "a11y_tree", "text"):
            v = obs.get(k)
            if isinstance(v, (str, bytes)) and v:
                return v if isinstance(v, str) else v.decode("utf-8", errors="replace")
    return ""


def _obs_to_screenshot_b64(obs: Any) -> Optional[str]:
    """Pull the screenshot bytes out of an OSWorld observation and return a
    base64 string (no data URL prefix)."""
    import base64

    if not isinstance(obs, dict):
        return None
    raw = obs.get("screenshot")
    if isinstance(raw, bytes):
        return base64.b64encode(raw).decode("ascii")
    if isinstance(raw, str):
        # Already a string — assume it's already base64 or a data URL.
        if raw.startswith("data:"):
            return raw.split(",", 1)[1]
        return raw
    return None


def run_one(
    task_id: str,
    task_config: Dict[str, Any],
    output_dir: str,
    max_steps: int,
    observation_type: str,
    fsm_context: Optional[RequestContext],
) -> RunResult:
    out_root = Path(output_dir)
    instance_dir = out_root / task_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    _session = SessionLogger(out_root, task_id)

    events: List[dict] = []
    actions: List[str] = []
    reward = 0.0
    error: Optional[str] = None

    env = None
    try:
        provider_name = os.environ.get("OS_WORLD_PROVIDER", "docker")
        enable_proxy = os.environ.get("OS_WORLD_ENABLE_PROXY", "0") not in (
            "0",
            "false",
            "",
        )

        if provider_name == "remote":
            from agsbench_gui_agent.remote_env import RemoteDesktopEnv

            env_manager_url = os.environ.get(
                "OS_WORLD_ENV_MANAGER_URL", "http://host.docker.internal:1905"
            )
            env = RemoteDesktopEnv(
                env_manager_url=env_manager_url,
                provider_name="docker",
                os_type="Ubuntu",
                action_space="pyautogui",
                require_a11y_tree=False,
                headless=True,
                screen_size=(1920, 1080),
                enable_proxy=enable_proxy,
                client_password="",
            )
        else:
            from desktop_env.desktop_env import DesktopEnv

            env = DesktopEnv(
                provider_name=provider_name,
                os_type="Ubuntu",
                action_space="pyautogui",
                require_a11y_tree=False,
                headless=True,
                screen_size=(1920, 1080),
                enable_proxy=enable_proxy,
                client_password="",
            )
        agent = _make_agent(env)

        # Match upstream's `run_single_example_kimi` / `run_single_example`
        # lifecycle: reset agent, reset env (loads task_config), wait 60s
        # for desktop to settle, then get the initial observation via
        # env._get_obs(). Reset signature differs:
        #   - KimiAgent.reset(_logger=...)            (kwarg-only)
        #   - Owl15Agent.reset(logger, vm_ip=...)     (positional + kwarg)
        # KimiAgent.reset also references `logging.getLogger(...)` but
        # never imports `logging` (uses loguru elsewhere) — calling
        # reset(None) raises NameError on Kimi.
        runtime_logger = logging.getLogger("agsbench_gui_agent.runner")
        kind = os.environ.get("OS_WORLD_AGENT_KIND", "owl15").lower()
        if kind == "kimi":
            agent.reset(_logger=runtime_logger)
        elif kind == "mobileagent_v3":
            pass  # MobileAgent-v3 resets inside the loop branch
        elif kind == "holo3":
            agent.reset()
        elif kind == "uitars":
            agent.reset()
        else:
            try:
                agent.reset(runtime_logger, vm_ip=env.vm_ip)
            except TypeError:
                agent.reset(runtime_logger)
        env.reset(task_config=task_config)
        time.sleep(60)
        obs = env._get_obs()
        instruction = task_config.get("instruction", "")

        # === MobileAgent-v3 loop (agent drives env internally) ===
        if kind == "mobileagent_v3":
            from types import SimpleNamespace

            ma_args = SimpleNamespace(
                grounding_stage=int(os.environ.get("OS_WORLD_GROUNDING_STAGE", "0")),
                grounding_info_level=int(
                    os.environ.get("OS_WORLD_GROUNDING_INFO_LEVEL", "1")
                ),
                enable_rag=int(os.environ.get("OS_WORLD_ENABLE_RAG", "1")),
                rag_path=os.environ.get(
                    "OS_WORLD_RAG_PATH",
                    "/workspace/os_world/.cache/OSWorld_upstream/mm_agents/mobileagent_v3/Perplexica_rag_knowledge_verified.json",
                ),
                guide_path=os.environ.get(
                    "OS_WORLD_GUIDE_PATH",
                    "/workspace/os_world/.cache/OSWorld_upstream/mm_agents/mobileagent_v3/experience.json",
                ),
                max_trajectory_length=max_steps,
            )
            agent.reset()
            for step_idx in range(max_steps):
                try:
                    global_state, action_code, step_status, env_reward, env_done = (
                        agent.step(instruction, env, ma_args)
                    )
                except Exception as step_exc:
                    events.append({"step": step_idx, "error": str(step_exc)})
                    break
                if action_code:
                    actions.append(action_code)
                events.append(
                    {
                        "step": step_idx,
                        "action": action_code or "",
                        "done": bool(env_done),
                    }
                )
                if not step_status:
                    break
                if env_done:
                    break
            time.sleep(30)
            try:
                result_val = env.evaluate()
                reward = float(result_val) if result_val else 0.0
            except Exception as eval_exc:
                reward = 0.0
                error = f"evaluate failed: {eval_exc}"
            _update_preds(Path(output_dir), task_id, reward, len(actions))
            traj_data = {
                "task_id": task_id,
                "events": events,
                "info": {
                    "reward": reward,
                    "n_steps": len(actions),
                    "error": error,
                },
            }
            traj_path = instance_dir / f"{task_id}.traj.json"
            traj_path.write_text(json.dumps(traj_data, indent=2))
            return RunResult(
                success=reward > 0,
                task_id=task_id,
                reward=reward,
                n_steps=len(actions),
                actions=actions,
                error=error,
            )

        # Optional screenshot dumping for click-grounding diagnostics.
        # Set OS_WORLD_DUMP_SCREENSHOTS=1 to save the pre-action screenshot
        # PLUS a copy of the screenshot with the emitted click coord drawn
        # as a red dot for each step. Files go to instance_dir/screenshots/.
        dump_screens = os.environ.get("OS_WORLD_DUMP_SCREENSHOTS", "0") not in (
            "0",
            "false",
            "",
        )
        screens_dir = instance_dir / "screenshots"
        if dump_screens:
            screens_dir.mkdir(parents=True, exist_ok=True)

        done = False
        for step_idx in range(max_steps):
            # Snapshot the input screenshot BEFORE the agent decides, so we
            # see what the model actually conditioned on.
            if dump_screens and isinstance(obs, dict):
                shot = obs.get("screenshot")
                if isinstance(shot, (bytes, bytearray)):
                    (screens_dir / f"step{step_idx:03d}_in.png").write_bytes(
                        bytes(shot)
                    )

            _t0_llm = time.time()
            pick = llm_pick_action_call(
                agent=agent,
                instruction=instruction,
                obs=obs,
                task_id=task_id,
                fsm_context=fsm_context,
            )
            _dur_llm = time.time() - _t0_llm
            try:
                _raw = getattr(agent, "_last_response", None) or {}
                _u = _raw.get("usage") if isinstance(_raw, dict) else None
                _usage = None
                if _u:
                    _cdet = _u.get("completion_tokens_details") or {}
                    _pdet = _u.get("prompt_tokens_details") or {}
                    _usage = {
                        "prompt_tokens": _u.get("prompt_tokens", 0),
                        "completion_tokens": _u.get("completion_tokens", 0),
                        "total_tokens": _u.get("total_tokens", 0),
                        "reasoning_tokens": _cdet.get("reasoning_tokens", 0),
                        "cached_tokens": _pdet.get("cached_tokens", 0),
                    }
                _session.log_llm_call(
                    model=getattr(agent, "model", "unknown"),
                    messages=getattr(agent, "messages", None),
                    response_text=pick.get("response", ""),
                    actions=pick.get("actions", []),
                    usage=_usage,
                    duration_s=_dur_llm,
                    step=step_idx,
                )
            except Exception as _log_err:
                _USAGE_LOG.warning("session log_llm_call error: %s", _log_err)

            # If we have a screenshot AND the action contains pyautogui.click,
            # render a marked copy with the click point drawn so we can
            # visually verify the coord-mapping outcome.
            if dump_screens and isinstance(obs, dict):
                shot = obs.get("screenshot")
                step_actions_pre = pick.get("actions") or []
                if (
                    isinstance(shot, (bytes, bytearray))
                    and step_actions_pre
                    and isinstance(step_actions_pre[0], str)
                ):
                    try:
                        import io
                        import re

                        from PIL import Image, ImageDraw

                        m = re.search(
                            r"pyautogui\.(?:click|doubleClick|tripleClick|moveTo)\((-?\d+)\s*,\s*(-?\d+)",
                            step_actions_pre[0],
                        )
                        if m:
                            cx, cy = int(m.group(1)), int(m.group(2))
                            im = Image.open(io.BytesIO(bytes(shot))).convert("RGB")
                            d = ImageDraw.Draw(im)
                            d.ellipse(
                                [cx - 12, cy - 12, cx + 12, cy + 12],
                                outline="red",
                                width=4,
                            )
                            d.line(
                                [cx - 20, cy, cx + 20, cy],
                                fill="red",
                                width=2,
                            )
                            d.line(
                                [cx, cy - 20, cx, cy + 20],
                                fill="red",
                                width=2,
                            )
                            im.save(screens_dir / f"step{step_idx:03d}_marked.png")
                    except Exception as _e:  # noqa: BLE001
                        pass
            # Upstream KimiAgent.predict() returns (response, actions,
            # info_dict). Empty / error actions → end the episode.
            step_actions = pick.get("actions") or []
            if (
                not step_actions
                or step_actions[0] == ""
                or step_actions[0].lower().startswith("error")
            ):
                events.append(
                    {
                        "step": step_idx,
                        "action": "",
                        "reward": 0.0,
                        "done": False,
                        "note": "empty_or_error_actions",
                    }
                )
                break

            for action in step_actions:
                actions.append(action)
                step_result = env_step_call(
                    env, action, task_id=task_id, fsm_context=fsm_context
                )
                reward = step_result["reward"]
                done = bool(step_result["done"])
                events.append(
                    {
                        "step": step_idx,
                        "action": action,
                        "reward": reward,
                        "done": done,
                    }
                )
                try:
                    _session.log_env_step(
                        step=step_idx, action=action, reward=reward, done=done
                    )
                except Exception as _log_err:
                    _USAGE_LOG.warning("session log_env_step error: %s", _log_err)
                obs = getattr(env, "last_obs", obs)
                if done or action in ("DONE", "FAIL"):
                    break
            if done or any(a in ("DONE", "FAIL") for a in step_actions):
                break
        # Per upstream: wait for environment to settle before grading.
        time.sleep(30)

        # Final evaluation — DesktopEnv exposes evaluate() that runs the
        # task's built-in grader (file/registry/cmd inspections) and returns
        # the 0/1 reward.
        try:
            reward = float(env.evaluate())
        except Exception:
            pass
    except Exception as e:
        error = str(e)
        events.append({"error": error, "traceback": traceback.format_exc()})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            _session.log_session_end(reward=reward, n_steps=len(actions), error=error)
        except Exception as _log_err:
            _USAGE_LOG.warning("session log_session_end error: %s", _log_err)
        (instance_dir / f"{task_id}.traj.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "events": events,
                    "info": {"reward": reward, "n_steps": len(actions), "error": error},
                },
                indent=2,
            )
        )
        _update_preds(out_root, task_id, reward, len(actions))

    return RunResult(
        success=(reward > 0) and (error is None),
        task_id=task_id,
        reward=reward,
        n_steps=len(actions),
        actions=actions,
        error=error,
    )
