"""Single-task runner for browsergym.

Agent harness follows the upstream DemoAgent implementation from
BrowserGym (demo_agent/agent.py), but split into two instrumented
functions (llm_pick_action_call, env_step_call) for fast-service
per-step timing/telemetry.

Key behaviors preserved from DemoAgent:
- HighLevelActionSet for action space description and parsing
- flatten_axtree_to_str for observation preprocessing
- Full action history in prompt context
- Raw LLM output passed to env.step (browsergym handles parsing)
- Goal from goal_object, open tabs, last_action_error in context

NB: do NOT add ``from __future__ import annotations`` here.
"""

import json
import os
import time as _time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional

from fast_service import RequestContext

from agsbench_webagent.fsm import agsbench_webagent_fsm
from agsbench_webagent.session_logger import SessionLogger


def _image_to_jpg_base64_url(image) -> str:
    """Convert a numpy array or PIL Image to a base64-encoded JPEG data URL."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode in ("RGBA", "LA"):
        image = image.convert("RGB")

    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()

    return f"data:image/jpeg;base64,{image_base64}"


_PREDS_LOCK = Lock()


@dataclass
class RunResult:
    success: bool
    instance_id: str
    task_id: str
    reward: float = 0.0
    n_steps: int = 0
    terminated: bool = False
    truncated: bool = False
    actions: List[str] = field(default_factory=list)
    error: Optional[str] = None


# === Timing helpers (monitor_only) ==========================================


@agsbench_webagent_fsm.fast_service(monitor_only=True)
def env_step_call(
    env: Any,
    action: str,
    fsm_context: RequestContext = None,
) -> dict:
    """Timed wrapper around ``BrowserEnv.step``. One row per agent step."""
    obs, reward, terminated, truncated, info = env.step(action)
    return {
        "obs": obs,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": info,
    }


@agsbench_webagent_fsm.fast_service(monitor_only=True)
def llm_pick_action_call(
    messages: List[dict],
    model: str,
    fsm_context: RequestContext = None,
) -> dict:
    """Timed wrapper around the LLM call that picks the next browsergym action."""
    return _call_llm(messages=messages, model=model)


# === LLM call ==============================================================


def _call_llm(messages: List[dict], model: str) -> dict:
    """Call the OpenAI-compatible API. Returns the raw action text plus
    full response metadata for session logging.

    Known limitation: prompt size is not capped. For pages with large
    accessibility trees (e.g. Wikipedia), the accumulated prompt
    (axtree + full action history) can exceed the model's context window
    after just a few steps, causing a 400 error that terminates the task.
    This is left unfixed intentionally — the correct long-term solution is
    axtree truncation or history windowing, which requires evaluating the
    trade-off between context richness and reliability.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
        base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
    )

    t0 = _time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=4096,
    )
    duration_s = _time.perf_counter() - t0

    import re as _re

    msg = resp.choices[0].message

    # 1. Prefer structured tool_calls (kimi_k2 parser with --enable-auto-tool-choice).
    tool_calls = getattr(msg, "tool_calls", None) or []
    if tool_calls:
        tc = tool_calls[0]
        fn = tc.function
        args_str = fn.arguments or ""
        try:
            import json as _json
            args = _json.loads(args_str)
            # Reconstruct as BrowserGym action string: func_name(arg0) or func_name(key=val)
            positional = list(args.values())
            action = f"{fn.name}({repr(positional[0]) if len(positional)==1 else ', '.join(repr(v) for v in positional)})"
        except Exception:
            action = f"{fn.name}({args_str})"
    else:
        raw_response = msg.content or ""

        # 2. Triple-backtick code blocks (primary text format).
        # Strip optional language specifier (e.g. ```python\n) but only when followed by newline.
        action_match = _re.search(r"```(?:[a-zA-Z0-9]+\n)?(.*?)\s*```", raw_response, _re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
        else:
            # 3. Kimi-K2 native token format embedded in content.
            # Observed sub-formats:
            #   a) functions.NAME:N <|tool_call_argument_begin|> {JSON} [<|tool_call_end|>]
            #   b) functions.NAME:N <|tool_call_argument_begin|> [empty] [<|tool_calls_section_end|>]
            #   c) functions.NAME:N  (no args section; :N is call-index, used as arg for 1-arg fns)
            #   d) functions.NAME(ARGS)  (old paren style)
            fn_match = _re.search(
                r"<\|tool_call_begin\|>\s*functions\.(\w+)(?::(\d+))?",
                raw_response
            )
            if fn_match:
                import json as _json
                fn_name = fn_match.group(1)
                call_idx = fn_match.group(2)  # sequence/index suffix, may be None
                rest = raw_response[fn_match.end():]

                arg_begin_tag = "<|tool_call_argument_begin|>"
                arg_begin_pos = rest.find(arg_begin_tag)

                if arg_begin_pos >= 0:
                    raw_args = rest[arg_begin_pos + len(arg_begin_tag):]
                    # Trim at any closing token or end-of-string
                    for term in ("<|tool_call_end|>", "<|tool_calls_section_end|>", "<|tool_call_begin|>"):
                        idx = raw_args.find(term)
                        if idx >= 0:
                            raw_args = raw_args[:idx]
                    raw_args = raw_args.strip()

                    if raw_args.startswith("{"):
                        try:
                            args = _json.loads(raw_args)
                            positional = list(args.values())
                            action = f"{fn_name}({', '.join(repr(v) for v in positional)})"
                        except Exception:
                            action = f"{fn_name}({raw_args})"
                    elif raw_args:
                        action = f"{fn_name}({raw_args})"
                    else:
                        # Empty args: use call_idx as fallback argument
                        action = f"{fn_name}({call_idx})" if call_idx else f"{fn_name}()"
                else:
                    # No <|tool_call_argument_begin|>: check for paren style
                    paren_match = _re.match(r"\s*[:(]([^<]*?)(?:\)|$)", rest, _re.DOTALL)
                    if paren_match and "(" in rest[:10]:
                        fn_args = paren_match.group(1).strip().rstrip(")")
                        action = f"{fn_name}({fn_args})"
                    else:
                        # No args at all: use call_idx as fallback
                        action = f"{fn_name}({call_idx})" if call_idx else f"{fn_name}()"
            else:
                # 4. Fallback: use entire content (may fail in BrowserGym parser).
                action = raw_response.strip()

    return {
        "action": action,
        "model": model,
        "messages": messages,
        "completion_response": resp,
        "duration_s": duration_s,
    }


# === Observation preprocessing (mirrors DemoAgent.obs_preprocessor) =========


def _preprocess_obs(obs: dict) -> dict:
    """Extract the fields needed for prompt construction from a raw
    browsergym observation dict."""
    from browsergym.utils.obs import (
        flatten_axtree_to_str,
        flatten_dom_to_str,
        prune_html,
    )

    result = {
        "chat_messages": obs.get("chat_messages", []),
        "goal_object": obs.get("goal_object", []),
        "last_action": obs.get("last_action", ""),
        "last_action_error": obs.get("last_action_error", ""),
        "open_pages_urls": obs.get("open_pages_urls", []),
        "open_pages_titles": obs.get("open_pages_titles", []),
        "active_page_index": obs.get("active_page_index", 0),
    }

    # Flatten accessibility tree.
    axtree_object = obs.get("axtree_object")
    if axtree_object is not None:
        result["axtree_txt"] = flatten_axtree_to_str(axtree_object)
    else:
        result["axtree_txt"] = obs.get("axtree_txt", "")

    # Pruned HTML (optional, used if dom_object is present).
    dom_object = obs.get("dom_object")
    if dom_object is not None:
        result["pruned_html"] = prune_html(flatten_dom_to_str(dom_object))
    else:
        result["pruned_html"] = obs.get("pruned_html", "")

    # Screenshot (numpy array or PIL Image).
    result["screenshot"] = obs.get("screenshot")

    return result


# === Prompt construction (mirrors DemoAgent.get_action) =====================


def _build_messages(
    obs: dict,
    action_set_description: str,
    action_history: List[str],
    use_axtree: bool = True,
    use_html: bool = False,
    use_screenshot: bool = False,
) -> List[dict]:
    """Build the OpenAI-style messages list, matching DemoAgent's prompt
    structure exactly."""
    system_msgs = []
    user_msgs = []

    # System instruction — non-chat mode (goal-based tasks).
    system_msgs.append(
        {
            "type": "text",
            "text": (
                "# Instructions\n\n"
                "Review the current state of the page and all other information to find the best\n"
                "possible next action to accomplish your goal. Your answer will be interpreted\n"
                "and executed by a program, make sure to follow the formatting instructions.\n\n"
                "IMPORTANT: You are operating inside a local WebArena environment. You MUST only\n"
                "navigate to URLs provided in the task or already open in the browser tabs. Do NOT\n"
                "navigate to external websites such as google.com, google.com/maps, osrm.org,\n"
                "nominatim.openstreetmap.org, or any site outside the local environment.\n"
            ),
        }
    )

    # Goal.
    user_msgs.append({"type": "text", "text": "# Goal\n"})
    goal_object = obs.get("goal_object", [])
    if goal_object:
        user_msgs.extend(goal_object)
    else:
        # Fallback for envs that don't provide goal_object as openai-style messages.
        user_msgs.append({"type": "text", "text": "(no goal provided)\n"})

    # Open tabs.
    user_msgs.append({"type": "text", "text": "# Currently open tabs\n"})
    for page_index, (page_url, page_title) in enumerate(
        zip(obs.get("open_pages_urls", []), obs.get("open_pages_titles", []))
    ):
        active = (
            " (active tab)" if page_index == obs.get("active_page_index", 0) else ""
        )
        user_msgs.append(
            {
                "type": "text",
                "text": f"Tab {page_index}{active}\n  Title: {page_title}\n  URL: {page_url}\n",
            }
        )

    # Accessibility tree.
    if use_axtree:
        user_msgs.append(
            {
                "type": "text",
                "text": f"# Current page Accessibility Tree\n\n{obs.get('axtree_txt', '')}\n\n",
            }
        )

    # Pruned HTML.
    if use_html and obs.get("pruned_html"):
        user_msgs.append(
            {
                "type": "text",
                "text": f"# Current page DOM\n\n{obs.get('pruned_html', '')}\n\n",
            }
        )

    # Screenshot.
    if use_screenshot and obs.get("screenshot") is not None:
        user_msgs.append({"type": "text", "text": "# Current page Screenshot\n"})
        user_msgs.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": _image_to_jpg_base64_url(obs["screenshot"]),
                    "detail": "auto",
                },
            }
        )

    # Action space description.
    user_msgs.append(
        {
            "type": "text",
            "text": (
                f"# Action Space\n\n{action_set_description}\n\n"
                "Here are examples of actions with chain-of-thought reasoning:\n\n"
                "I now need to click on the Submit button to send the form. "
                "I will use the click action on the button, which has bid 12.\n"
                '```click("12")```\n\n'
                "I found the information requested by the user, I will send it to the chat.\n"
                '```send_msg_to_user("The price for a 15\\" laptop is 1499 USD.")```\n\n'
            ),
        }
    )

    # Past actions.
    if action_history:
        user_msgs.append({"type": "text", "text": "# History of past actions\n"})
        for action in action_history:
            user_msgs.append({"type": "text", "text": f"\n{action}\n"})

        if obs.get("last_action_error"):
            user_msgs.append(
                {
                    "type": "text",
                    "text": (
                        "# Error message from last action\n\n"
                        f"{obs['last_action_error']}\n\n"
                    ),
                }
            )

    # Next action request.
    user_msgs.append(
        {
            "type": "text",
            "text": (
                "# Next action\n\n"
                "You will now think step by step and produce your next best action. "
                "Reflect on your past actions, any resulting error message, and the current "
                "state of the page before deciding on your next action.\n"
            ),
        }
    )

    # Assemble into openai messages format.
    system_content = [m for m in system_msgs]
    user_content = [m for m in user_msgs]

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


# === Per-task run ==========================================================


def _update_preds(
    output_dir: Path,
    instance_id: str,
    task_id: str,
    reward: float,
    actions: List[str],
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
        data[instance_id] = {
            "instance_id": instance_id,
            "task_id": task_id,
            "reward": reward,
            "n_actions": len(actions),
        }
        preds_path.write_text(json.dumps(data, indent=2))


def _save_trajectory(
    instance_dir: Path,
    instance_id: str,
    events: List[dict],
    info: dict,
):
    instance_dir.mkdir(parents=True, exist_ok=True)
    (instance_dir / f"{instance_id}.traj.json").write_text(
        json.dumps(
            {"events": events, "info": info, "instance_id": instance_id}, indent=2
        )
    )


def run_one(
    task_id: str,
    instance_id: str,
    seed: int,
    max_steps: int,
    output_dir: str,
    fsm_context: Optional[RequestContext],
) -> RunResult:
    out_root = Path(output_dir)
    instance_dir = out_root / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)

    events: List[dict] = []
    actions: List[str] = []
    reward = 0.0
    terminated = False
    truncated = False
    error: Optional[str] = None

    session = SessionLogger(
        output_dir=str(out_root), instance_id=instance_id, task_id=task_id
    )

    env = None
    try:
        import gymnasium as gym
        from browsergym.core.action.highlevel import HighLevelActionSet

        # Register the relevant browsergym suites.
        if task_id.startswith("browsergym/miniwob"):
            import browsergym.miniwob  # noqa: F401
        elif task_id.startswith("browsergym/webarena_verified"):
            import browsergym.webarena_verified  # noqa: F401
        elif task_id.startswith("browsergym/webarena"):
            import browsergym.webarena  # noqa: F401
        elif task_id.startswith("browsergym/"):
            import browsergym.core  # noqa: F401
        else:
            raise ValueError(f"task_id {task_id!r} must start with 'browsergym/'.")

        # Action set matching DemoAgent defaults.
        action_set = HighLevelActionSet(
            subsets=["chat", "tab", "nav", "bid", "infeas"],
            strict=False,
            multiaction=False,
            demo_mode="off",
        )
        action_set_description = action_set.describe(
            with_long_description=False, with_examples=True
        )

        model = os.environ.get("WEBAGENT_MODEL", "deepseek-chat")
        use_axtree = os.environ.get("WEBAGENT_USE_AXTREE", "1") == "1"
        use_html = os.environ.get("WEBAGENT_USE_HTML", "0") == "1"
        use_screenshot = os.environ.get("WEBAGENT_USE_SCREENSHOT", "0") == "1"

        chromium_path = os.environ.get("WEBAGENT_CHROMIUM_PATH", "")
        pw_kwargs = {}
        if chromium_path:
            pw_kwargs["executable_path"] = chromium_path

        env = gym.make(
            task_id,
            headless=True,
            action_mapping=action_set.to_python_code,
            timeout=int(os.environ.get("WEBAGENT_TIMEOUT", "10000")),
            pw_chromium_kwargs=pw_kwargs,
        )
        obs, info = env.reset(seed=seed)

        for step_idx in range(max_steps):
            # Preprocess observation (flatten axtree, prune html).
            processed_obs = _preprocess_obs(obs)

            # Build messages matching DemoAgent prompt structure.
            messages = _build_messages(
                obs=processed_obs,
                action_set_description=action_set_description,
                action_history=actions,
                use_axtree=use_axtree,
                use_html=use_html,
                use_screenshot=use_screenshot,
            )

            # LLM call (instrumented).
            pick = llm_pick_action_call(
                messages=messages,
                model=model,
                fsm_context=fsm_context,
            )
            action = pick["action"]
            actions.append(action)

            # Log LLM call with full I/O.
            session.log_llm_call(
                step=step_idx,
                model=pick.get("model", ""),
                messages=pick.get("messages", []),
                completion_response=pick.get("completion_response"),
                action_parsed=action,
                duration_s=pick.get("duration_s"),
            )

            # Env step (instrumented). Pass raw LLM output — browsergym
            # handles action parsing via the action_mapping we set.
            step_result = env_step_call(env, action, fsm_context=fsm_context)
            obs = step_result["obs"]
            reward = step_result["reward"]
            terminated = step_result["terminated"]
            truncated = step_result["truncated"]
            events.append(
                {
                    "step": step_idx,
                    "action": action,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                }
            )

            # Log env step.
            session.log_env_step(
                step=step_idx,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                obs_text=processed_obs.get("axtree_txt", ""),
            )

            if terminated or truncated:
                break
    except Exception as e:
        error = str(e)
        events.append({"error": error, "traceback": traceback.format_exc()})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        info_blob = {
            "task_id": task_id,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "n_steps": len(actions),
            "error": error,
        }
        _save_trajectory(instance_dir, instance_id, events, info_blob)
        _update_preds(out_root, instance_id, task_id, reward, actions)
        session.log_session_end(
            reward=reward,
            n_steps=len(actions),
            success=(reward > 0) and (error is None),
            error=error,
        )

    return RunResult(
        success=(reward > 0) and (error is None),
        instance_id=instance_id,
        task_id=task_id,
        reward=reward,
        n_steps=len(actions),
        terminated=terminated,
        truncated=truncated,
        actions=actions,
        error=error,
    )
