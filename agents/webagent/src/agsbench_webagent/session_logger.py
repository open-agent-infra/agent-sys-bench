"""Per-task session logger for the webagent benchmark.

Writes a JSONL file per task capturing every LLM call (with the full
chat.completions response object) and every env step. No truncation
is applied to logged content.

Architecture: the SessionLogger sends records to a background writer
thread via a queue. The writer drains the queue and flushes to disk,
so the hot path (agent loop) never blocks on file I/O.

Output: <output_dir>/<instance_id>/<instance_id>.session.jsonl
"""

import atexit
import datetime as _dt
import json
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


_SENTINEL = object()


class _BackgroundWriter:
    """Single background thread that drains a queue and writes to a file."""

    def __init__(self, path: Path):
        self._path = path
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name=f"session-writer-{path.stem}", daemon=True
        )
        self._thread.start()

    def _run(self):
        with self._path.open("a", encoding="utf-8") as fh:
            while True:
                record = self._queue.get()
                if record is _SENTINEL:
                    break
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                fh.flush()

    def write(self, record: dict):
        self._queue.put(record)

    def close(self):
        """Signal the writer to finish and wait for it to drain."""
        self._queue.put(_SENTINEL)
        self._thread.join(timeout=10)


class SessionLogger:
    """Append-only JSONL logger for one browsergym task session.

    All writes go through a background thread so the agent loop is
    never blocked by disk I/O.
    """

    def __init__(self, output_dir: str, instance_id: str, task_id: str):
        session_dir = Path(output_dir) / instance_id
        session_dir.mkdir(parents=True, exist_ok=True)
        self._writer = _BackgroundWriter(session_dir / f"{instance_id}.session.jsonl")
        atexit.register(self._writer.close)
        self._write(
            {
                "event": "session_start",
                "instance_id": instance_id,
                "task_id": task_id,
                "timestamp": self._now(),
            }
        )

    @staticmethod
    def _now() -> str:
        return _dt.datetime.utcnow().isoformat(timespec="milliseconds") + "Z"

    def _write(self, record: dict):
        self._writer.write(record)

    def log_llm_call(
        self,
        *,
        step: int,
        model: str,
        messages: List[Dict[str, Any]],
        completion_response: Any,
        action_parsed: str,
        duration_s: Optional[float] = None,
    ):
        """Record one LLM call with the full chat.completions response.

        ``completion_response`` is the raw response object from the OpenAI
        client (or its dict/json serialization). No truncation is applied.
        """
        # Serialize the response object if it has a model_dump method (pydantic).
        if hasattr(completion_response, "model_dump"):
            response_data = completion_response.model_dump()
        elif hasattr(completion_response, "to_dict"):
            response_data = completion_response.to_dict()
        else:
            response_data = completion_response

        self._write(
            {
                "event": "llm_call",
                "timestamp": self._now(),
                "step": step,
                "model": model,
                "messages": messages,
                "completion_response": response_data,
                "action_parsed": action_parsed,
                "duration_s": (
                    round(duration_s, 4) if duration_s is not None else None
                ),
            }
        )

    def log_env_step(
        self,
        *,
        step: int,
        action: str,
        reward: float,
        terminated: bool,
        truncated: bool,
        obs_text: str = "",
    ):
        """Record one env.step execution. Full observation text is stored."""
        self._write(
            {
                "event": "env_step",
                "timestamp": self._now(),
                "step": step,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "obs_text": obs_text,
            }
        )

    def log_session_end(
        self,
        *,
        reward: float,
        n_steps: int,
        success: bool,
        error: Optional[str] = None,
    ):
        """Record session completion and flush the writer."""
        self._write(
            {
                "event": "session_end",
                "timestamp": self._now(),
                "reward": reward,
                "n_steps": n_steps,
                "success": success,
                "error": error,
            }
        )
        # Ensure all records are flushed before the session object is dropped.
        self._writer.close()
