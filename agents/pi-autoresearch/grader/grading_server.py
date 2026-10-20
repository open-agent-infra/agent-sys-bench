"""MLE-bench grading sidecar for the auto-research scenario.

Reads the prepared competitions under MLEBENCH_DATA_DIR (including the
private answers, which the agent container never mounts).

  GET  /health
  POST /validate?competition_id=<id>   -F file=@submission.csv
       -> {"valid": bool, "result": str}           (no score; agent-facing)
  POST /grade?competition_id=<id>      -F file=@submission.csv
       -> MLE-bench CompetitionReport dict          (runner-only)

/grade requires the X-Grader-Token header when GRADER_TOKEN is set. The
runner keeps that token out of the agent's environment, so the agent can
check its submission format but cannot read its test score.
"""

import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

from mlebench.grade import grade_csv, validate_submission
from mlebench.registry import registry

DATA_DIR = Path(os.getenv("MLEBENCH_DATA_DIR", "/data"))
TOKEN = os.getenv("GRADER_TOKEN", "")
HOST = os.getenv("GRADER_HOST", "127.0.0.1")
PORT = int(os.getenv("GRADER_PORT", "5005"))

app = Flask(__name__)
_registry = registry.set_data_dir(DATA_DIR)


def _with_submission(fn):
    competition_id = request.args.get("competition_id", "")
    if not competition_id or "file" not in request.files:
        return jsonify({"error": "need ?competition_id= and a 'file' upload"}), 400
    try:
        competition = _registry.get_competition(competition_id)
    except Exception as e:
        return jsonify({"error": f"unknown competition {competition_id}: {e}"}), 404
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "submission.csv"
        request.files["file"].save(path)
        try:
            return fn(path, competition)
        except Exception as e:
            return jsonify({"error": "grader failed", "details": str(e)}), 500


@app.route("/validate", methods=["POST"])
def validate():
    def _run(path, competition):
        valid, message = validate_submission(path, competition)
        return jsonify({"valid": bool(valid), "result": message})
    return _with_submission(_run)


@app.route("/grade", methods=["POST"])
def grade():
    if TOKEN and request.headers.get("X-Grader-Token") != TOKEN:
        return jsonify({"error": "forbidden"}), 403

    def _run(path, competition):
        report = grade_csv(path, competition).to_dict()
        report.pop("submission_path", None)  # temp path, meaningless to callers
        return jsonify(report)
    return _with_submission(_run)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, threaded=True)
