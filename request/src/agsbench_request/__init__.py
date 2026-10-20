"""AgentSysBench request package — workload generators and data preparation.

This package contains scenario-specific code (manifest parsing, request
construction, success evaluation) that is ONLY used by the loadgen, never by
agents. Wire protocol types (Request/Result/RPC_NAME) live in
sdk/scenario-protocol/ and are shared by both loadgen and agents.

Two unified entry points:
  python -m request.loadgen --scenario <name> --endpoint <url> ...
  python -m request.prepare_data --scenario <name> ...
"""

import importlib.util
from pathlib import Path
from typing import Any, Callable, Optional

REQUEST_DIR = Path(__file__).parent


def load_request_module(scenario_name: str, module_name: str = "loadgen"):
    """Dynamically load a scenario module by name (handles hyphens in dir names).

    Args:
        scenario_name: e.g. "swe", "terminal-use"
        module_name: "loadgen" or "prepare_data"

    Returns:
        The loaded module object.
    """
    # Try underscore variant first (for Python-importable dirs)
    dir_name = scenario_name.replace("-", "_")
    file_path = REQUEST_DIR / scenario_name / f"{module_name}.py"
    if not file_path.exists():
        file_path = REQUEST_DIR / dir_name / f"{module_name}.py"
    if not file_path.exists():
        raise FileNotFoundError(
            f"No {module_name}.py found for scenario '{scenario_name}' "
            f"(looked in {REQUEST_DIR}/{scenario_name}/ and {REQUEST_DIR}/{dir_name}/)"
        )
    spec = importlib.util.spec_from_file_location(
        f"agsbench_request.{scenario_name}.{module_name}", file_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def list_request() -> list[str]:
    """List all available scenario names."""
    return [
        d.name
        for d in REQUEST_DIR.iterdir()
        if d.is_dir() and (d / "loadgen.py").exists() and not d.name.startswith("_")
    ]
