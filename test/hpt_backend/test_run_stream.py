import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BACKEND_SRC_CANDIDATES = (
    ROOT / "apps" / "langgraph_backend" / "src",
    ROOT / "langgraph_hpt" / "backend" / "src",
)
BACKEND_SRC = next(
    (path for path in BACKEND_SRC_CANDIDATES if path.exists()),
    BACKEND_SRC_CANDIDATES[0],
)
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from hpt_agent.graph import stream_run_events
from hpt_agent.plugin_loader import register_task_from_files
from hpt_agent.storage import TaskStorage


def _prepare_task(tmp_path: Path) -> tuple[str, TaskStorage]:
    plugin_path = tmp_path / "objective_plugin.py"
    plugin_path.write_text(
        """
def objective(params: dict) -> float:
    x = float(params["x"])
    y = float(params["y"])
    return (x - 0.1) ** 2 + (y + 0.2) ** 2
""".strip()
        + "\n",
        encoding="utf-8",
    )

    space_path = tmp_path / "search_space.json"
    space_path.write_text(
        json.dumps(
            {
                "parameters": [
                    {"name": "x", "type": "float", "lb": -1.0, "ub": 1.0},
                    {"name": "y", "type": "float", "lb": -1.0, "ub": 1.0},
                ]
            }
        ),
        encoding="utf-8",
    )
    storage = TaskStorage(tmp_path / "storage")
    task_id = register_task_from_files(storage, plugin_path, space_path, "stream-demo")
    return task_id, storage


def test_stream_run_events_has_core_phases_and_final_result(tmp_path: Path):
    task_id, storage = _prepare_task(tmp_path)
    payload = {
        "task_id": task_id,
        "method": "rs",
        "T": 2,
        "T_ini": 1,
        "T_rep": 1,
        "seed": 9,
        "objective_timeout_s": 3.0,
        "storage_root": str(storage.root),
    }

    events = list(stream_run_events(payload))
    phases = [event["phase"] for event in events]

    assert "task_loaded" in phases
    assert "schema_validated" in phases
    assert "plugin_loaded" in phases
    assert "optimizer_initialized" in phases
    assert "iteration_completed" in phases
    assert phases[-1] == "run_completed"

    final = events[-1]
    assert "result" in final
    assert isinstance(final["result"]["regrets"], list)
    assert np.array(final["result"]["regrets"]).shape == (1, 3)
