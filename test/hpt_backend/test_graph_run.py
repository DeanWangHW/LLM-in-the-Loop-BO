import json
import sys
from pathlib import Path


BACKEND_SRC = Path(__file__).resolve().parents[2] / "langgraph_hpt" / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from hpt_agent.graph import graph
from hpt_agent.plugin_loader import register_task_from_files
from hpt_agent.storage import TaskStorage


def _prepare_task(tmp_path: Path) -> tuple[str, TaskStorage]:
    plugin_path = tmp_path / "objective_plugin.py"
    plugin_path.write_text(
        """
def objective(params: dict) -> float:
    x = float(params["x"])
    y = float(params["y"])
    return (x - 0.25) ** 2 + (y + 0.15) ** 2
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
    task_id = register_task_from_files(storage, plugin_path, space_path, "graph-demo")
    return task_id, storage


def test_graph_invoke_returns_histories_and_regrets(tmp_path: Path):
    task_id, storage = _prepare_task(tmp_path)

    payload = {
        "task_id": task_id,
        "method": "rs",
        "T": 3,
        "T_ini": 1,
        "T_rep": 1,
        "seed": 7,
        "objective_timeout_s": 3.0,
        "storage_root": str(storage.root),
    }
    result = graph.invoke(payload)

    assert "result" in result
    histories, regrets = result["result"]
    assert len(histories) == 1
    assert len(histories[0]) == 4
    assert regrets.shape == (1, 4)


def test_graph_stream_emits_step_updates(tmp_path: Path):
    task_id, storage = _prepare_task(tmp_path)
    payload = {
        "task_id": task_id,
        "method": "rs",
        "T": 2,
        "T_ini": 1,
        "T_rep": 1,
        "seed": 11,
        "objective_timeout_s": 3.0,
        "storage_root": str(storage.root),
    }
    updates = list(graph.stream(payload, stream_mode="updates"))

    step_names = [list(chunk.keys())[0] for chunk in updates if isinstance(chunk, dict) and chunk]
    assert "load_task" in step_names
    assert "validate_schema" in step_names
    assert "init_optimizer" in step_names
    assert "observe_update" in step_names
    assert "finalize_run" in step_names

