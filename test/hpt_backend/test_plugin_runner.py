import json
import sys
from pathlib import Path

import pytest


BACKEND_SRC = Path(__file__).resolve().parents[2] / "langgraph_hpt" / "backend" / "src"
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))

from hpt_agent.objective_runner import evaluate_objective
from hpt_agent.plugin_loader import register_task_from_files
from hpt_agent.storage import TaskStorage


def _write_sample_plugin(path: Path):
    path.write_text(
        """
def objective(params: dict) -> float:
    depth = int(params["depth"])
    lr = float(params["lr"])
    use_bias = bool(params["use_bias"])
    kernel = params["kernel"]
    penalty = 0.0 if kernel == "rbf" else 0.1
    bias_penalty = 0.0 if use_bias else 0.2
    return abs(depth - 5) * 0.1 + abs(lr - 0.03) + penalty + bias_penalty
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _space_payload():
    return {
        "parameters": [
            {"name": "lr", "type": "float", "lb": 1e-4, "ub": 0.1},
            {"name": "depth", "type": "int", "lb": 2, "ub": 10},
            {"name": "use_bias", "type": "bool"},
            {"name": "kernel", "type": "categorical", "choices": ["linear", "rbf"]},
        ]
    }


def test_register_task_and_evaluate_objective(tmp_path: Path):
    plugin_path = tmp_path / "objective_plugin.py"
    _write_sample_plugin(plugin_path)

    space_path = tmp_path / "search_space.json"
    space_path.write_text(json.dumps(_space_payload()), encoding="utf-8")

    storage = TaskStorage(tmp_path / "storage")
    task_id = register_task_from_files(
        storage=storage,
        plugin_file=plugin_path,
        search_space_file=space_path,
        task_name="demo-task",
    )

    task = storage.load_task(task_id)
    value = evaluate_objective(
        task=task,
        params={"lr": 0.03, "depth": 5, "use_bias": True, "kernel": "rbf"},
        timeout_s=5,
    )
    assert value == pytest.approx(0.0)


def test_evaluate_objective_timeout(tmp_path: Path):
    plugin_path = tmp_path / "objective_plugin.py"
    plugin_path.write_text(
        """
import time

def objective(params: dict) -> float:
    time.sleep(2)
    return 1.0
""".strip()
        + "\n",
        encoding="utf-8",
    )

    space_path = tmp_path / "search_space.json"
    space_path.write_text(json.dumps({"parameters": []}), encoding="utf-8")

    storage = TaskStorage(tmp_path / "storage")
    task_id = register_task_from_files(
        storage=storage,
        plugin_file=plugin_path,
        search_space_file=space_path,
        task_name="timeout-task",
    )
    task = storage.load_task(task_id)

    with pytest.raises(TimeoutError):
        evaluate_objective(task=task, params={}, timeout_s=0.2)

