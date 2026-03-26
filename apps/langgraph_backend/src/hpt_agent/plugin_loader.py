from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Callable

from .storage import TaskStorage


def register_task_from_files(
    storage: TaskStorage,
    plugin_file: Path | str,
    search_space_file: Path | str,
    task_name: str | None = None,
    entrypoint: str = "objective",
) -> str:
    plugin_path = Path(plugin_file)
    search_space_path = Path(search_space_file)
    plugin_bytes = plugin_path.read_bytes()
    search_space_payload = json.loads(search_space_path.read_text(encoding="utf-8"))
    return storage.create_task(
        plugin_bytes=plugin_bytes,
        plugin_filename=plugin_path.name,
        search_space_payload=search_space_payload,
        task_name=task_name,
        entrypoint=entrypoint,
    )


def load_objective_callable(task: dict) -> Callable[[dict], float]:
    module_path = Path(task["module_path"]).resolve()
    entrypoint = task["entrypoint"]
    module_name = f"hpt_plugin_{task['task_id']}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, entrypoint):
        raise AttributeError(f"Entrypoint '{entrypoint}' not found in plugin module.")
    fn = getattr(module, entrypoint)
    if not callable(fn):
        raise TypeError(f"Entrypoint '{entrypoint}' is not callable.")
    return fn


def validate_task_entrypoint(task: dict) -> None:
    load_objective_callable(task)

