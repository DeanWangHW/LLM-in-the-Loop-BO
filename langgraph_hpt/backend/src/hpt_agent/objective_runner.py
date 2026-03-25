from __future__ import annotations

import importlib.util
import multiprocessing as mp
import queue
from pathlib import Path
from typing import Any, Dict

from .search_space import normalize_params, parse_search_space


def _worker(module_path: str, entrypoint: str, params: Dict[str, Any], out_q: mp.Queue):
    try:
        module_name = f"hpt_eval_{Path(module_path).stem}_{id(params)}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load plugin module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        fn = getattr(module, entrypoint)
        if not callable(fn):
            raise TypeError(f"Entrypoint '{entrypoint}' is not callable.")
        value = float(fn(params))
        out_q.put({"ok": True, "value": value})
    except Exception as exc:  # pragma: no cover - process boundary
        out_q.put({"ok": False, "error": repr(exc)})


def evaluate_objective(task: dict, params: Dict[str, Any], timeout_s: float = 30.0) -> float:
    specs = parse_search_space(task["search_space_raw"])
    normalized = normalize_params(params, specs)
    out_q: mp.Queue = mp.Queue(maxsize=1)
    proc = mp.Process(
        target=_worker,
        args=(str(task["module_path"]), task["entrypoint"], normalized, out_q),
        daemon=True,
    )
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=1)
        raise TimeoutError(f"Objective execution timed out after {timeout_s} seconds.")

    try:
        payload = out_q.get_nowait()
    except queue.Empty as exc:
        raise RuntimeError("Objective process exited without a result.") from exc

    if not payload.get("ok"):
        raise RuntimeError(payload.get("error", "Unknown objective error."))
    return float(payload["value"])

