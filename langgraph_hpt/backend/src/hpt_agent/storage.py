from __future__ import annotations

import hashlib
import json
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class TaskStorage:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.tasks_dir = self.root / "tasks"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)

    def _task_dir(self, task_id: str) -> Path:
        return self.tasks_dir / task_id

    @staticmethod
    def _sha256(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def create_task(
        self,
        *,
        plugin_bytes: bytes,
        plugin_filename: str,
        search_space_payload: Dict[str, Any],
        task_name: str | None = None,
        entrypoint: str = "objective",
    ) -> str:
        task_id = uuid.uuid4().hex[:12]
        task_dir = self._task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=False)

        suffix = Path(plugin_filename).suffix.lower()
        if suffix not in {".py", ".zip"}:
            raise ValueError("Plugin file must be .py or .zip")

        plugin_path = task_dir / f"plugin{suffix}"
        plugin_path.write_bytes(plugin_bytes)

        module_path = plugin_path
        if suffix == ".zip":
            extract_dir = task_dir / "plugin_src"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(plugin_path, "r") as zf:
                zf.extractall(extract_dir)
            candidate = extract_dir / "objective_plugin.py"
            if candidate.is_file():
                module_path = candidate
            else:
                py_files = sorted(extract_dir.rglob("*.py"))
                if not py_files:
                    raise ValueError("Zip plugin must contain at least one .py file.")
                module_path = py_files[0]

        search_space_path = task_dir / "search_space.json"
        search_space_path.write_text(
            json.dumps(search_space_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        manifest = {
            "task_id": task_id,
            "task_name": task_name or task_id,
            "entrypoint": entrypoint,
            "plugin_filename": plugin_filename,
            "plugin_sha256": self._sha256(plugin_bytes),
            "plugin_type": suffix.lstrip("."),
            "plugin_path": str(plugin_path),
            "module_path": str(module_path),
            "search_space_path": str(search_space_path),
            "signature": "objective(params: dict) -> float",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (task_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return task_id

    def load_task(self, task_id: str) -> Dict[str, Any]:
        task_dir = self._task_dir(task_id)
        manifest_path = task_dir / "manifest.json"
        search_space_path = task_dir / "search_space.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Task '{task_id}' does not exist.")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        search_space_raw = json.loads(search_space_path.read_text(encoding="utf-8"))
        task = {
            "task_id": task_id,
            "task_dir": str(task_dir),
            "manifest": manifest,
            "entrypoint": manifest["entrypoint"],
            "module_path": manifest["module_path"],
            "search_space_raw": search_space_raw,
            "signature": manifest.get("signature", "objective(params: dict) -> float"),
        }
        return task

