from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SystemConfig:
    storage_root: Path
    max_parallel_runs: int = 4
    sandbox_policy: str = "subprocess"

    @classmethod
    def from_env(cls) -> "SystemConfig":
        root = os.environ.get("HPT_STORAGE_ROOT")
        if root is None:
            root = "apps/langgraph_backend/.hpt_data"
        return cls(
            storage_root=Path(root).resolve(),
            max_parallel_runs=int(os.environ.get("HPT_MAX_PARALLEL_RUNS", "4")),
            sandbox_policy=os.environ.get("HPT_SANDBOX_POLICY", "subprocess"),
        )


@dataclass(frozen=True)
class RunConfig:
    task_id: str
    method: str
    T: int
    T_ini: int
    T_rep: int
    seed: Optional[int] = None
    max_retries: int = 5
    objective_timeout_s: float = 30.0
    desc: Optional[Dict[str, Any]] = None
    storage_root: Optional[Path] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "RunConfig":
        if "task_id" not in payload:
            raise ValueError("Missing required field 'task_id'.")
        method = str(payload.get("method", "hebo")).lower()
        storage_root = payload.get("storage_root")
        return cls(
            task_id=str(payload["task_id"]),
            method=method,
            T=max(int(payload.get("T", 20)), 1),
            T_ini=max(int(payload.get("T_ini", payload.get("dim", 1))), 0),
            T_rep=max(int(payload.get("T_rep", 1)), 1),
            seed=payload.get("seed", None),
            max_retries=max(int(payload.get("max_retries", 5)), 1),
            objective_timeout_s=max(float(payload.get("objective_timeout_s", 30.0)), 0.01),
            desc=payload.get("desc"),
            storage_root=Path(storage_root).resolve() if storage_root else None,
        )
