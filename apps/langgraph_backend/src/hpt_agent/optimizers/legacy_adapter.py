from __future__ import annotations

import importlib
import pathlib
import random
import sys
from typing import Any, Dict, List, Tuple

from .base import OptimizerAdapter
from ..search_space import sample_random_params

LEGACY_METHODS = {
    "rs",
    "bo",
    "llambo",
    "llambo_l",
    "transient",
    "justify",
    "constrained",
}


def _ensure_legacy_graph_path() -> None:
    """Ensure repository root is importable for hpt_search_graphs."""
    repo_root = pathlib.Path(__file__).resolve().parents[5]
    if (repo_root / "hpt_search_graphs").is_dir() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_legacy_graph_components():
    _ensure_legacy_graph_path()
    from hpt_search_graphs import (
        build_bo_graph,
        build_constrained_graph,
        build_justify_graph,
        build_llambo_graph,
        build_llambo_l_graph,
        build_transient_graph,
    )
    from hpt_search_graphs.base import HPTGraphConfig, HPTGraphState

    graph_builders = {
        "bo": build_bo_graph,
        "llambo": build_llambo_graph,
        "llambo_l": build_llambo_l_graph,
        "transient": build_transient_graph,
        "justify": build_justify_graph,
        "constrained": build_constrained_graph,
    }
    return graph_builders, HPTGraphConfig, HPTGraphState


def _load_torch():
    """Load torch lazily so RS mode does not depend on torch runtime."""
    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ImportError("torch is required for legacy graph methods.") from exc


class LegacyAdapter(OptimizerAdapter):
    def __init__(
        self,
        *,
        method: str,
        search_space: List[Dict[str, Any]],
        desc: Dict[str, Any] | None,
        seed: int | None,
        max_retries: int,
    ):
        super().__init__()
        self.method = method.lower()
        if self.method not in LEGACY_METHODS:
            raise ValueError(f"Unknown legacy method: {method}")
        self.search_space = list(search_space)
        self.desc = desc or {}
        self.rng = random.Random(seed)
        self.max_retries = max_retries
        self._graph = None
        self._graph_config_cls = None
        self._graph_state_cls = None
        if self.method != "rs":
            graph_builders, hpt_graph_config_cls, hpt_graph_state_cls = _load_legacy_graph_components()
            self._graph = graph_builders[self.method]()
            self._graph_config_cls = hpt_graph_config_cls
            self._graph_state_cls = hpt_graph_state_cls
        self._base_config: Any = None

    @property
    def _is_numeric(self) -> bool:
        return all(spec["type"] in {"float", "int"} for spec in self.search_space)

    @property
    def _param_names(self) -> List[str]:
        return [spec["name"] for spec in self.search_space]

    def _vector_to_params(self, vector: List[float] | Tuple[float, ...]) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for spec, value in zip(self.search_space, vector):
            if spec["type"] == "int":
                val = int(round(float(value)))
                val = max(spec["lb"], min(spec["ub"], val))
            else:
                val = float(value)
            params[spec["name"]] = val
        return params

    def _params_to_vector(self, params: Dict[str, Any]) -> Tuple[float, ...]:
        return tuple(float(params[name]) for name in self._param_names)

    def warmstart(self, t_ini: int, evaluate_fn) -> List[Tuple[Dict[str, Any], float]]:
        if self.method == "rs":
            return super().warmstart(t_ini, evaluate_fn)

        if not self._is_numeric:
            raise ValueError(f"Legacy method '{self.method}' only supports numeric parameters.")
        torch = _load_torch()

        lower = torch.tensor([spec["lb"] for spec in self.search_space], dtype=torch.float64)
        upper = torch.tensor([spec["ub"] for spec in self.search_space], dtype=torch.float64)

        def objective_list(values: List[float]) -> float:
            return float(evaluate_fn(self._vector_to_params(values)))

        if self._graph_config_cls is None:
            raise RuntimeError("Legacy graph config class is not initialized.")

        config = self._graph_config_cls(
            method=self.method,
            bounds=(lower, upper),
            objective=objective_list,
            dim=len(self.search_space),
            desc=self.desc,
            T=1,
            T_ini=max(int(t_ini), 1),
            T_rep=1,
            verbose=False,
            seed=self.rng.randint(0, 10**9),
            max_retries=self.max_retries,
            initial_history=None,
        )
        self._base_config = config
        history_raw = self._graph.initialize_history(config)
        history: List[Tuple[Dict[str, Any], float]] = []
        for x, y in history_raw:
            params = self._vector_to_params(list(x))
            val = float(y)
            history.append((params, val))
            super().observe(params, val)
        return history

    def suggest(self, history: List[Tuple[Dict[str, Any], float]], iter_idx: int):
        if self.method == "rs":
            params = sample_random_params(self.search_space, self.rng)
            return params, "random", {"iter_idx": int(iter_idx)}

        if self._base_config is None:
            raise RuntimeError("LegacyAdapter must be warmstarted before suggest().")
        if self._graph_state_cls is None:
            raise RuntimeError("Legacy graph state class is not initialized.")

        old_history = [(self._params_to_vector(p), float(v)) for p, v in history]
        old_state = self._graph_state_cls(
            config=self._base_config,
            rep_idx=0,
            iter_idx=int(iter_idx),
            history=old_history,
            regret=[min(v for _, v in old_history)] if old_history else [],
            diagnostics={},
        )
        candidate, source, diagnostics = self._graph.propose_candidate(self._base_config, old_state)
        params = self._vector_to_params(list(candidate))
        return params, source, diagnostics
