from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class OptimizerAdapter(ABC):
    def __init__(self):
        self._observations: List[Tuple[Dict[str, Any], float]] = []

    def warmstart(self, t_ini: int, evaluate_fn) -> List[Tuple[Dict[str, Any], float]]:
        history: List[Tuple[Dict[str, Any], float]] = []
        for _ in range(t_ini):
            params, _, _ = self.suggest(history=history, iter_idx=len(history))
            value = float(evaluate_fn(params))
            self.observe(params, value)
            history.append((dict(params), value))
        return history

    @abstractmethod
    def suggest(self, history: List[Tuple[Dict[str, Any], float]], iter_idx: int):
        """Return (candidate_params, source, diagnostics)."""

    def observe(self, params: Dict[str, Any], value: float) -> None:
        self._observations.append((dict(params), float(value)))

    def best(self) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
        if not self._observations:
            return None, None
        params, value = min(self._observations, key=lambda item: item[1])
        return dict(params), float(value)

