from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from .base import OptimizerAdapter
from ..search_space import to_hebo_design_space

try:  # pragma: no cover - optional dependency
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO
except Exception:  # pragma: no cover
    DesignSpace = None
    HEBO = None


class HEBOAdapter(OptimizerAdapter):
    def __init__(self, search_space: List[Dict[str, Any]], seed: int | None = None):
        if DesignSpace is None or HEBO is None:
            raise ImportError("HEBO is not installed. Please install 'hebo' to use method='hebo'.")
        super().__init__()
        if seed is not None:
            np.random.seed(seed)
        hebo_space = to_hebo_design_space(search_space)
        space = DesignSpace().parse(hebo_space)
        self._optimizer = HEBO(space)
        self._last_rec = None
        self._last_params: Dict[str, Any] | None = None

    @staticmethod
    def _to_python_value(value):
        if hasattr(value, "item"):
            return value.item()
        return value

    def suggest(self, history: List[Tuple[Dict[str, Any], float]], iter_idx: int):
        rec = self._optimizer.suggest(n_suggestions=1)
        params = {k: self._to_python_value(v) for k, v in rec.iloc[0].to_dict().items()}
        self._last_rec = rec
        self._last_params = dict(params)
        return params, "hebo", {"iter_idx": int(iter_idx)}

    def observe(self, params: Dict[str, Any], value: float) -> None:
        rec = self._last_rec
        if rec is None:
            import pandas as pd

            rec = pd.DataFrame([params])
        y = np.array([[float(value)]], dtype=float)
        self._optimizer.observe(rec, y)
        super().observe(params, value)

