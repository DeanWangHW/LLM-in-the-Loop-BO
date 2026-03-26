from __future__ import annotations

from typing import Any, Dict, List, Tuple, TypedDict


HistoryPoint = Tuple[Dict[str, Any], float]


class HPTBackendState(TypedDict, total=False):
    run_config: Any
    system_config: Any
    storage: Any
    task: Dict[str, Any]
    search_space: List[Dict[str, Any]]

    status: str
    phase: str

    rep_idx: int
    iter_idx: int
    history: List[HistoryPoint]
    regret: List[float]
    histories_all: List[List[HistoryPoint]]
    regrets_all: List[List[float]]

    adapter: Any
    candidate: Dict[str, Any]
    candidate_source: str
    candidate_diagnostics: Dict[str, Any]
    candidate_value: float

    best_params: Dict[str, Any]
    best_value: float
    error_count: int
    diagnostics: Dict[str, Any]
    result: Any

