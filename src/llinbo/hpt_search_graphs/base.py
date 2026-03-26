from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None


HistoryPoint = Tuple[Tuple[float, ...], float]


def _to_float_tuple(values: Sequence[Any]) -> Tuple[float, ...]:
    if hasattr(values, "tolist"):
        values = values.tolist()
    return tuple(float(v) for v in values)


def _normalize_history(history: Sequence[Tuple[Sequence[Any], Any]]) -> List[HistoryPoint]:
    normalized: List[HistoryPoint] = []
    for x, y in history:
        normalized.append((_to_float_tuple(x), float(y)))
    return normalized


@dataclass(frozen=True)
class HPTGraphConfig:
    method: str
    bounds: Any
    objective: Any
    dim: int
    desc: Dict[str, Any]
    T: int
    T_ini: int
    T_rep: int
    verbose: bool = True
    seed: Optional[int] = None
    max_retries: int = 5
    initial_history: Optional[List[HistoryPoint]] = None

    @classmethod
    def from_context(cls, context: Any) -> "HPTGraphConfig":
        initial_history = getattr(context, "initial_history", None)
        if initial_history is not None:
            initial_history = _normalize_history(initial_history)

        max_retries = int(getattr(context, "max_retries", 5))
        max_retries = max(max_retries, 1)

        return cls(
            method=getattr(context, "method", "custom").lower(),
            bounds=getattr(context, "bounds"),
            objective=getattr(context, "obj"),
            dim=int(getattr(context, "dim")),
            desc=getattr(context, "desc", {}),
            T=int(getattr(context, "T")),
            T_ini=int(getattr(context, "T_ini")),
            T_rep=int(getattr(context, "T_rep")),
            verbose=bool(getattr(context, "verbose", True)),
            seed=getattr(context, "seed", None),
            max_retries=max_retries,
            initial_history=initial_history,
        )


@dataclass
class HPTGraphState:
    config: HPTGraphConfig
    rep_idx: int = 0
    iter_idx: int = 0
    history: List[HistoryPoint] = field(default_factory=list)
    regret: List[float] = field(default_factory=list)
    histories_all: List[List[HistoryPoint]] = field(default_factory=list)
    regrets_all: List[List[float]] = field(default_factory=list)
    candidate: Optional[Tuple[float, ...]] = None
    candidate_source: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    candidate_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "rep_idx": self.rep_idx,
            "iter_idx": self.iter_idx,
            "history": self.history,
            "regret": self.regret,
            "histories_all": self.histories_all,
            "regrets_all": self.regrets_all,
            "candidate": self.candidate,
            "candidate_source": self.candidate_source,
            "diagnostics": self.diagnostics,
            "error_count": self.error_count,
            "candidate_value": self.candidate_value,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "HPTGraphState":
        return cls(
            config=payload["config"],
            rep_idx=int(payload.get("rep_idx", 0)),
            iter_idx=int(payload.get("iter_idx", 0)),
            history=list(payload.get("history", [])),
            regret=list(payload.get("regret", [])),
            histories_all=list(payload.get("histories_all", [])),
            regrets_all=list(payload.get("regrets_all", [])),
            candidate=payload.get("candidate"),
            candidate_source=payload.get("candidate_source"),
            diagnostics=dict(payload.get("diagnostics", {})),
            error_count=int(payload.get("error_count", 0)),
            candidate_value=payload.get("candidate_value"),
        )


class BaseHPTMethodGraph(ABC):
    """Base class for one HPT method graph with explicit config/state separation."""

    def __init__(self, name: str):
        self.name = name
        self._graph = self._compile_graph()

    def _compile_graph(self):
        if StateGraph is None:  # pragma: no cover

            class _FallbackGraph:
                def __init__(self, owner: "BaseHPTMethodGraph"):
                    self.owner = owner

                def invoke(self, payload: Dict[str, Any]):
                    return self.owner._invoke_fallback(payload)

            return _FallbackGraph(self)

        workflow = StateGraph(dict)
        workflow.add_node("init_run", self._init_run_node)
        workflow.add_node("init_rep", self._init_rep_node)
        workflow.add_node("warmstart_or_load", self._warmstart_or_load_node)
        workflow.add_node("iter_router", self._passthrough_node)
        workflow.add_node("propose_candidate", self._propose_candidate_node)
        workflow.add_node("evaluate_candidate", self._evaluate_candidate_node)
        workflow.add_node("update_history_regret", self._update_history_regret_node)
        workflow.add_node("finalize_rep", self._finalize_rep_node)
        workflow.add_node("rep_router", self._passthrough_node)
        workflow.add_node("aggregate_output", self._aggregate_output_node)

        workflow.add_edge(START, "init_run")
        workflow.add_edge("init_run", "init_rep")
        workflow.add_edge("init_rep", "warmstart_or_load")
        workflow.add_edge("warmstart_or_load", "iter_router")
        workflow.add_conditional_edges(
            "iter_router",
            self._route_iter,
            {"propose_candidate": "propose_candidate", "finalize_rep": "finalize_rep"},
        )
        workflow.add_edge("propose_candidate", "evaluate_candidate")
        workflow.add_edge("evaluate_candidate", "update_history_regret")
        workflow.add_edge("update_history_regret", "iter_router")
        workflow.add_edge("finalize_rep", "rep_router")
        workflow.add_conditional_edges(
            "rep_router",
            self._route_rep,
            {"init_rep": "init_rep", "aggregate_output": "aggregate_output"},
        )
        workflow.add_edge("aggregate_output", END)
        return workflow.compile(name=f"hpt-{self.name}-graph")

    def _invoke_fallback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = self._init_run_state(payload["config"])

        while state.rep_idx < state.config.T_rep:
            state = self._init_rep(state)
            state = self._warmstart_or_load(state)

            while state.iter_idx < state.config.T:
                state = self._propose_candidate(state)
                state = self._evaluate_candidate(state)
                state = self._update_history_regret(state)

            state = self._finalize_rep(state)

        return self._aggregate_output(state)

    def _init_run_state(self, config: HPTGraphConfig) -> HPTGraphState:
        if config.seed is not None:
            np.random.seed(config.seed)
            if torch is not None:  # pragma: no branch
                torch.manual_seed(config.seed)
        return HPTGraphState(config=config)

    @staticmethod
    def _passthrough_node(state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    def _route_iter(self, state: Dict[str, Any]) -> str:
        config: HPTGraphConfig = state["config"]
        return "propose_candidate" if int(state["iter_idx"]) < config.T else "finalize_rep"

    def _route_rep(self, state: Dict[str, Any]) -> str:
        config: HPTGraphConfig = state["config"]
        return "init_rep" if int(state["rep_idx"]) < config.T_rep else "aggregate_output"

    def _init_run_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        config = state["config"]
        if not isinstance(config, HPTGraphConfig):
            config = HPTGraphConfig.from_context(config)
        return self._init_run_state(config).to_dict()

    @staticmethod
    def _init_rep(state: HPTGraphState) -> HPTGraphState:
        state.iter_idx = 0
        state.history = []
        state.regret = []
        state.candidate = None
        state.candidate_source = None
        state.candidate_value = None
        state.diagnostics = {}
        state.error_count = 0
        return state

    def _init_rep_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return self._init_rep(HPTGraphState.from_dict(state)).to_dict()

    def _warmstart_or_load(self, state: HPTGraphState) -> HPTGraphState:
        if state.config.initial_history is not None:
            history = copy.deepcopy(state.config.initial_history)
        else:
            history = self.initialize_history(state.config)

        normalized = _normalize_history(history)
        if not normalized:
            raise ValueError("Warmstart history cannot be empty.")
        state.history = normalized
        state.regret = [min(y for _, y in normalized)]
        return state

    def _warmstart_or_load_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        runtime = HPTGraphState.from_dict(state)
        runtime = self._warmstart_or_load(runtime)
        return runtime.to_dict()

    @staticmethod
    def _normalize_candidate(candidate: Any, dim: int) -> Tuple[float, ...]:
        if isinstance(candidate, tuple) and len(candidate) == 1 and hasattr(candidate[0], "tolist"):
            candidate = candidate[0].tolist()
        elif hasattr(candidate, "tolist"):
            candidate = candidate.tolist()

        if not isinstance(candidate, (list, tuple)):
            raise ValueError(f"Candidate must be a sequence, got {type(candidate)}")

        candidate_tuple = tuple(float(v) for v in candidate)
        if len(candidate_tuple) != dim:
            raise ValueError(f"Candidate dimension mismatch: expected {dim}, got {len(candidate_tuple)}")
        return candidate_tuple

    @staticmethod
    def _run_with_retries(fn, max_retries: int):
        last_exc = None
        for attempt in range(max_retries):
            try:
                return fn(), attempt
            except Exception as exc:  # pragma: no cover - retry behavior
                last_exc = exc
        raise RuntimeError(f"Exceeded max retries ({max_retries})") from last_exc

    def _propose_candidate(self, state: HPTGraphState) -> HPTGraphState:
        (candidate, source, diagnostics), retry_count = self._run_with_retries(
            lambda: self.propose_candidate(state.config, state),
            state.config.max_retries,
        )
        state.candidate = self._normalize_candidate(candidate, state.config.dim)
        state.candidate_source = source
        state.diagnostics = diagnostics or {}
        state.error_count += retry_count
        return state

    def _propose_candidate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        runtime = HPTGraphState.from_dict(state)
        runtime = self._propose_candidate(runtime)
        return runtime.to_dict()

    @staticmethod
    def _evaluate_candidate(state: HPTGraphState) -> HPTGraphState:
        if state.candidate is None:
            raise ValueError("Candidate must be set before evaluation.")
        state.candidate_value = float(state.config.objective(list(state.candidate)))
        return state

    def _evaluate_candidate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        runtime = HPTGraphState.from_dict(state)
        runtime = self._evaluate_candidate(runtime)
        return runtime.to_dict()

    @staticmethod
    def _update_history_regret(state: HPTGraphState) -> HPTGraphState:
        if state.candidate is None or state.candidate_value is None:
            raise ValueError("Candidate and candidate value must be set before updating history.")
        state.history.append((state.candidate, float(state.candidate_value)))
        state.regret.append(min(y for _, y in state.history))
        state.iter_idx += 1
        state.candidate = None
        state.candidate_source = None
        state.candidate_value = None
        return state

    def _update_history_regret_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        runtime = HPTGraphState.from_dict(state)
        runtime = self._update_history_regret(runtime)
        return runtime.to_dict()

    @staticmethod
    def _finalize_rep(state: HPTGraphState) -> HPTGraphState:
        state.histories_all.append(copy.deepcopy(state.history))
        state.regrets_all.append(list(state.regret))
        state.rep_idx += 1
        return state

    def _finalize_rep_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        runtime = HPTGraphState.from_dict(state)
        runtime = self._finalize_rep(runtime)
        return runtime.to_dict()

    @staticmethod
    def _aggregate_output(state: HPTGraphState) -> Dict[str, Any]:
        regrets_array = np.array(state.regrets_all, dtype=float)
        payload = state.to_dict()
        payload["result"] = (state.histories_all, regrets_array)
        return payload

    def _aggregate_output_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        runtime = HPTGraphState.from_dict(state)
        return self._aggregate_output(runtime)

    @abstractmethod
    def initialize_history(self, config: HPTGraphConfig) -> Sequence[Tuple[Sequence[Any], Any]]:
        """Create initial history for one repetition."""

    @abstractmethod
    def propose_candidate(self, config: HPTGraphConfig, state: HPTGraphState) -> Tuple[Sequence[Any], str, Dict[str, Any]]:
        """Propose the next candidate point from the current state."""

    def run(self, context: Any):
        config = HPTGraphConfig.from_context(context)
        result = self._graph.invoke({"config": config})
        return result["result"]


class HPTWorkflowBase:
    """Shared runtime context for HPT search graphs."""

    def __init__(
        self,
        method,
        bounds,
        objective,
        dim,
        desc,
        T=20,
        T_ini=None,
        T_rep=1,
        verbose=True,
        seed=None,
        max_retries=5,
        initial_history=None,
    ):
        self.method = method.lower()
        self.obj = objective
        self.dim = dim
        self.desc = desc
        self.T = T
        self.T_ini = T_ini if T_ini is not None else dim
        self.T_rep = T_rep
        self.verbose = verbose
        self.bounds = bounds
        self.seed = seed
        self.max_retries = max_retries
        self.initial_history = initial_history
        self.graphs: Dict[str, BaseHPTMethodGraph] = {}

    def run(self):
        if self.method not in self.graphs:
            raise ValueError(f"Method '{self.method}' is not implemented.")
        return self.graphs[self.method].run(self)
