from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict, Iterable

import numpy as np

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from hpt_agent.configuration import RunConfig, SystemConfig
from hpt_agent.objective_runner import evaluate_objective
from hpt_agent.optimizers import HEBOAdapter, LegacyAdapter
from hpt_agent.plugin_loader import validate_task_entrypoint
from hpt_agent.search_space import parse_search_space
from hpt_agent.storage import TaskStorage

try:  # pragma: no cover - optional dependency
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None


class HPTBackendWorkflow:
    def __init__(self, system_config: SystemConfig | None = None):
        self.system_config = system_config or SystemConfig.from_env()
        self.graph = self._compile_graph()

    @staticmethod
    def _passthrough(state: Dict[str, Any]) -> Dict[str, Any]:
        return state

    @staticmethod
    def _with_update(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        merged = dict(state)
        merged.update(kwargs)
        return merged

    def _route_iter(self, state: Dict[str, Any]) -> str:
        run_config: RunConfig = state["run_config"]
        return "suggest_candidate" if int(state["iter_idx"]) < run_config.T else "finalize_rep"

    def _route_rep(self, state: Dict[str, Any]) -> str:
        run_config: RunConfig = state["run_config"]
        return "init_rep" if int(state["rep_idx"]) < run_config.T_rep else "finalize_run"

    def _compile_graph(self):
        if StateGraph is None:  # pragma: no cover
            return _FallbackGraph(self)

        workflow = StateGraph(dict)
        workflow.add_node("init_run", self._init_run_node)
        workflow.add_node("load_task", self._load_task_node)
        workflow.add_node("validate_schema", self._validate_schema_node)
        workflow.add_node("load_objective", self._load_objective_node)
        workflow.add_node("init_optimizer", self._init_optimizer_node)
        workflow.add_node("init_rep", self._init_rep_node)
        workflow.add_node("iter_router", self._passthrough)
        workflow.add_node("suggest_candidate", self._suggest_candidate_node)
        workflow.add_node("evaluate_candidate", self._evaluate_candidate_node)
        workflow.add_node("observe_update", self._observe_update_node)
        workflow.add_node("finalize_rep", self._finalize_rep_node)
        workflow.add_node("rep_router", self._passthrough)
        workflow.add_node("finalize_run", self._finalize_run_node)

        workflow.add_edge(START, "init_run")
        workflow.add_edge("init_run", "load_task")
        workflow.add_edge("load_task", "validate_schema")
        workflow.add_edge("validate_schema", "load_objective")
        workflow.add_edge("load_objective", "init_optimizer")
        workflow.add_edge("init_optimizer", "init_rep")
        workflow.add_edge("init_rep", "iter_router")
        workflow.add_conditional_edges(
            "iter_router",
            self._route_iter,
            {
                "suggest_candidate": "suggest_candidate",
                "finalize_rep": "finalize_rep",
            },
        )
        workflow.add_edge("suggest_candidate", "evaluate_candidate")
        workflow.add_edge("evaluate_candidate", "observe_update")
        workflow.add_edge("observe_update", "iter_router")
        workflow.add_edge("finalize_rep", "rep_router")
        workflow.add_conditional_edges(
            "rep_router",
            self._route_rep,
            {"init_rep": "init_rep", "finalize_run": "finalize_run"},
        )
        workflow.add_edge("finalize_run", END)
        return workflow.compile(name="hpt-backend-agent")

    def _init_run_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        run_config = RunConfig.from_payload(state)
        storage_root = run_config.storage_root or self.system_config.storage_root
        storage = TaskStorage(storage_root)
        return self._with_update(
            state,
            run_config=run_config,
            storage=storage,
            status="running",
            phase="run_started",
            rep_idx=0,
            iter_idx=0,
            histories_all=[],
            regrets_all=[],
            error_count=0,
            diagnostics={},
        )

    @staticmethod
    def _load_task_node(state: Dict[str, Any]) -> Dict[str, Any]:
        run_config: RunConfig = state["run_config"]
        storage: TaskStorage = state["storage"]
        task = storage.load_task(run_config.task_id)
        merged = dict(state)
        merged.update({"task": task, "phase": "task_loaded"})
        return merged

    @staticmethod
    def _validate_schema_node(state: Dict[str, Any]) -> Dict[str, Any]:
        specs = parse_search_space(state["task"]["search_space_raw"])
        merged = dict(state)
        merged.update({"search_space": specs, "phase": "schema_validated"})
        return merged

    @staticmethod
    def _load_objective_node(state: Dict[str, Any]) -> Dict[str, Any]:
        validate_task_entrypoint(state["task"])
        merged = dict(state)
        merged.update({"phase": "plugin_loaded"})
        return merged

    @staticmethod
    def _build_adapter(state: Dict[str, Any]):
        run_config: RunConfig = state["run_config"]
        if run_config.method == "hebo":
            return HEBOAdapter(state["search_space"], seed=run_config.seed)
        return LegacyAdapter(
            method=run_config.method,
            search_space=state["search_space"],
            desc=run_config.desc,
            seed=run_config.seed,
            max_retries=run_config.max_retries,
        )

    def _init_optimizer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        adapter = self._build_adapter(state)
        merged = dict(state)
        merged.update({"adapter": adapter, "phase": "optimizer_initialized"})
        return merged

    def _init_rep_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        adapter = state["adapter"]
        run_config: RunConfig = state["run_config"]
        task = state["task"]

        def objective_fn(params: Dict[str, Any]) -> float:
            return evaluate_objective(task=task, params=params, timeout_s=run_config.objective_timeout_s)

        history = adapter.warmstart(run_config.T_ini, objective_fn)
        regret = [min(v for _, v in history)] if history else []
        best_params, best_value = adapter.best()
        merged = dict(state)
        merged.update(
            {
                "history": history,
                "regret": regret,
                "iter_idx": 0,
                "candidate": None,
                "candidate_source": None,
                "candidate_diagnostics": {},
                "candidate_value": None,
                "best_params": best_params,
                "best_value": best_value,
                "phase": "rep_started",
            }
        )
        return merged

    def _suggest_candidate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        adapter = state["adapter"]
        candidate, source, diagnostics = adapter.suggest(
            history=state["history"],
            iter_idx=int(state["iter_idx"]),
        )
        merged = dict(state)
        merged.update(
            {
                "candidate": candidate,
                "candidate_source": source,
                "candidate_diagnostics": diagnostics,
                "phase": "candidate_suggested",
            }
        )
        return merged

    def _evaluate_candidate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        task = state["task"]
        run_config: RunConfig = state["run_config"]
        value = evaluate_objective(
            task=task,
            params=state["candidate"],
            timeout_s=run_config.objective_timeout_s,
        )
        merged = dict(state)
        merged.update({"candidate_value": value, "phase": "candidate_evaluated"})
        return merged

    @staticmethod
    def _observe_update_node(state: Dict[str, Any]) -> Dict[str, Any]:
        adapter = state["adapter"]
        candidate = dict(state["candidate"])
        value = float(state["candidate_value"])
        adapter.observe(candidate, value)

        history = list(state["history"]) + [(candidate, value)]
        if state["regret"]:
            regret = list(state["regret"]) + [min(state["regret"][-1], value)]
        else:
            regret = [value]

        best_params, best_value = adapter.best()
        merged = dict(state)
        merged.update(
            {
                "history": history,
                "regret": regret,
                "iter_idx": int(state["iter_idx"]) + 1,
                "best_params": best_params,
                "best_value": best_value,
                "phase": "iteration_completed",
            }
        )
        return merged

    @staticmethod
    def _finalize_rep_node(state: Dict[str, Any]) -> Dict[str, Any]:
        histories_all = list(state["histories_all"]) + [list(state["history"])]
        regrets_all = list(state["regrets_all"]) + [list(state["regret"])]
        merged = dict(state)
        merged.update(
            {
                "histories_all": histories_all,
                "regrets_all": regrets_all,
                "rep_idx": int(state["rep_idx"]) + 1,
                "phase": "rep_completed",
            }
        )
        return merged

    @staticmethod
    def _finalize_run_node(state: Dict[str, Any]) -> Dict[str, Any]:
        regrets = np.array(state["regrets_all"], dtype=float)
        result = (state["histories_all"], regrets)
        merged = dict(state)
        merged.update(
            {
                "result": result,
                "status": "succeeded",
                "phase": "run_completed",
            }
        )
        return merged


class _FallbackGraph:
    """Fallback runtime when langgraph package is unavailable."""

    def __init__(self, owner: HPTBackendWorkflow):
        self.owner = owner

    def invoke(self, payload: Dict[str, Any]):
        last_state = None
        for chunk in self.stream(payload, stream_mode="updates"):
            last_state = chunk
        return last_state["__state__"] if last_state else {}

    def stream(self, payload: Dict[str, Any], stream_mode: str = "updates") -> Iterable[Dict[str, Any]]:
        state: Dict[str, Any] = {}

        def run_step(name: str, fn):
            nonlocal state
            update = fn(payload if name == "init_run" else state)
            state.update(update)
            chunk = {name: update, "__state__": dict(state)}
            return chunk

        yield run_step("init_run", self.owner._init_run_node)
        yield run_step("load_task", self.owner._load_task_node)
        yield run_step("validate_schema", self.owner._validate_schema_node)
        yield run_step("load_objective", self.owner._load_objective_node)
        yield run_step("init_optimizer", self.owner._init_optimizer_node)

        while True:
            yield run_step("init_rep", self.owner._init_rep_node)
            while self.owner._route_iter(state) == "suggest_candidate":
                yield run_step("suggest_candidate", self.owner._suggest_candidate_node)
                yield run_step("evaluate_candidate", self.owner._evaluate_candidate_node)
                yield run_step("observe_update", self.owner._observe_update_node)
            yield run_step("finalize_rep", self.owner._finalize_rep_node)
            if self.owner._route_rep(state) != "init_rep":
                break

        yield run_step("finalize_run", self.owner._finalize_run_node)


graph = HPTBackendWorkflow().graph


def _jsonable_result(result):
    histories, regrets = result
    return {
        "histories": histories,
        "regrets": np.asarray(regrets, dtype=float).tolist(),
    }


def run_once(payload: Dict[str, Any]) -> Dict[str, Any]:
    response = graph.invoke(payload)
    result = response.get("result")
    if result is None:
        raise RuntimeError("Graph run did not return result.")
    return _jsonable_result(result)


def stream_run_events(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for chunk in graph.stream(payload, stream_mode="updates"):
        if not isinstance(chunk, dict):
            continue
        for node, update in chunk.items():
            if not isinstance(update, dict):
                continue
            phase = update.get("phase", node)
            event: Dict[str, Any] = {
                "node": node,
                "phase": phase,
            }
            for key in (
                "status",
                "task_id",
                "rep_idx",
                "iter_idx",
                "candidate",
                "candidate_source",
                "candidate_value",
                "best_value",
            ):
                if key in update:
                    event[key] = update[key]
            if "result" in update:
                event["result"] = _jsonable_result(update["result"])
            yield event
