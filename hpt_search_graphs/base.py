from abc import ABC, abstractmethod
from typing import Any, Dict

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None


class BaseHPTMethodGraph(ABC):
    """ABC for one HPT method graph (one class per method)."""

    def __init__(self, name: str):
        self.name = name
        self._graph = self._compile_graph()

    def _compile_graph(self):
        if StateGraph is None:  # pragma: no cover
            class _FallbackGraph:
                def __init__(self, owner):
                    self.owner = owner

                def invoke(self, state: Dict[str, Any]):
                    ctx = state["ctx"]
                    return {"ctx": ctx, "result": self.owner.execute(ctx)}

            return _FallbackGraph(self)

        workflow = StateGraph(dict)
        workflow.add_node("execute", self._execute_node)
        workflow.add_edge(START, "execute")
        workflow.add_edge("execute", END)
        return workflow.compile(name=f"hpt-{self.name}-graph")

    def _execute_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ctx = state["ctx"]
        return {"ctx": ctx, "result": self.execute(ctx)}

    @abstractmethod
    def execute(self, context: Any):
        """Run one full optimization workflow for this method."""

    def run(self, context: Any):
        result = self._graph.invoke({"ctx": context})
        return result["result"]


class HPTWorkflowBase:
    """Shared runtime context for HPT search graphs."""

    def __init__(self, method, bounds, objective, dim, desc, T=20, T_ini=None, T_rep=1, verbose=True):
        self.method = method.lower()
        self.obj = objective
        self.dim = dim
        self.desc = desc
        self.T = T
        self.T_ini = T_ini if T_ini is not None else dim
        self.T_rep = T_rep
        self.verbose = verbose
        self.bounds = bounds
        self.graphs: Dict[str, BaseHPTMethodGraph] = {}

    def run(self):
        if self.method not in self.graphs:
            raise ValueError(f"Method '{self.method}' is not implemented.")
        return self.graphs[self.method].run(self)
