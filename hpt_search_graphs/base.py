from dataclasses import dataclass
from typing import Any, Callable, Dict

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None


@dataclass
class GraphRunnable:
    name: str
    graph: Any

    def run(self, context: Any):
        result = self.graph.invoke({"ctx": context})
        return result["result"]


def build_single_node_graph(name: str, run_fn: Callable[[Any], Any]) -> GraphRunnable:
    """Build a langgraph with START -> run_node -> END for one search method."""

    def run_node(state: Dict[str, Any]) -> Dict[str, Any]:
        ctx = state["ctx"]
        return {"ctx": ctx, "result": run_fn(ctx)}

    if StateGraph is None:  # lightweight fallback if langgraph is unavailable
        class _FallbackGraph:
            def invoke(self, state):
                return run_node(state)

        return GraphRunnable(name=name, graph=_FallbackGraph())

    workflow = StateGraph(dict)
    workflow.add_node("run_node", run_node)
    workflow.add_edge(START, "run_node")
    workflow.add_edge("run_node", END)
    return GraphRunnable(name=name, graph=workflow.compile())


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
        self.graphs: Dict[str, GraphRunnable] = {}

    def run(self):
        if self.method not in self.graphs:
            raise ValueError(f"Method '{self.method}' is not implemented.")
        return self.graphs[self.method].run(self)
