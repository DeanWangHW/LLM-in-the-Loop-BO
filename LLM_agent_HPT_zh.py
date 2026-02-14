"""中文版本：将每种参数搜索方法拆分为独立 graph。"""

from LLM_agent_HPT import LLAMAGENT_HPT, LLAMAGENT_L_HPT  # noqa: F401
from hpt_search_graphs import (
    build_bo_graph,
    build_constrained_graph,
    build_justify_graph,
    build_llambo_graph,
    build_llambo_l_graph,
    build_rs_graph,
    build_transient_graph,
)


class LLMIBO_HPT_ZH:
    """HPT 主入口：每个搜索方法由独立 graph 运行。"""

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

        self.graphs = {
            "rs": build_rs_graph(),
            "llambo": build_llambo_graph(),
            "llambo_l": build_llambo_l_graph(),
            "bo": build_bo_graph(),
            "transient": build_transient_graph(),
            "justify": build_justify_graph(),
            "constrained": build_constrained_graph(),
        }

        if self.method not in self.graphs:
            raise ValueError(f"Method '{self.method}' is not implemented.")

    def run(self):
        return self.graphs[self.method].run(self)
