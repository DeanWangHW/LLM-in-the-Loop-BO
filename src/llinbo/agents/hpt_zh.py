"""中文版本：将每种参数搜索方法拆分为独立 langgraph。"""

from ..hpt_search_graphs import (
    build_bo_graph,
    build_constrained_graph,
    build_justify_graph,
    build_llambo_graph,
    build_llambo_l_graph,
    build_rs_graph,
    build_transient_graph,
)
from ..hpt_search_graphs.base import HPTWorkflowBase


class LLMIBO_HPT_ZH(HPTWorkflowBase):
    """HPT 主入口：每个搜索方法由独立 langgraph 运行。"""

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
        super().__init__(
            method=method,
            bounds=bounds,
            objective=objective,
            dim=dim,
            desc=desc,
            T=T,
            T_ini=T_ini,
            T_rep=T_rep,
            verbose=verbose,
            seed=seed,
            max_retries=max_retries,
            initial_history=initial_history,
        )
        self.graphs = {
            "rs": build_rs_graph(),
            "llambo": build_llambo_graph(),
            "llambo_l": build_llambo_l_graph(),
            "bo": build_bo_graph(),
            "transient": build_transient_graph(),
            "justify": build_justify_graph(),
            "constrained": build_constrained_graph(),
        }
