from ..agents.hpt import LLAMAGENT_HPT

from .base import BaseHPTMethodGraph, HPTGraphConfig, HPTGraphState


class LLAMBOGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="llambo")

    def initialize_history(self, config: HPTGraphConfig):
        return LLAMAGENT_HPT([], func_desc=config.desc).llm_warmstarting(
            num_warmstart=config.T_ini,
            objective_function=config.objective,
        )

    def propose_candidate(self, config: HPTGraphConfig, state: HPTGraphState):
        candidate = LLAMAGENT_HPT(state.history, func_desc=config.desc).find_best_candidate()
        return candidate, "llm_surrogate", {}


def build_llambo_graph():
    return LLAMBOGraph()
