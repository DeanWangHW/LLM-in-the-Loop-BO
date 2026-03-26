from ..agents.hpt import LLAMAGENT_L_HPT

from .base import BaseHPTMethodGraph, HPTGraphConfig, HPTGraphState


class LLAMBOLGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="llambo_l")

    def initialize_history(self, config: HPTGraphConfig):
        return LLAMAGENT_L_HPT([], func_desc=config.desc).llm_warmstarting(
            num_warmstart=config.T_ini,
            objective_function=config.objective,
        )

    def propose_candidate(self, config: HPTGraphConfig, state: HPTGraphState):
        candidate = LLAMAGENT_L_HPT(state.history, func_desc=config.desc).sample_candidate_points()
        return candidate, "llm", {}


def build_llambo_l_graph():
    return LLAMBOLGraph()
