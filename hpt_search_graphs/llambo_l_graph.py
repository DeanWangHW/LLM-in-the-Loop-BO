import numpy as np
from tqdm import trange

from LLM_agent_HPT import LLAMAGENT_L_HPT

from .base import BaseHPTMethodGraph


class LLAMBOLGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="llambo_l")

    def execute(self, context):
        regrets, histories = [], []
        for _ in trange(context.T_rep, desc="LLAMBO-L", disable=not context.verbose):
            history = LLAMAGENT_L_HPT([], func_desc=context.desc).llm_warmstarting(
                num_warmstart=context.T_ini, objective_function=context.obj
            )
            regret = [np.min([y for _, y in history])]
            for _ in range(context.T):
                next_x = LLAMAGENT_L_HPT(history, func_desc=context.desc).sample_candidate_points()
                next_y = context.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(np.min([y for _, y in history]))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)


def build_llambo_l_graph():
    return LLAMBOLGraph()
