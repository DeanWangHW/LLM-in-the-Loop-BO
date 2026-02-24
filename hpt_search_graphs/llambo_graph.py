import numpy as np
from tqdm import trange

from LLM_agent_HPT import LLAMAGENT_HPT

from .base import BaseHPTMethodGraph


class LLAMBOGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="llambo")

    def execute(self, context):
        regrets, histories = [], []
        for _ in trange(context.T_rep, desc="LLAMBO", disable=not context.verbose):
            history = LLAMAGENT_HPT([], func_desc=context.desc).llm_warmstarting(
                num_warmstart=context.T_ini, objective_function=context.obj
            )
            regret = [np.min([y for _, y in history])]
            for t in range(context.T):
                while True:
                    try:
                        next_x = LLAMAGENT_HPT(history, func_desc=context.desc).find_best_candidate()
                        break
                    except Exception as e:
                        print(f"Retrying at iteration {t} due to error: {e}")
                        continue
                next_y = context.obj(next_x)
                history.append((tuple(next_x), next_y))
                regret.append(np.min([y for _, y in history]))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)


def build_llambo_graph():
    return LLAMBOGraph()
