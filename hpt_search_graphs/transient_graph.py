import numpy as np
import torch
from tqdm import trange

from helper_func import optimize_acqf_ucb, train_gp
from LLM_agent_HPT import LLAMAGENT_L_HPT

from .base import BaseHPTMethodGraph


class TransientGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="transient")

    def execute(self, context):
        regrets, histories = [], []
        for _ in trange(context.T_rep, desc="TRANSIENT", disable=not context.verbose):
            history = LLAMAGENT_L_HPT([], func_desc=context.desc).llm_warmstarting(
                num_warmstart=context.T_ini, objective_function=context.obj
            )
            regret = [np.min([y for _, y in history])]
            for i in range(context.T):
                p_t = min(i**2 / context.T, 1)
                if np.random.rand() < p_t:
                    lower_bounds = context.bounds[0]
                    upper_bounds = context.bounds[1]
                    X = torch.tensor([x for x, y in history], dtype=torch.float64)
                    Y = [y for x, y in history]
                    X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
                    history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
                    model = train_gp(history_gp)
                    beta_t = np.log((i + 1) * context.dim * np.pi**2 / 0.1 * 6) * 2
                    next_x = optimize_acqf_ucb(
                        model, bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]), beta=beta_t
                    )
                    next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds
                    next_y = context.obj(next_x.squeeze(0))
                    history.append((tuple(next_x.squeeze(0).tolist()), next_y))
                else:
                    while True:
                        try:
                            next_x = LLAMAGENT_L_HPT(history, func_desc=context.desc).sample_candidate_points()
                            break
                        except Exception:
                            print("call llambo failed, retrying...")
                            continue
                    next_y = context.obj(next_x)
                    history.append((tuple(next_x), next_y))

                regret.append(np.min([y for _, y in history]))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)


def build_transient_graph():
    return TransientGraph()
