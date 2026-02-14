import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound
from tqdm import trange

from helper_func import find_max_variance_bound, optimize_acqf_ucb, train_gp
from LLM_agent_HPT import LLAMAGENT_L_HPT

from .base import BaseHPTMethodGraph


class JustifyGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="justify")

    def execute(self, context):
        regrets, histories = [], []
        for _ in trange(context.T_rep, desc="GPJ", disable=not context.verbose):
            history = LLAMAGENT_L_HPT([], func_desc=context.desc).llm_warmstarting(
                num_warmstart=context.T_ini, objective_function=context.obj
            )
            regret = [np.min([y for _, y in history])]
            lower_bounds = context.bounds[0]
            upper_bounds = context.bounds[1]
            X = torch.tensor([x for x, y in history], dtype=torch.float64)
            Y = [y for x, y in history]
            X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
            history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
            model = train_gp(history_gp)
            max_var = find_max_variance_bound(
                model, bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]), dim=context.dim, resolution=10
            )
            for t in range(context.T):
                X = torch.tensor([x for x, y in history], dtype=torch.float64)
                Y = [y for x, y in history]
                X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
                history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
                model = train_gp(history_gp)
                beta_t = np.log((t + 1) * context.dim * np.pi**2 / 0.1 * 6) * 2
                next_x = optimize_acqf_ucb(
                    model, bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]), beta=beta_t
                )

                while True:
                    try:
                        next_x_LLM = LLAMAGENT_L_HPT(history, func_desc=context.desc).sample_candidate_points()
                        break
                    except Exception:
                        print("call llambo_l failed, retrying...")
                        continue

                next_x_LLM_rescaled = (
                    (torch.tensor(next_x_LLM, dtype=torch.float64) - lower_bounds) / (upper_bounds - lower_bounds)
                ).tolist()
                ucb = UpperConfidenceBound(model, beta=beta_t)
                psi_t = max_var / (t + 1)
                if ucb(next_x).item() > ucb(torch.tensor([next_x_LLM_rescaled], dtype=torch.float64)).item() + psi_t:
                    next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds
                    next_y = context.obj(next_x.squeeze(0).tolist())
                    history.append((tuple(next_x.squeeze(0).tolist()), next_y))
                else:
                    next_y = context.obj(next_x_LLM)
                    history.append((tuple(next_x_LLM), next_y))
                regret.append(np.min([y for _, y in history]))

            regrets.append(regret)
            histories.append(history)
        return histories, np.array(regrets)


def build_justify_graph():
    return JustifyGraph()
