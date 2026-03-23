import numpy as np
import torch
from botorch.acquisition import UpperConfidenceBound

from helper_func import find_max_variance_bound, optimize_acqf_ucb, train_gp
from LLM_agent_HPT import LLAMAGENT_L_HPT

from .base import BaseHPTMethodGraph, HPTGraphConfig, HPTGraphState


class JustifyGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="justify")

    def initialize_history(self, config: HPTGraphConfig):
        return LLAMAGENT_L_HPT([], func_desc=config.desc).llm_warmstarting(
            num_warmstart=config.T_ini,
            objective_function=config.objective,
        )

    def propose_candidate(self, config: HPTGraphConfig, state: HPTGraphState):
        lower_bounds = config.bounds[0]
        upper_bounds = config.bounds[1]
        X = torch.tensor([x for x, _ in state.history], dtype=torch.float64)
        Y = [y for _, y in state.history]
        X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
        history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
        model = train_gp(history_gp)

        beta_t = np.log((state.iter_idx + 1) * config.dim * np.pi**2 / 0.1 * 6) * 2
        gp_x = optimize_acqf_ucb(
            model,
            bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]),
            beta=beta_t,
        )

        llm_x = LLAMAGENT_L_HPT(state.history, func_desc=config.desc).sample_candidate_points()
        llm_x_rescaled = ((torch.tensor(llm_x, dtype=torch.float64) - lower_bounds) / (upper_bounds - lower_bounds)).tolist()

        max_var = state.diagnostics.get("max_var")
        if max_var is None:
            max_var = find_max_variance_bound(
                model,
                bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]),
                dim=config.dim,
                resolution=10,
            )

        ucb = UpperConfidenceBound(model, beta=beta_t)
        psi_t = max_var / (state.iter_idx + 1)
        gp_ucb = ucb(gp_x).item()
        llm_ucb = ucb(torch.tensor([llm_x_rescaled], dtype=torch.float64)).item()

        diagnostics = {
            "beta_t": float(beta_t),
            "psi_t": float(psi_t),
            "max_var": float(max_var),
            "gp_ucb": float(gp_ucb),
            "llm_ucb": float(llm_ucb),
        }

        if gp_ucb > llm_ucb + psi_t:
            gp_x = gp_x * (upper_bounds - lower_bounds) + lower_bounds
            return gp_x.squeeze(0).tolist(), "gp", diagnostics
        return llm_x, "llm", diagnostics


def build_justify_graph():
    return JustifyGraph()
