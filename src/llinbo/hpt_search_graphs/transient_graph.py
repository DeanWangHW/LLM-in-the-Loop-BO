import numpy as np
import torch

from ..core.helper_func import optimize_acqf_ucb, train_gp
from ..agents.hpt import LLAMAGENT_L_HPT

from .base import BaseHPTMethodGraph, HPTGraphConfig, HPTGraphState


class TransientGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="transient")

    def initialize_history(self, config: HPTGraphConfig):
        return LLAMAGENT_L_HPT([], func_desc=config.desc).llm_warmstarting(
            num_warmstart=config.T_ini,
            objective_function=config.objective,
        )

    @staticmethod
    def _propose_gp(config: HPTGraphConfig, state: HPTGraphState):
        lower_bounds = config.bounds[0]
        upper_bounds = config.bounds[1]
        X = torch.tensor([x for x, _ in state.history], dtype=torch.float64)
        Y = [y for _, y in state.history]
        X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
        history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
        model = train_gp(history_gp)
        beta_t = np.log((state.iter_idx + 1) * config.dim * np.pi**2 / 0.1 * 6) * 2
        next_x = optimize_acqf_ucb(
            model,
            bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]),
            beta=beta_t,
        )
        next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds
        return next_x.squeeze(0).tolist(), {"beta_t": float(beta_t)}

    def propose_candidate(self, config: HPTGraphConfig, state: HPTGraphState):
        p_t = min(state.iter_idx**2 / max(config.T, 1), 1)
        if np.random.rand() < p_t:
            candidate, diagnostics = self._propose_gp(config, state)
            diagnostics["p_t"] = float(p_t)
            return candidate, "gp", diagnostics

        candidate = LLAMAGENT_L_HPT(state.history, func_desc=config.desc).sample_candidate_points()
        return candidate, "llm", {"p_t": float(p_t)}


def build_transient_graph():
    return TransientGraph()
