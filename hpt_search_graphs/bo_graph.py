import numpy as np
import torch

from helper_func import generate_ini_data, optimize_acqf_ucb, train_gp

from .base import BaseHPTMethodGraph, HPTGraphConfig, HPTGraphState


class BOGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="bo")

    def initialize_history(self, config: HPTGraphConfig):
        return generate_ini_data(func=config.objective, n=config.T_ini, bounds=config.bounds)

    def propose_candidate(self, config: HPTGraphConfig, state: HPTGraphState):
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
        candidate = next_x.squeeze(0).tolist()
        return candidate, "gp", {"beta_t": float(beta_t)}


def build_bo_graph():
    return BOGraph()
