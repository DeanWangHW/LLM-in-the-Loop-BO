from multiprocessing import Pool, cpu_count

import numpy as np
import torch

from ..core.helper_func import (
    find_gp_maximum,
    optimize_acqf_ucb,
    select_next_design_point_bound,
    train_gp,
)
from ..agents.hpt import LLAMAGENT_L_HPT, build_gp_model

from .base import BaseHPTMethodGraph, HPTGraphConfig, HPTGraphState


class ConstrainedGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="constrained")

    def initialize_history(self, config: HPTGraphConfig):
        return LLAMAGENT_L_HPT([], func_desc=config.desc).llm_warmstarting(
            num_warmstart=config.T_ini,
            objective_function=config.objective,
        )

    def propose_candidate(self, config: HPTGraphConfig, state: HPTGraphState):
        lower_bounds = config.bounds[0]
        upper_bounds = config.bounds[1]
        sraw_new = 10000
        sraw = int(np.floor(sraw_new / (state.iter_idx + 1) ** 2))

        X = torch.tensor([x for x, _ in state.history], dtype=torch.float64)
        Y = [y for _, y in state.history]
        X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
        history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
        model = train_gp(history_gp)

        beta_t = np.log((state.iter_idx + 1) * config.dim * np.pi**2 / 0.1 * 6) * 2
        llm_x = LLAMAGENT_L_HPT(state.history, func_desc=config.desc).sample_candidate_points()
        llm_x_rescaled = ((torch.tensor(llm_x, dtype=torch.float64) - lower_bounds) / (upper_bounds - lower_bounds)).tolist()

        post_max = find_gp_maximum(model, config.bounds, num_restarts=10, raw_samples=100)
        better_samples = []
        if sraw > 1:
            with torch.no_grad():
                posterior = model.posterior(torch.tensor(llm_x_rescaled, dtype=torch.float64).unsqueeze(0))
                samples = posterior.rsample(sample_shape=torch.Size([sraw]))
            for sample in samples.view(-1):
                if sample.item() > post_max:
                    better_samples.append(sample.item())

        diagnostics = {
            "beta_t": float(beta_t),
            "sraw": int(sraw),
            "num_better_samples": int(len(better_samples)),
            "post_max": float(post_max),
        }

        if not better_samples:
            next_x = optimize_acqf_ucb(
                model,
                bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]),
                beta=beta_t,
            )
            next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds
            return next_x.squeeze(0).tolist(), "gp", diagnostics

        args_list = [(llm_x, sample_val, state.history, lower_bounds, upper_bounds) for sample_val in better_samples]
        with Pool(min(cpu_count(), len(args_list))) as pool:
            models = pool.map(build_gp_model, args_list)

        model_dict = {i: model_item for i, model_item in enumerate(models)}
        next_x = select_next_design_point_bound(
            model_dict=model_dict,
            bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]),
            beta_t=beta_t,
            dim=config.dim,
        )
        next_x = (torch.tensor(next_x, dtype=torch.float64) * (upper_bounds - lower_bounds) + lower_bounds).tolist()
        return next_x, "cgp", diagnostics


def build_constrained_graph():
    return ConstrainedGraph()
