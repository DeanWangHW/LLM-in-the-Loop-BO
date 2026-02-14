import numpy as np
import torch
from tqdm import trange

from helper_func import generate_ini_data, optimize_acqf_ucb, train_gp

from .base import build_single_node_graph


def run_bo(context):
    regrets, histories = [], []
    for _ in trange(context.T_rep, desc="BO", disable=not context.verbose):
        history = generate_ini_data(func=context.obj, n=context.T_ini, bounds=context.bounds)
        regret = [np.min([y for _, y in history])]
        for i in range(context.T):
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
            regret.append(np.min([y for _, y in history]))

        regrets.append(regret)
        histories.append(history)
    return histories, np.array(regrets)


def build_bo_graph():
    return build_single_node_graph(name="bo", run_fn=run_bo)
