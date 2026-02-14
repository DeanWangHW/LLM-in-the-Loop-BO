import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from tqdm import trange

from helper_func import find_gp_maximum, optimize_acqf_ucb, select_next_design_point_bound, train_gp
from LLM_agent_HPT import LLAMAGENT_L_HPT, build_gp_model

from .base import SearchGraph


def run_constrained(context):
    regrets, histories = [], []
    lower_bounds = context.bounds[0]
    upper_bounds = context.bounds[1]

    for _ in trange(context.T_rep, desc="CONSTRAINED", disable=not context.verbose):
        sraw_new = 10000
        history = LLAMAGENT_L_HPT([], func_desc=context.desc).llm_warmstarting(
            num_warmstart=context.T_ini, objective_function=context.obj
        )
        regret = [np.min([y for _, y in history])]
        for t in range(context.T):
            sraw = int(np.floor(sraw_new / (t + 1) ** 2))
            X = torch.tensor([x for x, y in history], dtype=torch.float64)
            Y = [y for x, y in history]
            X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)
            history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]
            model = train_gp(history_gp)
            beta_t = np.log((t + 1) * context.dim * np.pi**2 / 0.1 * 6) * 2

            while True:
                try:
                    next_x_LLM = LLAMAGENT_L_HPT(history, func_desc=context.desc).sample_candidate_points()
                    break
                except Exception:
                    print("call LLAMBO-L failed, retrying...")
                    continue

            next_x_LLM_rescaled = (
                (torch.tensor(next_x_LLM, dtype=torch.float64) - lower_bounds) / (upper_bounds - lower_bounds)
            ).tolist()
            better_samples = []
            post_max = find_gp_maximum(model, context.bounds, num_restarts=10, raw_samples=100)
            if sraw > 1:
                with torch.no_grad():
                    posterior = model.posterior(torch.tensor(next_x_LLM_rescaled, dtype=torch.float64).unsqueeze(0))
                    samples = posterior.rsample(sample_shape=torch.Size([sraw]))
                for s in samples.view(-1):
                    if s.item() > post_max:
                        better_samples.append(s.item())

            if len(better_samples) == 0:
                next_x = optimize_acqf_ucb(
                    model, bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]), beta=beta_t
                )
                next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds
                next_y = context.obj(next_x.squeeze(0).tolist())
                history.append((tuple(next_x.squeeze(0).tolist()), next_y))
            else:
                args_list = [(next_x_LLM, sample_val, history, lower_bounds, upper_bounds) for sample_val in better_samples]
                with Pool(min(cpu_count(), len(args_list))) as pool:
                    models = pool.map(build_gp_model, args_list)

                model_dict = {i: model for i, model in enumerate(models)}
                next_x = select_next_design_point_bound(
                    model_dict=model_dict,
                    bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]),
                    beta_t=beta_t,
                    dim=context.dim,
                )
                next_x = ((torch.tensor(next_x, dtype=torch.float64)) * (upper_bounds - lower_bounds) + lower_bounds).tolist()
                next_y = context.obj(next_x)
                history.append((tuple(next_x), next_y))
            regret.append(np.min([y for _, y in history]))

        regrets.append(regret)
        histories.append(history)
    return histories, np.array(regrets)


def build_constrained_graph():
    return SearchGraph(name="constrained", run_fn=run_constrained)
