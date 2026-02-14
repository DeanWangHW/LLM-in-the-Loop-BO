import numpy as np
import torch
from tqdm import trange

from helper_func import generate_ini_data

from .base import SearchGraph


def run_rs(context):
    regrets, histories = [], []
    for _ in trange(context.T_rep, desc="RANDOM", disable=not context.verbose):
        history = generate_ini_data(func=context.obj, n=context.T_ini, bounds=context.bounds)
        regret = [np.min([y for _, y in history])]
        for _ in range(context.T):
            x = torch.rand(context.dim)
            x = context.bounds[0] + (context.bounds[1] - context.bounds[0]) * x
            x = x.tolist()
            y = context.obj(x)
            history.append((tuple(x), y))
            regret.append(np.min([y for _, y in history]))

        regrets.append(regret)
        histories.append(history)
    return histories, np.array(regrets)


def build_rs_graph():
    return SearchGraph(name="rs", run_fn=run_rs)
