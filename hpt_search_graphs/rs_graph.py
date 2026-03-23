import torch

from helper_func import generate_ini_data

from .base import BaseHPTMethodGraph, HPTGraphConfig, HPTGraphState


class RSGraph(BaseHPTMethodGraph):
    def __init__(self):
        super().__init__(name="rs")

    def initialize_history(self, config: HPTGraphConfig):
        return generate_ini_data(func=config.objective, n=config.T_ini, bounds=config.bounds)

    def propose_candidate(self, config: HPTGraphConfig, state: HPTGraphState):
        x = torch.rand(config.dim)
        x = config.bounds[0] + (config.bounds[1] - config.bounds[0]) * x
        return x.tolist(), "random", {}


def build_rs_graph():
    return RSGraph()
