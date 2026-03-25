import pytest

pytest.importorskip("torch")
import torch

from hpt_search_graphs.llambo_l_graph import build_llambo_l_graph


class FakeAgent:
    def __init__(self, history, func_desc=None):
        self.history = history

    def llm_warmstarting(self, num_warmstart, objective_function):
        return [((0.1, 0.2), objective_function([0.1, 0.2]))]

    def sample_candidate_points(self):
        return [0.3, 0.4]


class Ctx:
    T_rep = 1
    T_ini = 1
    T = 1
    dim = 2
    verbose = False
    desc = {}
    bounds = (torch.tensor([0.0, 0.0], dtype=torch.float64), torch.tensor([1.0, 1.0], dtype=torch.float64))

    @staticmethod
    def obj(x):
        return float(sum(x))


def test_llambo_l_graph(monkeypatch):
    monkeypatch.setattr("hpt_search_graphs.llambo_l_graph.LLAMAGENT_L_HPT", FakeAgent)
    graph = build_llambo_l_graph()
    histories, regrets = graph.run(Ctx())
    assert len(histories[0]) == 2
    assert regrets.shape == (1, 2)
