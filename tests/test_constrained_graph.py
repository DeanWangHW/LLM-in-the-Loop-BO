import pytest

pytest.importorskip("torch")

import torch

from hpt_search_graphs.constrained_graph import build_constrained_graph


class FakePosterior:
    def rsample(self, sample_shape):
        return torch.zeros(sample_shape[0], 1, dtype=torch.float64)


class FakeModel:
    def posterior(self, x):
        return FakePosterior()


class FakeAgent:
    def __init__(self, history, func_desc=None):
        self.history = history

    def llm_warmstarting(self, num_warmstart, objective_function):
        return [((0.1, 0.2), objective_function([0.1, 0.2]))]

    def sample_candidate_points(self):
        return [0.4, 0.4]


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


def test_constrained_graph(monkeypatch):
    monkeypatch.setattr("hpt_search_graphs.constrained_graph.LLAMAGENT_L_HPT", FakeAgent)
    monkeypatch.setattr("hpt_search_graphs.constrained_graph.train_gp", lambda history_gp: FakeModel())
    monkeypatch.setattr("hpt_search_graphs.constrained_graph.find_gp_maximum", lambda model, bounds, num_restarts, raw_samples: 1.0)
    monkeypatch.setattr(
        "hpt_search_graphs.constrained_graph.optimize_acqf_ucb",
        lambda model, bounds, beta: torch.tensor([[0.2, 0.2]], dtype=torch.float64),
    )
    graph = build_constrained_graph()
    histories, regrets = graph.run(Ctx())
    assert len(histories[0]) == 2
    assert regrets.shape == (1, 2)
