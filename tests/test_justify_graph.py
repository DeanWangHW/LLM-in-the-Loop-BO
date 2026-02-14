import torch

from hpt_search_graphs.justify_graph import build_justify_graph


class FakeAgent:
    def __init__(self, history, func_desc=None):
        self.history = history

    def llm_warmstarting(self, num_warmstart, objective_function):
        return [((0.1, 0.2), objective_function([0.1, 0.2]))]

    def sample_candidate_points(self):
        return [0.4, 0.4]


class FakeUCB:
    def __init__(self, model, beta):
        pass

    def __call__(self, x):
        return torch.tensor(0.0, dtype=torch.float64)


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


def test_justify_graph(monkeypatch):
    monkeypatch.setattr("hpt_search_graphs.justify_graph.LLAMAGENT_L_HPT", FakeAgent)
    monkeypatch.setattr("hpt_search_graphs.justify_graph.train_gp", lambda history_gp: object())
    monkeypatch.setattr("hpt_search_graphs.justify_graph.find_max_variance_bound", lambda model, bounds, dim, resolution: 0.0)
    monkeypatch.setattr(
        "hpt_search_graphs.justify_graph.optimize_acqf_ucb",
        lambda model, bounds, beta: torch.tensor([[0.3, 0.3]], dtype=torch.float64),
    )
    monkeypatch.setattr("hpt_search_graphs.justify_graph.UpperConfidenceBound", FakeUCB)
    graph = build_justify_graph()
    histories, regrets = graph.run(Ctx())
    assert len(histories[0]) == 2
    assert regrets.shape == (1, 2)
