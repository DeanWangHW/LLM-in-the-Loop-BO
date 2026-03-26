import pytest

pytest.importorskip("torch")

import torch

from hpt_search_graphs.bo_graph import build_bo_graph


class Ctx:
    T_rep = 1
    T_ini = 1
    T = 1
    dim = 2
    verbose = False
    bounds = (torch.tensor([0.0, 0.0], dtype=torch.float64), torch.tensor([1.0, 1.0], dtype=torch.float64))

    @staticmethod
    def obj(x):
        return float(torch.as_tensor(x, dtype=torch.float64).sum().item())


def test_bo_graph(monkeypatch):
    monkeypatch.setattr("hpt_search_graphs.bo_graph.generate_ini_data", lambda func, n, bounds: [((0.1, 0.2), func([0.1, 0.2]))])
    monkeypatch.setattr("hpt_search_graphs.bo_graph.train_gp", lambda history_gp: object())
    monkeypatch.setattr(
        "hpt_search_graphs.bo_graph.optimize_acqf_ucb",
        lambda model, bounds, beta: torch.tensor([[0.3, 0.4]], dtype=torch.float64),
    )
    graph = build_bo_graph()
    histories, regrets = graph.run(Ctx())
    assert len(histories[0]) == 2
    assert regrets.shape == (1, 2)

