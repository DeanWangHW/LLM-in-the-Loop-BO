import pytest

pytest.importorskip("torch")

import torch

from hpt_search_graphs.rs_graph import build_rs_graph


class Ctx:
    T_rep = 1
    T_ini = 1
    T = 1
    dim = 2
    verbose = False
    bounds = (torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

    @staticmethod
    def obj(x):
        return float(sum(x))


def test_rs_graph(monkeypatch):
    monkeypatch.setattr("hpt_search_graphs.rs_graph.generate_ini_data", lambda func, n, bounds: [((0.1, 0.2), func([0.1, 0.2]))])
    graph = build_rs_graph()
    histories, regrets = graph.run(Ctx())
    assert len(histories) == 1
    assert len(histories[0]) == 2
    assert regrets.shape == (1, 2)
