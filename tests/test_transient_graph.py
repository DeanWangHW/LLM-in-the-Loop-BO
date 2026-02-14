from hpt_search_graphs.transient_graph import build_transient_graph


class FakeAgent:
    def __init__(self, history, func_desc=None):
        self.history = history

    def llm_warmstarting(self, num_warmstart, objective_function):
        return [((0.1, 0.2), objective_function([0.1, 0.2]))]

    def sample_candidate_points(self):
        return [0.2, 0.1]


class Ctx:
    T_rep = 1
    T_ini = 1
    T = 1
    dim = 2
    verbose = False
    desc = {}
    bounds = None

    @staticmethod
    def obj(x):
        return float(sum(x))


def test_transient_graph(monkeypatch):
    monkeypatch.setattr("hpt_search_graphs.transient_graph.LLAMAGENT_L_HPT", FakeAgent)
    monkeypatch.setattr("hpt_search_graphs.transient_graph.np.random.rand", lambda: 1.0)
    graph = build_transient_graph()
    histories, regrets = graph.run(Ctx())
    assert len(histories[0]) == 2
    assert regrets.shape == (1, 2)
