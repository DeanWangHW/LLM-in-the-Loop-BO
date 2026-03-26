import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BACKEND_SRC_CANDIDATES = (
    ROOT / "apps" / "langgraph_backend" / "src",
    ROOT / "langgraph_hpt" / "backend" / "src",
)
BACKEND_SRC = next(
    (path for path in BACKEND_SRC_CANDIDATES if path.exists()),
    BACKEND_SRC_CANDIDATES[0],
)
if str(BACKEND_SRC) not in sys.path:
    sys.path.insert(0, str(BACKEND_SRC))


def test_rs_warmstart_does_not_import_torch(monkeypatch):
    import hpt_agent.optimizers.legacy_adapter as legacy_adapter

    real_import_module = legacy_adapter.importlib.import_module
    import_attempts = []

    def guarded_import(name, *args, **kwargs):
        if name == "torch":
            import_attempts.append(name)
            raise AssertionError("torch import should not happen in rs mode")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(legacy_adapter.importlib, "import_module", guarded_import)

    adapter = legacy_adapter.LegacyAdapter(
        method="rs",
        search_space=[{"name": "x", "type": "float", "lb": -1.0, "ub": 1.0}],
        desc=None,
        seed=0,
        max_retries=1,
    )
    history = adapter.warmstart(2, lambda params: float(params["x"] ** 2))
    candidate, source, diagnostics = adapter.suggest(history, iter_idx=0)

    assert len(history) == 2
    assert source == "random"
    assert "x" in candidate
    assert diagnostics["iter_idx"] == 0
    assert import_attempts == []
