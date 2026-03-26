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

from hpt_agent.search_space import parse_search_space, to_hebo_design_space, validate_params


def test_parse_mixed_search_space_and_validate_params():
    payload = {
        "parameters": [
            {"name": "lr", "type": "float", "lb": 1e-4, "ub": 0.1},
            {"name": "depth", "type": "int", "lb": 2, "ub": 8},
            {"name": "use_bias", "type": "bool"},
            {"name": "kernel", "type": "categorical", "choices": ["linear", "rbf"]},
        ]
    }

    specs = parse_search_space(payload)

    assert len(specs) == 4
    assert specs[2]["choices"] == [False, True]
    assert specs[3]["choices"] == ["linear", "rbf"]

    validate_params(
        {"lr": 0.01, "depth": 4, "use_bias": True, "kernel": "rbf"},
        specs,
    )


def test_to_hebo_design_space_maps_bool_to_cat():
    payload = {
        "parameters": [
            {"name": "flag", "type": "bool"},
            {"name": "x", "type": "float", "lb": -1.0, "ub": 1.0},
        ]
    }

    specs = parse_search_space(payload)
    hebo_space = to_hebo_design_space(specs)

    flag = next(item for item in hebo_space if item["name"] == "flag")
    assert flag["type"] == "cat"
    assert flag["categories"] == [False, True]
