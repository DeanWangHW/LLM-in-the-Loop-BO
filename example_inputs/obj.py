from __future__ import annotations

import math
from typing import Any, Dict


def objective(params: Dict[str, Any]) -> float:
    """Mixed-space toy objective for frontend upload testing.

    Parameter schema (must match search_space.json):
    - lr: float
    - num_layers: int
    - use_dropout: bool
    - activation: categorical {"relu", "gelu", "tanh"}
    - batch_mode: categorical {"small", "medium", "large"}
    """

    lr = float(params["lr"])
    num_layers = int(params["num_layers"])
    use_dropout = bool(params["use_dropout"])
    activation = str(params["activation"])
    batch_mode = str(params["batch_mode"])

    activation_bias = {
        "relu": 0.03,
        "gelu": 0.0,
        "tanh": 0.08,
    }
    batch_bias = {
        "small": 0.06,
        "medium": 0.0,
        "large": 0.03,
    }
    if activation not in activation_bias:
        raise ValueError(f"Unknown activation: {activation}")
    if batch_mode not in batch_bias:
        raise ValueError(f"Unknown batch_mode: {batch_mode}")

    # Numerical basin centered around lr ~= 5e-3 and num_layers ~= 4.
    loss = (math.log10(lr) + 2.3) ** 2 + 0.04 * (num_layers - 4) ** 2
    loss += activation_bias[activation]
    loss += batch_bias[batch_mode]
    loss += -0.12 if use_dropout else 0.05

    # Cross-term makes the landscape non-separable across mixed dimensions.
    if activation == "gelu" and use_dropout and num_layers >= 4:
        loss -= 0.05

    return float(loss)
