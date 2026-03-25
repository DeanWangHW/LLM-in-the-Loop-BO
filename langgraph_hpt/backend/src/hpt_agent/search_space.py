from __future__ import annotations

import math
import random
from typing import Any, Dict, List


SUPPORTED_TYPES = {"float", "int", "bool", "categorical"}


def parse_search_space(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Search space payload must be a dictionary.")
    parameters = payload.get("parameters")
    if not isinstance(parameters, list):
        raise ValueError("Search space must contain a list field 'parameters'.")

    seen = set()
    specs: List[Dict[str, Any]] = []
    for raw in parameters:
        if not isinstance(raw, dict):
            raise ValueError("Each parameter spec must be a dictionary.")
        name = raw.get("name")
        ptype = str(raw.get("type", "")).lower()
        if not name or not isinstance(name, str):
            raise ValueError("Parameter 'name' must be a non-empty string.")
        if name in seen:
            raise ValueError(f"Duplicate parameter name: {name}")
        seen.add(name)
        if ptype not in SUPPORTED_TYPES:
            raise ValueError(f"Unsupported parameter type '{ptype}' for '{name}'.")

        spec: Dict[str, Any] = {"name": name, "type": ptype}
        if ptype == "float":
            lb = float(raw["lb"])
            ub = float(raw["ub"])
            if not math.isfinite(lb) or not math.isfinite(ub) or lb >= ub:
                raise ValueError(f"Invalid float bounds for '{name}'.")
            spec["lb"] = lb
            spec["ub"] = ub
        elif ptype == "int":
            lb = int(raw["lb"])
            ub = int(raw["ub"])
            if lb >= ub:
                raise ValueError(f"Invalid int bounds for '{name}'.")
            spec["lb"] = lb
            spec["ub"] = ub
        elif ptype == "bool":
            spec["choices"] = [False, True]
        elif ptype == "categorical":
            choices = raw.get("choices")
            if not isinstance(choices, list) or len(choices) == 0:
                raise ValueError(f"Categorical parameter '{name}' must define non-empty choices.")
            spec["choices"] = list(choices)

        specs.append(spec)

    return specs


def normalize_params(params: Dict[str, Any], specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(params, dict):
        raise ValueError("Params must be a dictionary.")
    spec_map = {item["name"]: item for item in specs}

    missing = [name for name in spec_map if name not in params]
    if missing:
        raise ValueError(f"Missing parameters: {missing}")
    extra = [name for name in params if name not in spec_map]
    if extra:
        raise ValueError(f"Unexpected parameters: {extra}")

    normalized: Dict[str, Any] = {}
    for name, spec in spec_map.items():
        value = params[name]
        ptype = spec["type"]
        if ptype == "float":
            val = float(value)
            if not (spec["lb"] <= val <= spec["ub"]):
                raise ValueError(f"Float parameter '{name}' out of bounds.")
            normalized[name] = val
        elif ptype == "int":
            if isinstance(value, bool):
                raise ValueError(f"Int parameter '{name}' cannot be bool.")
            val = int(value)
            if not (spec["lb"] <= val <= spec["ub"]):
                raise ValueError(f"Int parameter '{name}' out of bounds.")
            normalized[name] = val
        elif ptype == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"Bool parameter '{name}' must be bool.")
            normalized[name] = value
        elif ptype == "categorical":
            if value not in spec["choices"]:
                raise ValueError(f"Categorical parameter '{name}' not in choices.")
            normalized[name] = value
        else:  # pragma: no cover
            raise ValueError(f"Unsupported parameter type '{ptype}'.")

    return normalized


def validate_params(params: Dict[str, Any], specs: List[Dict[str, Any]]) -> None:
    normalize_params(params, specs)


def sample_random_params(specs: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    sampled: Dict[str, Any] = {}
    for spec in specs:
        ptype = spec["type"]
        name = spec["name"]
        if ptype == "float":
            sampled[name] = rng.uniform(spec["lb"], spec["ub"])
        elif ptype == "int":
            sampled[name] = rng.randint(spec["lb"], spec["ub"])
        elif ptype == "bool":
            sampled[name] = rng.choice([False, True])
        elif ptype == "categorical":
            sampled[name] = rng.choice(spec["choices"])
        else:  # pragma: no cover
            raise ValueError(f"Unsupported parameter type '{ptype}'.")
    return sampled


def to_hebo_design_space(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    mapped: List[Dict[str, Any]] = []
    for spec in specs:
        ptype = spec["type"]
        name = spec["name"]
        if ptype == "float":
            mapped.append({"name": name, "type": "num", "lb": spec["lb"], "ub": spec["ub"]})
        elif ptype == "int":
            mapped.append({"name": name, "type": "int", "lb": spec["lb"], "ub": spec["ub"]})
        elif ptype in {"bool", "categorical"}:
            mapped.append({"name": name, "type": "cat", "categories": list(spec["choices"])})
        else:  # pragma: no cover
            raise ValueError(f"Unsupported parameter type '{ptype}'.")
    return mapped

