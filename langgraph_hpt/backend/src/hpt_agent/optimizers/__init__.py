from .base import OptimizerAdapter
from .hebo_adapter import HEBOAdapter

__all__ = ["OptimizerAdapter", "HEBOAdapter", "LegacyAdapter"]


def __getattr__(name: str):
    if name == "LegacyAdapter":
        from .legacy_adapter import LegacyAdapter

        return LegacyAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
