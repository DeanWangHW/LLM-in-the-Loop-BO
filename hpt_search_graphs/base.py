from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class SearchGraph:
    """Simple executable graph wrapper for one search strategy."""

    name: str
    run_fn: Callable[[Any], Any]

    def run(self, context: Any):
        return self.run_fn(context)
