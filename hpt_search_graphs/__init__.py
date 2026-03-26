"""Compatibility shim package for legacy import path.

Use `llinbo.hpt_search_graphs` for new code.
"""

from __future__ import annotations

import pathlib
import sys

_SRC_ROOT = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from llinbo.hpt_search_graphs import *  # noqa: F401,F403

