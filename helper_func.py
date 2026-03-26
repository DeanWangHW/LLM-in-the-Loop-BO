"""Compatibility shim for legacy import path.

Use `llinbo.core.helper_func` for new code.
"""

from __future__ import annotations

import pathlib
import sys

_SRC_ROOT = pathlib.Path(__file__).resolve().parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from llinbo.core.helper_func import *  # noqa: F401,F403

