# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the weight-enumeration tests."""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .counter import CSSLogicalWeightCounter


def _native(function: Callable[..., Any], interpreted: bool) -> Any:
    """Return a jitted function, or its Python implementation to trace and mock-check logic."""
    return getattr(function, "py_func", function) if interpreted else function


def _get_sector_bases(
    checks_x: npt.ArrayLike, checks_z: npt.ArrayLike, **kwargs: Any
) -> tuple[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]], ...]:
    """Coefficient bases and quotient labels of the X and Z sectors of a CSS code."""
    return CSSLogicalWeightCounter(checks_x, checks_z, **kwargs)._get_sector_bases()
