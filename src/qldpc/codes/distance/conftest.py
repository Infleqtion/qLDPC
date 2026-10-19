# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for exact-distance dispatch, BZ and weight-enumeration tests."""

from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

from . import _binary as binary
from .weight_enumerators import CSSLogicalWeightCounter


@pytest.fixture
def force_information_sets() -> Iterator[None]:
    """Route Brouwer-Zimmermann searches through information sets instead of exhaustive search."""
    with (
        mock.patch.multiple(
            "qldpc.codes.distance.brouwer_zimmermann",
            _exhaustive_is_cheaper=mock.Mock(return_value=False),
            _brute_force_is_cheaper=mock.Mock(return_value=False),
        ),
        mock.patch("qldpc.codes.distance.core._exhaustive_is_cheaper", return_value=False),
    ):
        yield


def _get_random_full_rank_matrix(
    rng: np.random.Generator, rows: int, cols: int
) -> npt.NDArray[np.uint8]:
    """Return a random full-row-rank binary matrix with the requested row and column counts."""
    while True:
        matrix = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)
        if len(binary._get_independent_rows(matrix)) == rows:
            return matrix


def _bitwise_count(
    val: npt.ArrayLike, out: npt.NDArray[np.uint64] | None = None
) -> npt.NDArray[np.uint64]:
    """Return reference per-word bit counts for arrays accepted by optimized popcount helpers."""
    val = np.asarray(val)
    nbits = 8 * val.itemsize

    if out is None:
        out = np.empty_like(val)

    for indices in np.ndindex(val.shape):
        ival = int(val[indices])
        out[indices] = sum(ival >> i & 1 for i in range(nbits))

    return out


def _native(function: Callable[..., Any], interpreted: bool) -> Any:
    """Return a jitted function, or its Python implementation to trace and mock-check logic."""
    return getattr(function, "py_func", function) if interpreted else function


def _get_sector_bases(
    checks_x: npt.ArrayLike, checks_z: npt.ArrayLike, **kwargs: Any
) -> tuple[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]], ...]:
    """Coefficient bases and quotient labels of the X and Z sectors of a CSS code."""
    return CSSLogicalWeightCounter(checks_x, checks_z, **kwargs)._get_sector_bases()
