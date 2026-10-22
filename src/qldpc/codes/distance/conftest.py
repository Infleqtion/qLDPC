# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for exact-distance, binary-algebra, and BZ tests."""

from collections.abc import Iterator
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

from . import _binary as binary


@pytest.fixture
def force_information_sets() -> Iterator[None]:
    """Route Brouwer-Zimmermann searches through information sets instead of exhaustive search."""
    with (
        mock.patch.multiple(
            "qldpc.codes.distance._brouwer_zimmermann",
            exhaustive_is_cheaper=mock.Mock(return_value=False),
            _brute_force_is_cheaper=mock.Mock(return_value=False),
        ),
        mock.patch("qldpc.codes.distance.core.exhaustive_is_cheaper", return_value=False),
    ):
        yield


def get_random_full_rank_matrix(
    rng: np.random.Generator, rows: int, cols: int
) -> npt.NDArray[np.uint8]:
    """Return a random full-row-rank binary matrix with the requested row and column counts."""
    while True:
        matrix = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)
        if len(binary.get_independent_rows(matrix)) == rows:
            return matrix


def count_set_bits(val: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    """Return per-word bit counts, computed bit by bit as an independent reference for popcounts."""
    val = np.asarray(val)
    nbits = 8 * val.itemsize
    out = np.empty_like(val)
    for indices in np.ndindex(val.shape):
        ival = int(val[indices])
        out[indices] = sum(ival >> i & 1 for i in range(nbits))

    return out
