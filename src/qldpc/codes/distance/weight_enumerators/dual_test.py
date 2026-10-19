# SPDX-License-Identifier: Apache-2.0

"""Tests for stabilizer enumeration and the exact MacWilliams transform."""

from __future__ import annotations

import itertools
import math
from unittest import mock

import numpy as np
import pytest

from . import dual
from .conftest import _native
from .supports import pack


def _histogram(rows: np.ndarray) -> np.ndarray:
    """Weights of every word in the row space, computed by explicit span enumeration."""
    counts = np.zeros(rows.shape[1] + 1, dtype=np.int64)
    for coefficients in itertools.product((0, 1), repeat=len(rows)):
        word = np.bitwise_xor.reduce(rows[np.flatnonzero(coefficients)], axis=0, initial=0)
        counts[int(np.count_nonzero(word))] += 1
    return counts


@pytest.mark.parametrize("batch_size", [1, 3, 100])
def test_stabilizer_histograms_against_span(batch_size: int) -> None:
    x = np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=np.uint8)
    z = np.array([[1, 1, 1, 1, 1]], dtype=np.uint8)
    histogram_x, histogram_z = dual.get_stabilizer_histograms(
        pack(x), pack(z), 5, symmetric=False, batch_size=batch_size
    )
    np.testing.assert_array_equal(histogram_x, _histogram(x))
    np.testing.assert_array_equal(histogram_z, _histogram(z))
    symmetric_x, symmetric_z = dual.get_stabilizer_histograms(
        pack(x), pack(x), 5, symmetric=True, batch_size=batch_size
    )
    assert symmetric_x is symmetric_z
    np.testing.assert_array_equal(symmetric_x, _histogram(x))


def test_stabilizer_rank_limit() -> None:
    wide = pack(np.eye(63, 64, dtype=np.uint8))
    with pytest.raises(OverflowError, match="62"):
        dual.get_stabilizer_histograms(wide, wide[:0], 64, symmetric=False, batch_size=1)


def test_spectrum_and_minimum_of_a_known_code() -> None:
    """The [[4,2,2]] code, whose logical spectra are known directly."""
    checks = np.ones((1, 4), dtype=np.uint8)
    histogram = _histogram(checks)
    counts_x, counts_z = dual.get_macwilliams_spectrum(histogram, histogram, 1, 1, 4, 4)
    assert counts_x == counts_z == [0, 0, 6, 0, 0]
    assert dual.get_macwilliams_minimum(histogram, histogram, 1, 1, 4) == (2, 6, 6)
    assert dual.get_macwilliams_spectrum(histogram, histogram, 1, 1, 4, 0) == ([0], [0])


def test_exact_transform_guards_and_fallback() -> None:
    assert dual._get_safe_coefficients(100, 40).shape[0] < 50
    assert dual._get_safe_coefficients(1000, 0).shape[0] < 10
    for counts, coefficients in ((np.array([1]), (1,)), (np.array([2]), (-1,))):
        with pytest.raises(ArithmeticError, match="MacWilliams"):
            dual._dual_coefficient(counts, coefficients, 1)

    # Without a certified native prefix, the exact Python-integer path must agree with it.
    histogram = _histogram(np.ones((1, 4), dtype=np.uint8))
    with mock.patch.object(dual, "_minimum_native", return_value=(-1, 0, 0)):
        assert dual.get_macwilliams_minimum(histogram, histogram, 1, 1, 4) == (2, 6, 6)
    zeros = np.zeros(5, dtype=np.int64)
    with (
        mock.patch.object(dual, "_minimum_native", return_value=(-1, 0, 0)),
        pytest.raises(ArithmeticError, match="positive-dimensional"),
    ):
        dual.get_macwilliams_minimum(zeros, zeros, 1, 1, 4)


def test_negative_multiplicities_are_errors() -> None:
    histogram = _histogram(np.ones((1, 4), dtype=np.uint8))
    with (
        mock.patch.object(dual, "_dual_coefficient", return_value=0),
        pytest.raises(ArithmeticError, match="Negative"),
    ):
        dual.get_macwilliams_spectrum(histogram, histogram, 1, 1, 4, 4)
    zeros = np.zeros(5, dtype=np.int64)
    identity_only = np.array([1, 0, 0, 0, 0])
    with (
        mock.patch.object(dual, "_minimum_native", return_value=(-1, 0, 0)),
        pytest.raises(ArithmeticError, match="Negative"),
    ):
        dual.get_macwilliams_minimum(identity_only, zeros, 0, 0, 4)


@pytest.mark.parametrize("interpreted", [False, True])
def test_native_corner_cases(interpreted: bool) -> None:
    minimum = _native(dual._minimum_native, interpreted)
    with pytest.raises(ArithmeticError, match="Nonintegral"):
        minimum(np.array([1]), np.array([1]), 1, 1, np.array([[1]]))
    with pytest.raises(ArithmeticError, match="Negative"):
        minimum(np.array([2]), np.array([0]), 0, 0, np.array([[1]]))
    assert minimum(np.array([0]), np.array([0]), 0, 0, np.array([[1]])) == (-1, 0, 0)
    known = _histogram(np.ones((1, 4), dtype=np.uint8))
    assert minimum(known, known, 1, 1, dual._get_safe_coefficients(4, 1)) == (2, 6, 6)
    histogram = _native(dual._histogram_chunk, interpreted)
    rows = np.array([[0, 1], [1, 1]], dtype=np.uint64)
    counts = np.zeros(129, dtype=np.int64)
    counts[0] = 1
    histogram(rows, np.zeros(2, dtype=np.uint64), counts, 1, 4)
    assert counts[:3].tolist() == [1, 2, 1]
    single = np.array([[0b011], [0b110]], dtype=np.uint64)
    counts = np.zeros(4, dtype=np.int64)
    counts[0] = 1
    histogram(single, np.zeros(1, dtype=np.uint64), counts, 1, 4)
    assert counts.tolist() == [1, 0, 3, 0]


def test_krawtchouk_rows_are_integral_and_symmetric() -> None:
    """The streamed degrees satisfy K_w(j) C(n, j) == K_j(w) C(n, w)."""
    width = 9
    table = list(dual._iter_krawtchouk(width))
    for weight, jj in itertools.product(range(width + 1), repeat=2):
        assert table[weight][jj] * math.comb(width, jj) == table[jj][weight] * math.comb(
            width, weight
        )
