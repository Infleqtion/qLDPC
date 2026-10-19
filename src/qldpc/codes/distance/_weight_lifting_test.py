# SPDX-License-Identifier: Apache-2.0

"""Tests for lifting minimum supports to full physical-Pauli counts."""

from __future__ import annotations

import itertools
from unittest import mock

import numpy as np
import pytest

from . import _weight_lifting as lifting
from ._weight_rows import StorageBudget, pack
from .conftest import _native

_BUDGET = StorageBudget(limit_bytes=1 << 20, basis_bytes=0, batch_size=2)


@pytest.mark.parametrize("interpreted", [False, True])
def test_rank_word_integer_oracle(interpreted: bool) -> None:
    """Exhaust restricted three-row spaces, reusing poisoned scratch across every query."""
    rank = _native(lifting._rank_word, interpreted)
    scratch = np.full(64, 2**64 - 1, dtype=np.uint64)
    coordinates = (0, 31, 63)
    words = [
        sum(((bits >> jj) & 1) << pos for jj, pos in enumerate(coordinates)) for bits in range(8)
    ]
    for triple in itertools.product(words, repeat=3):
        backing = np.zeros(6, dtype=np.uint64)
        backing[::2] = triple
        rows = backing[::2]
        for mask in words:
            span = {0}
            for value in triple:
                span |= {old ^ (value & mask) for old in span}
            assert rank(rows, np.uint64(mask), scratch) == len(span).bit_length() - 1
        np.testing.assert_array_equal(rows, triple)
    assert rank(np.empty(0, dtype=np.uint64), np.uint64(2**64 - 1), scratch) == 0


@pytest.mark.parametrize("interpreted", [False, True])
def test_shortened_word_span_oracle(interpreted: bool) -> None:
    """Count contained words independently, including bit 63 and a reused rank buffer."""
    shortened = _native(lifting._shortened_histogram_one, interpreted)
    rows = np.array([5, 6, 8, 2**63], dtype=np.uint64)
    span = {0}
    for value in rows:
        span |= {old ^ int(value) for old in span}
    supports = np.array(
        [
            sum(((bits >> jj) & 1) << pos for jj, pos in enumerate((0, 1, 2, 3, 63)))
            for bits in range(32)
        ],
        dtype=np.uint64,
    )
    expected = np.zeros(65, dtype=np.int64)
    for support in supports:
        contained = sum(value & ~int(support) == 0 for value in span)
        expected[contained.bit_length() - 1] += 1
    np.testing.assert_array_equal(shortened(rows, supports, 64), expected)


@pytest.mark.parametrize("interpreted", [False, True])
def test_native_rank_and_shortened_corner_cases(interpreted: bool) -> None:
    rows = np.array([[0, 1], [1, 1]], dtype=np.uint64)
    shortened = _native(lifting._shortened_histogram, interpreted)
    np.testing.assert_array_equal(
        shortened(rows, np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint64), 2),
        [0, 2, 1],
    )
    np.testing.assert_array_equal(
        shortened(np.array([[3], [4]], dtype=np.uint64), np.array([[3], [7]], dtype=np.uint64), 2),
        [0, 1, 1],
    )
    rank = _native(lifting._rank_masked, interpreted)
    assert rank(np.array([[1], [2], [3]], dtype=np.uint64), np.array([3], dtype=np.uint64)) == 2
    assert (
        rank(
            np.array([[0, 1], [0, 2], [0, 3], [1, 0]], dtype=np.uint64),
            np.array([2**64 - 1, 2**64 - 1], dtype=np.uint64),
        )
        == 3
    )
    rank_word = _native(lifting._rank_word, interpreted)
    scratch = np.empty(64, dtype=np.uint64)
    assert rank_word(np.array([], dtype=np.uint64), np.uint64(3), scratch) == 0
    assert rank_word(np.array([1], dtype=np.uint64), np.uint64(0), scratch) == 0
    assert rank_word(np.array([1, 2, 3], dtype=np.uint64), np.uint64(3), scratch) == 2


def test_shared_and_dressed_supports() -> None:
    """A shared support gives X, Z and Y; a one-sided support gains its contained stabilizers."""
    no_stabilizers = pack(np.empty((0, 4), dtype=np.uint8))
    shared = pack(np.array([[1, 1, 0, 0]], dtype=np.uint8))
    nothing = shared[:0]
    assert (
        lifting.count_minimum_paulis(shared, shared, no_stabilizers, no_stabilizers, 2, _BUDGET)
        == 3
    )
    assert (
        lifting.count_minimum_paulis(shared, nothing, no_stabilizers, no_stabilizers, 2, _BUDGET)
        == 1
    )

    # An X-only support {0, 1} containing the Z stabilizer ZZ on those qubits has two dressings.
    z_stabilizers = pack(np.array([[1, 1, 0, 0]], dtype=np.uint8))
    assert (
        lifting.count_minimum_paulis(shared, nothing, no_stabilizers, z_stabilizers, 2, _BUDGET)
        == 2
    )


def test_arbitrary_precision_total_and_batching() -> None:
    """Dressing counts exponentiate in Python, so totals can exceed any native word."""
    histogram = np.zeros(81, dtype=np.int64)
    histogram[80] = 1
    empty = pack(np.empty((0, 80), dtype=np.uint8))
    supports = np.array([[1, 0]], dtype=np.uint64)
    with mock.patch.object(lifting, "_shortened_histogram", return_value=histogram):
        assert (
            lifting.count_minimum_paulis(supports, supports[:0], empty, empty, 80, _BUDGET) == 2**80
        )


def test_union_storage_is_checked_before_lifting() -> None:
    small = StorageBudget(limit_bytes=1, basis_bytes=0, batch_size=1)
    supports = np.array([[1]], dtype=np.uint64)
    empty = np.empty((0, 1), dtype=np.uint64)
    with pytest.raises(MemoryError, match="union"):
        lifting.count_minimum_paulis(supports, empty, empty, empty, 1, small)
