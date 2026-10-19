# SPDX-License-Identifier: Apache-2.0

"""Lift minimum X-only and Z-only supports to the full physical-Pauli multiplicity.

A support shared by both sectors carries an X, a Z, and a Y representative.  A support found in
only one sector can also be dressed by the opposite-type stabilizers that it contains, which is a
single rank query per support; see ``_shortened_histogram``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit

from .._binary import trailing_zeros
from .rows import PackedMatrix, StorageBudget, get_support_keys

####################################################################################################
# Pauli lifting


def count_minimum_paulis(
    supports_x: PackedMatrix,
    supports_z: PackedMatrix,
    stabilizers_x: PackedMatrix,
    stabilizers_z: PackedMatrix,
    minimum: int,
    budget: StorageBudget,
) -> int:
    """Count physical Pauli representatives of weight ``minimum``, ignoring phase.

    ``supports_x`` and ``supports_z`` are the distinct supports of the minimum X-only and Z-only
    logical words.  Stabilizers are packed row spaces.  The result is an arbitrary-size integer.
    """
    width = stabilizers_x.shape[1]
    keys_x, keys_z = get_support_keys(supports_x), get_support_keys(supports_z)

    # A support shared by both sectors carries exactly X, Z and Y words, three in all.
    shared = np.intersect1d(keys_x, keys_z, assume_unique=True)
    union_size = len(supports_x) + len(supports_z) - len(shared)
    if union_size * (96 * width) + budget.basis_bytes > budget.limit_bytes:
        raise MemoryError("Full-Pauli support union exceeds memory_limit_bytes")
    total = 3 * len(shared)

    # An X-only support can be dressed by Z stabilizers contained inside it, and conversely.
    # This needs one rank query per support, not one per candidate word.
    for keys, stabilizers in (
        (np.setdiff1d(keys_x, keys_z, assume_unique=True), stabilizers_z),
        (np.setdiff1d(keys_z, keys_x, assume_unique=True), stabilizers_x),
    ):
        supports = keys.view(np.uint64).reshape(-1, width)
        for start in range(0, len(supports), budget.batch_size):
            words = supports[start : start + budget.batch_size]
            histogram = _shortened_histogram(stabilizers, words, minimum)
            total += sum(
                int(count) << exponent for exponent, count in enumerate(histogram) if count
            )
    return total


####################################################################################################
# helper methods


@njit(cache=True)
def _rank_masked(rows: PackedMatrix, mask: npt.NDArray[np.uint64]) -> int:
    """Rank the restricted row space without unpacking its physical coordinates."""
    if rows.shape[1] == 1:
        return _rank_word(rows.reshape(-1), mask[0], np.empty(64, dtype=np.uint64))
    pivots = np.zeros((64 * rows.shape[1], rows.shape[1]), dtype=np.uint64)
    current = np.empty(rows.shape[1], dtype=np.uint64)
    rank = 0
    for row in range(len(rows)):
        for word in range(rows.shape[1]):
            current[word] = rows[row, word] & mask[word]
        while True:
            word = 0
            while word < rows.shape[1] and current[word] == 0:
                word += 1
            if word == rows.shape[1]:
                break
            pivot = 64 * word + trailing_zeros(current[word])
            if pivots[pivot, word] == 0:
                for jj in range(rows.shape[1]):
                    pivots[pivot, jj] = current[jj]
                rank += 1
                break
            for jj in range(rows.shape[1]):
                current[jj] ^= pivots[pivot, jj]
    return rank


@njit(cache=True, inline="always")
def _rank_word(
    rows: npt.NDArray[np.uint64],
    mask: np.uint64,
    pivots: npt.NDArray[np.uint64],
) -> int:
    """Use scalar masks and caller-owned scratch space for repeated single-word rank queries."""
    if not len(rows) or mask == 0:
        return 0
    pivots[:] = 0
    rank = 0
    # An array iterator would retain/release rows on every inlined support query.
    for index in range(len(rows)):
        value = rows[index] & mask
        while value:
            pivot = trailing_zeros(value)
            if pivots[pivot] == 0:
                pivots[pivot] = value
                rank += 1
                break
            value ^= pivots[pivot]
    return rank


@njit(cache=True)
def _shortened_histogram(
    stabilizers: PackedMatrix, supports: PackedMatrix, minimum: int
) -> npt.NDArray[np.int64]:
    """Count opposite-type stabilizers contained in minimum supports.

    A minimum X support cannot contain a nonzero X stabilizer: multiplying by it would give a
    lighter nontrivial logical.  On a support with no minimum Z word, all allowed Z components
    are therefore Z stabilizers.  Their number is 2**(rank(H_Z)-rank(H_Z[:,outside])).
    Group dimensions natively and exponentiate in Python to retain arbitrary-size cardinalities.
    """
    if stabilizers.shape[1] == 1:
        return _shortened_histogram_one(stabilizers.reshape(-1), supports.reshape(-1), minimum)
    histogram = np.zeros(minimum + 1, dtype=np.int64)
    for support in supports:
        dimension = len(stabilizers) - _rank_masked(stabilizers, ~support)
        histogram[dimension] += 1
    return histogram


@njit(cache=True)
def _shortened_histogram_one(
    stabilizers: npt.NDArray[np.uint64],
    supports: npt.NDArray[np.uint64],
    minimum: int,
) -> npt.NDArray[np.int64]:
    """Reuse one pivot buffer; no per-support views, mask vectors, or rank scratch allocations."""
    histogram = np.zeros(minimum + 1, dtype=np.int64)
    pivots = np.empty(64, dtype=np.uint64)
    for support in supports:
        outside = ~support
        dimension = len(stabilizers) - _rank_word(stabilizers, outside, pivots)
        histogram[dimension] += 1
    return histogram
