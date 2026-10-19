# SPDX-License-Identifier: Apache-2.0

"""Stabilizer enumeration and the MacWilliams transform for CSS logical weight spectra.

The weight histogram of the X (or Z) stabilizer group is enumerated by a Gray-code walk.  The
MacWilliams identity then gives the weights of the full normalizer, whose nontrivial part is the
set of logical representatives.  Dual enumeration scales with the stabilizer space rather than the
generally larger logical space.

References:

- https://en.wikipedia.org/wiki/MacWilliams_identity
"""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
from numba import njit

from ._binary_numba import popcount_word, trailing_zeros

PackedMatrix = npt.NDArray[np.uint64]
Histogram = npt.NDArray[np.int64]

####################################################################################################
# stabilizer weight histograms


def get_stabilizer_histograms(
    checks_x: PackedMatrix,
    checks_z: PackedMatrix,
    num_qubits: int,
    *,
    symmetric: bool,
    batch_size: int,
) -> tuple[Histogram, Histogram]:
    """Histogram the weights of the X and Z stabilizer groups, enumerating at most one of them.

    The native histogram counts in int64, so each rank must be at most 62.  The Gray-code walk is
    resumed in chunks of ``batch_size``, which bounds the time between interruptions.
    """
    if max(len(checks_x), len(checks_z)) > 62:
        raise OverflowError("Dual ranks above 62 require BZ instead of native histograms")
    counts = []
    for rows in (checks_x,) if symmetric else (checks_x, checks_z):
        histogram = np.zeros(num_qubits + 1, dtype=np.int64)
        histogram[0] = 1  # the identity; the walk below visits every nonzero combination
        current = np.zeros(rows.shape[1], dtype=np.uint64)
        for start in range(1, 1 << len(rows), batch_size):
            _histogram_chunk(
                rows, current, histogram, start, min(start + batch_size, 1 << len(rows))
            )
        counts.append(histogram)
    return counts[0], counts[0] if symmetric else counts[1]


####################################################################################################
# MacWilliams transform


def get_macwilliams_spectrum(
    histogram_x: Histogram,
    histogram_z: Histogram,
    rank_x: int,
    rank_z: int,
    num_qubits: int,
    max_weight: int,
) -> tuple[list[int], list[int]]:
    """Exact nontrivial X-only and Z-only counts through ``max_weight``, as Python integers.

    The X normalizer is the dual of the Z stabilizer group, so its weight enumerator comes from
    the Z histogram; the stabilizer words are then subtracted to leave only logical operators.
    """
    counts_x, counts_z = [], []
    for weight, coefficients in enumerate(
        itertools.islice(_iter_krawtchouk(num_qubits), max_weight + 1)
    ):
        count_x = _dual_coefficient(histogram_z, coefficients, rank_z) - int(histogram_x[weight])
        count_z = _dual_coefficient(histogram_x, coefficients, rank_x) - int(histogram_z[weight])
        if count_x < 0 or count_z < 0:
            raise ArithmeticError("Negative logical multiplicity")
        counts_x.append(count_x)
        counts_z.append(count_z)
    return counts_x, counts_z


def get_macwilliams_minimum(
    histogram_x: Histogram,
    histogram_z: Histogram,
    rank_x: int,
    rank_z: int,
    num_qubits: int,
) -> tuple[int, int, int]:
    """Smallest weight with a nontrivial logical, and its X-only and Z-only counts.

    A native scan handles the prefix of weights whose partial sums provably fit in int64.  If no
    logical appears there, exact Python-integer arithmetic continues from weight zero.
    """
    coefficients = _get_safe_coefficients(num_qubits, max(rank_x, rank_z))
    minimum, count_x, count_z = _minimum_native(
        histogram_x, histogram_z, rank_x, rank_z, coefficients
    )
    if minimum >= 0:
        return int(minimum), int(count_x), int(count_z)
    for weight, row in enumerate(_iter_krawtchouk(num_qubits)):
        count_x = _dual_coefficient(histogram_z, row, rank_z) - int(histogram_x[weight])
        count_z = _dual_coefficient(histogram_x, row, rank_x) - int(histogram_z[weight])
        if count_x < 0 or count_z < 0:
            raise ArithmeticError("Negative logical multiplicity")
        if count_x or count_z:
            return weight, count_x, count_z
    raise ArithmeticError("A positive-dimensional CSS code has no nontrivial logicals")


####################################################################################################
# helper methods


def _iter_krawtchouk(width: int) -> Iterator[tuple[int, ...]]:
    """Stream exact degrees so a low-weight query does not allocate a quadratic full table."""
    previous, current = (0,) * (width + 1), (1,) * (width + 1)
    for weight in range(width + 1):
        yield current
        if weight < width:
            current, previous = (
                tuple(
                    ((width - 2 * jj) * current[jj] - (width - weight + 1) * previous[jj])
                    // (weight + 1)
                    for jj in range(width + 1)
                ),
                current,
            )


@functools.lru_cache(maxsize=32)
def _get_safe_coefficients(width: int, rank: int) -> npt.NDArray[np.int64]:
    """Certify every intermediate partial sum by 2**rank * binomial(width, weight)."""
    limit = next(
        (
            weight
            for weight in range(width + 1)
            if (1 << rank) * math.comb(width, weight) > np.iinfo(np.int64).max
        ),
        width + 1,
    )
    rows = tuple(itertools.islice(_iter_krawtchouk(width), limit))
    return np.array(rows, dtype=np.int64).reshape(limit, width + 1)


def _dual_coefficient(histogram: Histogram, coefficients: tuple[int, ...], rank: int) -> int:
    numerator = sum(int(count) * value for count, value in zip(histogram, coefficients))
    value, remainder = divmod(numerator, 1 << rank)
    if remainder or value < 0:
        raise ArithmeticError("Invalid exact MacWilliams coefficient")
    return value


@njit(cache=True)
def _histogram_chunk(
    rows: PackedMatrix,
    current: npt.NDArray[np.uint64],
    counts: Histogram,
    start: int,
    stop: int,
) -> None:
    """Continue a Gray walk using the caller-owned current word and histogram."""
    if rows.shape[1] == 1:
        value = current[0]
        for index in range(start, stop):
            value ^= rows[trailing_zeros(np.uint64(index)), 0]
            counts[popcount_word(value)] += 1
        current[0] = value
        return
    for index in range(start, stop):
        changed = trailing_zeros(np.uint64(index))
        weight = np.int64(0)
        for word in range(rows.shape[1]):
            current[word] ^= rows[changed, word]
            weight += popcount_word(current[word])
        counts[weight] += 1


@njit(cache=True)
def _minimum_native(
    histogram_x: Histogram,
    histogram_z: Histogram,
    rank_x: int,
    rank_z: int,
    coefficients: npt.NDArray[np.int64],
) -> tuple[int, int, int]:
    """Transform only the previously certified int64-safe coefficient prefix."""
    for weight in range(len(coefficients)):
        nx, nz = np.int64(0), np.int64(0)
        for jj in range(len(histogram_x)):
            nx += histogram_z[jj] * coefficients[weight, jj]
            nz += histogram_x[jj] * coefficients[weight, jj]
        if nx % (1 << rank_z) or nz % (1 << rank_x):
            raise ArithmeticError("Nonintegral MacWilliams coefficient")
        ax = nx // (1 << rank_z) - histogram_x[weight]
        az = nz // (1 << rank_x) - histogram_z[weight]
        if ax < 0 or az < 0:
            raise ArithmeticError("Negative logical multiplicity")
        if ax or az:
            return weight, ax, az
    return -1, 0, 0
