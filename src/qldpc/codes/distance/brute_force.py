# SPDX-License-Identifier: Apache-2.0

"""Exact packed enumeration of independent logical and excluded binary bases."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._binary import (
    _count_trailing_zeros,
    _inplace_rowsum,
    _riffle,
    _rows_to_ints,
    _symplectic_weight,
)

####################################################################################################
# helper methods


def _get_distance_quantum_brute_force(
    logical_ops: npt.ArrayLike,
    stabilizers: npt.ArrayLike,
    *,
    cutoff: int,
    block_size: int,
    homogeneous: bool,
) -> int:
    """Enumerate independent logical and excluded bases prepared or verified in this module."""
    num_bits = np.shape(logical_ops)[-1]

    weight_func = np.bitwise_count if homogeneous else _symplectic_weight
    if not homogeneous:
        logical_ops = _riffle(logical_ops)
        stabilizers = _riffle(stabilizers)

    int_logical_ops = _rows_to_ints(logical_ops, dtype=np.uint64)
    int_stabilizers = _rows_to_ints(stabilizers, dtype=np.uint64)
    num_stabilizers = len(int_stabilizers)

    # Number of generators to include in the operational array.  Most calculations will then be
    # vectorized over ``2**block_size`` values.  Clamp at 0: when the packed word-count per row
    # exceeds ``block_size + 1`` the first term goes negative, which would make a negative slice and
    # build an uncapped ``2**(S-k)`` array, defeating the block_size cap.
    num_vectorized_ops = max(
        0,
        min(
            block_size + 1 - int_logical_ops.shape[-1],
            len(int_logical_ops) + len(int_stabilizers),
        ),
    )

    # Vectorize all combinations of first `num_vectorized_ops` stabilizers
    array = np.zeros((1, int_logical_ops.shape[-1]), dtype=np.uint64)
    for op in int_stabilizers[:num_vectorized_ops]:
        array = np.vstack([array, array ^ op])

    if num_vectorized_ops > num_stabilizers:
        # fill out block with products of some logical ops
        for op in int_logical_ops[: num_vectorized_ops - num_stabilizers]:
            array = np.vstack([array, array ^ op])

        int_logical_ops = int_logical_ops[num_vectorized_ops - num_stabilizers :]

    int_stabilizers = int_stabilizers[num_vectorized_ops:]

    # Everything below will run much faster if we use Fortran-style ordering
    arrayf = np.asarray(array, order="F", dtype=np.uint64)

    # out is the uint64 buffer for the in-loop weight_func calls: passing out=out keeps
    # np.bitwise_count's uint8 result cast to uint64, avoiding the weight-reduction overflow for
    # codes with >= 256 columns (the pre-loop call below casts explicitly for the same reason).
    out = np.empty_like(arrayf)

    # Min weight of the block containing logical ops.  Cast to uint64 because np.bitwise_count
    # returns uint8 without ``out=``, which would overflow the reduction below for >= 256 columns.
    weights = np.asarray(weight_func(arrayf[2**num_stabilizers :]), dtype=np.uint64)
    min_weight = _inplace_rowsum(weights).min(initial=num_bits)
    if min_weight <= cutoff:
        return int(min_weight)

    # Sweep over every remaining logical-op combination and, nested, every remaining stabilizer
    # combination, on top of the vectorized block above.  The sweep uses a reflected Gray code:
    # successive Gray-code words differ in one bit, whose position is the trailing-zero count of the
    # step index.  Each step therefore flips a single operator into/out of the running XOR
    # (``arrayf``) with one ^=, so the loop walks the 2**k - 1 nonzero combinations of the remaining
    # operators (the empty combination is the pre-loop state) without rebuilding any.
    # ``min_weight`` tracks the lightest operator seen; ``cutoff`` lets the sweep stop once that
    # bound is reached (see Returns).
    for li in range(1, 2 ** len(int_logical_ops)):
        arrayf ^= int_logical_ops[_count_trailing_zeros(li)]
        weights = weight_func(arrayf, out=out)
        min_weight = _inplace_rowsum(weights).min(initial=min_weight)
        if min_weight <= cutoff:
            return int(min_weight)

        for si in range(1, 2 ** len(int_stabilizers)):
            arrayf ^= int_stabilizers[_count_trailing_zeros(si)]
            weights = weight_func(arrayf, out=out)
            min_weight = _inplace_rowsum(weights).min(initial=min_weight)
            if min_weight <= cutoff:
                return int(min_weight)

    return int(min_weight)
