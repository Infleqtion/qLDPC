# SPDX-License-Identifier: Apache-2.0

"""Native binary primitives shared by distance and weight enumeration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from numba import njit, types
from numba.extending import intrinsic, overload

####################################################################################################
# binary word primitives


def popcount_word(value: np.uint64) -> np.int64:
    """Count all 64 bits, with an identical Python/debug fallback."""
    return np.int64(int(value).bit_count())


def trailing_zeros(value: np.uint64) -> np.int64:
    """Count trailing zero bits, returning 64 for zero."""
    integer = int(value)
    return np.int64((integer & -integer).bit_length() - 1 if integer else 64)


####################################################################################################
# packed binary elimination


@njit(cache=True)
def row_reduce_binary(
    matrix: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    columns: npt.NDArray[np.int_],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.int_]]:
    """Reduce selected columns over GF(2), applying identical row operations to binary labels.

    Rows and labels are packed once, so elimination uses word-wise XOR rather than Python loops
    over byte-valued rows.  Physical and label widths are independent and may both exceed 64.
    Input arrays are never modified; pivot order follows the supplied column order.
    """
    if len(labels) != len(matrix):
        raise ValueError("Binary matrix and label row counts differ")
    for column in columns:
        if column < 0 or column >= matrix.shape[1]:
            raise IndexError("Binary elimination column is out of range")
    working, tags = _pack_rows(matrix), _pack_rows(labels)
    pivots = np.empty(min(len(matrix), len(columns)), dtype=np.int_)
    rank = 0
    for column in columns:
        if rank == len(matrix):
            break
        word = column // 64
        mask = np.uint64(1) << np.uint64(column % 64)
        source = len(matrix)
        for row in range(rank, len(matrix)):
            if working[row, word] & mask:
                source = row
                break
        if source == len(matrix):
            continue
        if source != rank:
            for jj in range(working.shape[1]):
                temporary = working[rank, jj]
                working[rank, jj] = working[source, jj]
                working[source, jj] = temporary
            for jj in range(tags.shape[1]):
                temporary = tags[rank, jj]
                tags[rank, jj] = tags[source, jj]
                tags[source, jj] = temporary
        for row in range(len(matrix)):
            if row != rank and working[row, word] & mask:
                for jj in range(working.shape[1]):
                    working[row, jj] ^= working[rank, jj]
                for jj in range(tags.shape[1]):
                    tags[row, jj] ^= tags[rank, jj]
        pivots[rank] = column
        rank += 1
    return (
        _unpack_rows(working, matrix.shape[1]),
        _unpack_rows(tags, labels.shape[1]),
        pivots[:rank],
    )


@njit(cache=True, inline="always")
def _pack_rows(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint64]:
    """Validate every entry and accumulate each word locally before its single output store."""
    result = np.empty((len(matrix), (matrix.shape[1] + 63) // 64), dtype=np.uint64)
    for row in range(len(matrix)):
        for word in range(result.shape[1]):
            packed = np.uint64(0)
            start = 64 * word
            for bit in range(min(64, matrix.shape[1] - start)):
                value = matrix[row, start + bit]
                if value != 0 and value != 1:
                    raise ValueError("Binary elimination requires binary matrix and label entries")
                packed |= np.uint64(value) << np.uint64(bit)
            result[row, word] = packed
    return result


@njit(cache=True, inline="always")
def _unpack_rows(matrix: npt.NDArray[np.uint64], width: int) -> npt.NDArray[np.uint8]:
    result = np.empty((len(matrix), width), dtype=np.uint8)
    for row in range(len(matrix)):
        for column in range(width):
            result[row, column] = (matrix[row, column // 64] >> np.uint64(column % 64)) & np.uint64(
                1
            )
    return result


####################################################################################################
# native compiler extensions


def _ctpop_definition(context: Any, value: Any) -> Any:
    """Lower the typed uint64 operation directly to LLVM's population-count intrinsic."""
    if value != types.uint64:
        raise TypeError("Population count requires uint64")

    def emit(context: Any, builder: Any, signature: Any, arguments: Any) -> Any:
        return builder.ctpop(arguments[0])

    return types.uint64(types.uint64), emit


def _cttz_definition(context: Any, value: Any) -> Any:
    """Use the defined-zero form: cttz(0) must return 64 in both execution modes."""
    if value != types.uint64:
        raise TypeError("Trailing-zero count requires uint64")

    def emit(context: Any, builder: Any, signature: Any, arguments: Any) -> Any:
        return builder.cttz(arguments[0], context.get_constant(types.boolean, False))

    return types.uint64(types.uint64), emit


_ctpop = cast(Callable[[np.uint64], np.uint64], intrinsic(_ctpop_definition))
_cttz = cast(Callable[[np.uint64], np.uint64], intrinsic(_cttz_definition))


@overload(popcount_word, inline="always")
def _overload_popcount(value: Any) -> Callable[[np.uint64], np.int64]:
    def implementation(value: Any) -> np.int64:
        return np.int64(_ctpop(value))

    return implementation


@overload(trailing_zeros, inline="always")
def _overload_trailing_zeros(value: Any) -> Callable[[np.uint64], np.int64]:
    def implementation(value: Any) -> np.int64:
        return np.int64(_cttz(value))

    return implementation
