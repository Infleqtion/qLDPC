# SPDX-License-Identifier: Apache-2.0

"""Shared binary algebra, packed weights, native bit primitives and array conversion.

The weight-enumeration backends pack physical coordinate zero into the low bit of a word, unlike
``_rows_to_ints`` below; do not mix the two layouts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
from numba import njit, types
from numba.extending import intrinsic, overload

_MASK55 = np.uint64(0x5555555555555555)
_MASK33 = np.uint64(0x3333333333333333)
_MASK0F = np.uint64(0x0F0F0F0F0F0F0F0F)
_MASK01 = np.uint64(0x0101010101010101)


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


####################################################################################################
# helper methods


def _assert_binary(vectors: npt.ArrayLike, name: str) -> None:
    """Assert that the given vectors are binary.

    The enumeration below packs each row into the bits of ``uint64`` words and combines rows by
    bitwise XOR, so it is only correct over GF(2).  Non-binary input would otherwise be truncated
    silently and yield a wrong distance rather than an error.
    """
    order = getattr(type(vectors), "order", None)
    if order is not None and order != 2:
        raise ValueError(
            f"Distance calculations only support binary codes, but {name} is defined over"
            f" GF({order})"
        )
    array = np.asarray(vectors)
    if array.size and not np.all((array == 0) | (array == 1)):
        raise ValueError(
            f"Distance calculations only support binary codes, but {name} has entries other"
            " than 0 and 1"
        )


def _as_binary_matrix(
    vectors: npt.ArrayLike, *, num_cols: int | None = None
) -> npt.NDArray[np.uint8]:
    """Convert a collection of binary rows to a two-dimensional uint8 array."""
    matrix = np.asarray(vectors, dtype=np.uint8)
    if matrix.size == 0:
        if num_cols is None:
            if matrix.ndim != 2:
                raise ValueError("Cannot infer the width of an empty binary matrix")
            num_cols = matrix.shape[1]
        return np.empty((0, num_cols), dtype=np.uint8)
    matrix = np.atleast_2d(matrix)
    if matrix.ndim != 2:
        raise ValueError("Binary generators must be a two-dimensional matrix")
    if num_cols is not None and matrix.shape[1] != num_cols:
        raise ValueError(
            f"Binary matrices have incompatible widths {num_cols} and {matrix.shape[1]}"
        )
    return matrix


def _row_reduce_binary(
    matrix: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None = None,
    *,
    columns: npt.NDArray[np.int_] | None = None,
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8] | None,
    npt.NDArray[np.int_],
]:
    """Row-reduce a binary matrix over selected columns, transforming labels in parallel."""
    columns = np.arange(matrix.shape[1], dtype=int) if columns is None else columns
    tags = np.empty((len(matrix), 0), dtype=np.uint8) if labels is None else labels
    reduced, reduced_tags, pivots = row_reduce_binary(matrix, tags, columns)
    return reduced, None if labels is None else reduced_tags, pivots


def _get_independent_rows(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Return a row-reduced basis for the row space of a binary matrix."""
    reduced, _, pivots = _row_reduce_binary(matrix)
    return reduced[: len(pivots)]


def _get_nested_code_basis(
    logical_ops: npt.NDArray[np.uint8],
    stabilizers: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Build an adapted basis for ``span(stabilizers, logical_ops) / span(stabilizers)``."""
    stabilizer_basis = _get_independent_rows(stabilizers)
    stabilizer_pivots = [int(np.flatnonzero(row)[0]) for row in stabilizer_basis]
    reduced_logical_ops = logical_ops.copy()
    for row, pivot in zip(stabilizer_basis, stabilizer_pivots, strict=True):
        reduced_logical_ops[reduced_logical_ops[:, pivot] == 1] ^= row

    logical_quotient = _get_independent_rows(reduced_logical_ops)
    basis = np.vstack([stabilizer_basis, logical_quotient]).astype(np.uint8, copy=False)
    labels = np.zeros((len(basis), len(logical_quotient)), dtype=np.uint8)
    labels[len(stabilizer_basis) :] = np.eye(len(logical_quotient), dtype=np.uint8)
    return basis, labels


def _get_packed_row_weights(
    rows: npt.NDArray[np.uint64],
    weight_func: Callable[..., npt.NDArray[np.uint64]],
) -> npt.NDArray[np.uint64]:
    """Compute Hamming weights of packed binary rows without narrow-integer overflow."""
    word_weights = np.asarray(weight_func(rows), dtype=np.uint64)
    return word_weights.sum(axis=-1, dtype=np.uint64)


def _symplectic_to_hamming(vectors: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Map ``(X | Z)`` to ``(X | Z | X xor Z)``, doubling symplectic weight.

    This is the ``Saved_isometry`` reduction from https://arxiv.org/abs/2408.10743.
    """
    half_width = vectors.shape[1] // 2
    vectors_x = vectors[:, :half_width]
    vectors_z = vectors[:, half_width:]
    return np.hstack([vectors_x, vectors_z, vectors_x ^ vectors_z])


####################################################################################################
# weight functions (Hamming and symplectic popcount) and backend selection


def _get_hamming_weight_fn() -> tuple[Callable[..., npt.NDArray[np.uint64]], int]:
    if getattr(np, "bitwise_count", None) is not None:
        weight_fn = np.bitwise_count
        return weight_fn, 0

    return _hamming_weight, 1


def _get_symplectic_weight_fn() -> tuple[Callable[..., npt.NDArray[np.uint64]], int]:
    if getattr(np, "bitwise_count", None) is not None:
        np_bitwise_count = np.bitwise_count

        def weight_fn(
            arr: npt.NDArray[np.uint64],
            buf: npt.NDArray[np.uint64] | None = None,
            out: npt.NDArray[np.uint64] | None = None,
        ) -> npt.NDArray[np.uint64]:
            """Symplectic weight of an integer."""
            buf = np.right_shift(arr, 1, out=buf)
            buf |= arr
            buf &= _MASK55
            return np_bitwise_count(buf, out=out)

        return weight_fn, 1

    return _symplectic_weight, 1


####################################################################################################
# bit-packing and array helpers


def _count_trailing_zeros(val: int) -> int:
    """Returns the position of the least significant 1 in the binary representation of `val`."""
    return (val & -val).bit_length() - 1


def _inplace_rowsum(arr: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Destructively compute ``arr.sum(-1)``, placing the result in the first column of ``arr``.

    When complete, the returned sum will be stored in ``arr[..., 0]``, while other entries in
    ``arr[..., 1:]`` will be left in indeterminate states.  This permits a faster sum
    implementation.
    """
    width = arr.shape[-1]
    while width > 1:
        split = width // 2
        arr[..., :split] += arr[..., width - split : width]
        width -= split

    return arr[..., 0]


def _rows_to_ints(
    array: npt.ArrayLike, dtype: npt.DTypeLike = np.uint64, axis: int = -1
) -> npt.NDArray[np.uint64]:
    """Pack rows of a binary array into rows of the given integral type."""
    array = np.asarray(array, dtype=dtype)
    tsize = array.itemsize * 8

    if array.size == 0:
        num_words = int(np.ceil(array.shape[-1] / tsize))
        return np.empty((*array.shape[:-1], num_words), dtype=dtype)

    def _to_int(bits: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
        """Pack `bits` into a single integer (of type `dtype`)."""
        return (bits << np.arange(len(bits) - 1, -1, -1, dtype=dtype)).sum(dtype=dtype)

    def _to_ints(bits: npt.NDArray[np.uint64]) -> list[npt.NDArray[np.uint64]]:
        """Pack a single row of bits into a row of integers."""
        return [_to_int(bits[i : i + tsize]) for i in range(0, np.shape(bits)[-1], tsize)]

    return np.apply_along_axis(_to_ints, axis, array)


def _riffle(array: npt.ArrayLike) -> npt.ArrayLike:
    """'Riffle' Pauli strings, putting X and Z support bits for each qubit next to each other."""
    arr = np.asarray(array)
    num_bits = arr.shape[-1]
    assert num_bits % 2 == 0
    if arr.size == 0:
        return np.empty((0, num_bits), dtype=arr.dtype)  # nothing to riffle (e.g. no stabilizers)
    return np.reshape(arr, (-1, 2, num_bits // 2)).transpose(0, 2, 1).reshape(-1, num_bits)


####################################################################################################
# numpy < 2.0 does not provide np.bitwise_count; the methods below provide a fallback


def _hamming_weight(
    arr: npt.NDArray[np.uint64],
    buf: npt.NDArray[np.uint64] | None = None,
    out: npt.NDArray[np.uint64] | None = None,
) -> npt.NDArray[np.uint64]:
    """Somewhat efficient (vectorized) Hamming weight calculation.

    Assumes 64-bit uints.  For `numpy >= 2.0.0`, it's generally better to use `np.bitwise_count`
    (which uses processors' builtin `popcnt` instruction).  Unfortunately this isn't available for
    numpy < 2.0.0.

    The mask-and-shift steps are the classic SWAR (SIMD-within-a-register) popcount; see
    https://en.wikipedia.org/wiki/Hamming_weight.
    """
    out = np.right_shift(arr, 1, out=out)
    out &= _MASK55
    out = np.subtract(arr, out, out=out)

    buf = np.right_shift(out, 2, out=buf)
    buf &= _MASK33
    out &= _MASK33
    out += buf

    buf = np.right_shift(out, 4, out=buf)
    out += buf
    out &= _MASK0F

    out = np.multiply(out, _MASK01, out=out)
    out >>= np.uint64(56)
    return out


def _symplectic_weight(
    arr: npt.NDArray[np.uint64],
    buf: npt.NDArray[np.uint64] | None = None,
    out: npt.NDArray[np.uint64] | None = None,
) -> npt.NDArray[np.uint64]:
    """Somewhat efficient (vectorized) symplectic weight calculation.

    Assumes 64-bit uints.  This function is equivalent to (but slightly more efficient than) the
    expression ``_hamming_weight((arr | (arr >> 1)) & 0x5555555555555555, buf=buf, out=out)``.
    """
    out = np.right_shift(arr, 1, out=out)
    out |= arr
    out &= _MASK55

    buf = np.right_shift(out, 2, out=buf)
    buf &= _MASK33
    out &= _MASK33
    out += buf

    buf = np.right_shift(out, 4, out=buf)
    out += buf
    out &= _MASK0F

    out *= _MASK01
    out >>= np.uint64(56)
    return out
