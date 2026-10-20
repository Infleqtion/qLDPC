# SPDX-License-Identifier: Apache-2.0

"""Shared binary algebra, packed weights and array conversion for distance searches."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from llvmlite import ir
from numba import njit, types
from numba.core.base import BaseContext
from numba.core.typing.context import BaseContext as TypingContext
from numba.core.typing.templates import Signature
from numba.extending import lower_builtin, type_callable

_MASK55 = np.uint64(0x5555555555555555)
_MASK33 = np.uint64(0x3333333333333333)
_MASK0F = np.uint64(0x0F0F0F0F0F0F0F0F)
_MASK01 = np.uint64(0x0101010101010101)


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
    """Row-reduce a binary matrix over selected columns, transforming labels in parallel.

    Pivots are taken in the order of ``columns`` (by default, every column in order), each from the
    first remaining row with a one in that column, and every other row is cleared in that column.
    The inputs are not modified.
    """
    width = matrix.shape[1]
    columns = np.arange(width) if columns is None else np.asarray(columns)
    if len(columns) and (columns.min() < 0 or columns.max() >= width):
        raise IndexError("Row-reduction columns are out of range")
    tags = np.empty((len(matrix), 0), dtype=np.uint8) if labels is None else labels
    reduced, reduced_tags, pivots = _row_reduce_packed(matrix, tags, columns.astype(np.int64))
    return reduced, None if labels is None else reduced_tags, pivots.astype(int)


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
# native bit counting, packing and row reduction


def _popcount_word(value: np.uint64) -> np.int64:
    """Count the set bits of one 64-bit word.

    Compiled code calls LLVM's population-count instruction instead of this Python fallback.
    """
    return np.int64(int(value).bit_count())


@type_callable(_popcount_word)
def _type_popcount_word(context: TypingContext) -> Callable[[types.Type], types.Type | None]:
    """Declare that compiled code may call ``_popcount_word`` on a uint64 word."""

    def typer(value: types.Type) -> types.Type | None:
        return types.int64 if value == types.uint64 else None

    return typer


@lower_builtin(_popcount_word, types.uint64)
def _lower_popcount_word(
    context: BaseContext, builder: ir.IRBuilder, signature: Signature, arguments: list[ir.Value]
) -> ir.Value:
    """Compile ``_popcount_word`` to LLVM's population-count instruction.

    The count is at most 64, so the 64-bit result is the same as a signed integer.
    """
    return builder.ctpop(arguments[0])


@njit(cache=True)
def _pack_bits(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint64]:
    """Pack binary rows into uint64 words, with column j in bit j % 64 of word j // 64.

    Entries must be 0 or 1.  This layout differs from that of ``_rows_to_ints``, which puts the
    first column in the most significant bit; do not mix the two.
    """
    packed = np.zeros((len(matrix), (matrix.shape[1] + 63) // 64), dtype=np.uint64)
    for row in range(len(matrix)):
        for column in range(matrix.shape[1]):
            if matrix[row, column]:
                packed[row, column // 64] |= np.uint64(1) << np.uint64(column % 64)
    return packed


@njit(cache=True)
def _unpack_bits(packed: npt.NDArray[np.uint64], width: int) -> npt.NDArray[np.uint8]:
    """Inverse of ``_pack_bits`` for rows with ``width`` columns."""
    matrix = np.empty((len(packed), width), dtype=np.uint8)
    for row in range(len(packed)):
        for column in range(width):
            matrix[row, column] = (packed[row, column // 64] >> np.uint64(column % 64)) & 1
    return matrix


@njit(cache=True)
def _row_reduce_packed(
    matrix: npt.NDArray[np.uint8], labels: npt.NDArray[np.uint8], columns: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Implement ``_row_reduce_binary`` on packed copies, for in-range columns."""
    rows, tags = _pack_bits(matrix), _pack_bits(labels)
    num_rows = len(rows)
    pivots = np.empty(min(num_rows, len(columns)), dtype=np.int64)
    rank = 0
    for column in columns:
        if rank == num_rows:
            break
        word = column // 64
        bit = np.uint64(1) << np.uint64(column % 64)
        source = rank
        while source < num_rows and not rows[source, word] & bit:
            source += 1
        if source == num_rows:
            continue
        if source != rank:
            for index in range(rows.shape[1]):
                rows[rank, index], rows[source, index] = rows[source, index], rows[rank, index]
            for index in range(tags.shape[1]):
                tags[rank, index], tags[source, index] = tags[source, index], tags[rank, index]
        for row in range(num_rows):
            if row != rank and rows[row, word] & bit:
                for index in range(rows.shape[1]):
                    rows[row, index] ^= rows[rank, index]
                for index in range(tags.shape[1]):
                    tags[row, index] ^= tags[rank, index]
        pivots[rank] = column
        rank += 1
    return _unpack_bits(rows, matrix.shape[1]), _unpack_bits(tags, labels.shape[1]), pivots[:rank]


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
