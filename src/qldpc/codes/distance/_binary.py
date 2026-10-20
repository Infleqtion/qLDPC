# SPDX-License-Identifier: Apache-2.0

"""Shared binary algebra, packed weights and array conversion for distance searches."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

_MASK55 = np.uint64(0x5555555555555555)
_MASK33 = np.uint64(0x3333333333333333)
_MASK0F = np.uint64(0x0F0F0F0F0F0F0F0F)


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
    matrix = matrix.copy()
    labels = None if labels is None else labels.copy()
    columns = np.arange(matrix.shape[1], dtype=int) if columns is None else columns
    pivots: list[int] = []
    pivot_row = 0

    for col in columns:
        candidates = np.flatnonzero(matrix[pivot_row:, col])
        if not len(candidates):
            continue
        source_row = pivot_row + int(candidates[0])
        if source_row != pivot_row:
            matrix[[pivot_row, source_row]] = matrix[[source_row, pivot_row]]
            if labels is not None:
                labels[[pivot_row, source_row]] = labels[[source_row, pivot_row]]

        rows_to_clear = np.flatnonzero(matrix[:, col])
        rows_to_clear = rows_to_clear[rows_to_clear != pivot_row]
        matrix[rows_to_clear] ^= matrix[pivot_row]
        if labels is not None:
            labels[rows_to_clear] ^= labels[pivot_row]

        pivots.append(int(col))
        pivot_row += 1
        if pivot_row == len(matrix):
            break

    return matrix, labels, np.asarray(pivots, dtype=int)


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


def _get_packed_row_weights(rows: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Compute Hamming weights of packed binary rows without narrow-integer overflow."""
    return np.bitwise_count(rows).sum(axis=-1, dtype=np.uint64)


def _symplectic_to_hamming(vectors: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Map ``(X | Z)`` to ``(X | Z | X xor Z)``, doubling symplectic weight.

    This is the ``Saved_isometry`` reduction from https://arxiv.org/abs/2408.10743.
    """
    half_width = vectors.shape[1] // 2
    vectors_x = vectors[:, :half_width]
    vectors_z = vectors[:, half_width:]
    return np.hstack([vectors_x, vectors_z, vectors_x ^ vectors_z])


####################################################################################################
# weight functions


def _hamming_weight(
    words: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
) -> npt.NDArray[np.uint64]:
    """Count the set bits of each packed word.

    Without ``out``, the counts are uint8, so sums over many words must use a wider dtype.
    """
    return np.bitwise_count(words, out=out)


def _symplectic_weight(
    words: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
) -> npt.NDArray[np.uint64]:
    """Count the qubits in the support of each riffled word (see ``_riffle``).

    Bits ``2j`` and ``2j + 1`` hold the X and Z support of qubit ``j``, so a qubit is in the support
    if either bit is set.  The result is uint64, and ``out`` doubles as scratch space.
    """
    out = np.right_shift(words, 1, out=out)
    out |= words
    out &= _MASK55
    return np.bitwise_count(out, out=out)


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
