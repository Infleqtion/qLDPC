# SPDX-License-Identifier: Apache-2.0

"""CPU primitives and their Python/disabled-JIT and compiler-extension boundaries."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
from numba import njit, types

from qldpc.codes.distance import _binary_numba as binary


@njit
def _native(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts = np.empty(len(values), dtype=np.int64)
    zeros = np.empty_like(counts)
    for index, value in enumerate(values):
        counts[index] = binary.popcount_word(value)
        zeros[index] = binary.trailing_zeros(value)
    return counts, zeros


def test_native_and_python_bits() -> None:
    rng = np.random.default_rng(78)
    values = np.concatenate(
        (
            np.array([0, 1, 2**63, 2**64 - 1], dtype=np.uint64),
            rng.integers(0, 2**64, 512, dtype=np.uint64),
        )
    )
    counts, zeros = _native(values)
    python_counts, python_zeros = getattr(_native, "py_func", _native)(values)
    np.testing.assert_array_equal(counts, python_counts)
    np.testing.assert_array_equal(zeros, python_zeros)
    for index, value in enumerate(values):
        expected_zero = (int(value) & -int(value)).bit_length() - 1 if value else 64
        assert counts[index] == binary.popcount_word(value) == int(value).bit_count()
        assert zeros[index] == binary.trailing_zeros(value) == expected_zero


def test_compiler_boundary() -> None:
    for definition in (binary._ctpop_definition, binary._cttz_definition):
        with pytest.raises(TypeError, match="uint64"):
            definition(None, types.int64)
    context = SimpleNamespace(get_constant=lambda kind, value: value)
    builder = SimpleNamespace(ctpop=lambda value: value, cttz=lambda value, zero_undef: value)
    for definition in (binary._ctpop_definition, binary._cttz_definition):
        signature, emit = definition(None, types.uint64)
        assert signature.return_type == types.uint64
        assert emit(context, builder, None, [np.uint64(7)]) == 7
    with mock.patch.object(binary, "_ctpop", return_value=np.uint64(3)):
        assert binary._overload_popcount(types.uint64)(np.uint64(7)) == 3
    with mock.patch.object(binary, "_cttz", return_value=np.uint64(2)):
        assert binary._overload_trailing_zeros(types.uint64)(np.uint64(4)) == 2


####################################################################################################
# packed binary elimination


@pytest.mark.parametrize("interpreted", [False, True])
def test_packed_elimination(interpreted: bool) -> None:
    """Match an independent byte-valued oracle for partial columns and multiword labels."""
    rng = np.random.default_rng(720)
    reduce = (
        getattr(binary.row_reduce_binary, "py_func", binary.row_reduce_binary)
        if interpreted
        else binary.row_reduce_binary
    )
    kernels = ("_pack_rows", "_unpack_rows")
    with (
        mock.patch.multiple(
            binary,
            **{
                name: getattr(getattr(binary, name), "py_func", getattr(binary, name))
                for name in kernels
            },
        )
        if interpreted
        else mock.patch.dict({}, {})
    ):
        for width in (0, 1, 7, 63, 64, 65, 130):
            for height in (0, 1, 3, 7):
                for label_width in (0, 1, 65):
                    matrix = rng.integers(0, 2, (height, width), dtype=np.uint8)
                    labels = rng.integers(0, 2, (height, label_width), dtype=np.uint8)
                    original, original_labels = matrix.copy(), labels.copy()
                    for columns in (
                        np.arange(width),
                        rng.permutation(width)[: width // 2],
                        np.tile(np.arange(width), 2),
                    ):
                        actual = reduce(matrix, labels, columns)
                        expected = _dense_elimination(matrix, labels, columns)
                        for result, reference in zip(actual, expected, strict=True):
                            np.testing.assert_array_equal(result, reference)
                    np.testing.assert_array_equal(matrix, original)
                    np.testing.assert_array_equal(labels, original_labels)
        with pytest.raises(ValueError, match="row counts"):
            reduce(np.zeros((2, 2), dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8), np.arange(2))
        for columns in (np.array([-1]), np.array([2])):
            with pytest.raises(IndexError, match="column"):
                reduce(np.eye(2, dtype=np.uint8), np.zeros((2, 0), dtype=np.uint8), columns)
        for matrix, labels in (
            (np.array([[2]], dtype=np.uint8), np.zeros((1, 0), dtype=np.uint8)),
            (np.eye(1, dtype=np.uint8), np.array([[2]], dtype=np.uint8)),
        ):
            with pytest.raises(ValueError, match="binary"):
                reduce(matrix, labels, np.arange(1))


@pytest.mark.parametrize("interpreted", [False, True])
def test_word_local_packing(interpreted: bool) -> None:
    """Check every word, including padding and strided inputs, against integer bit assembly."""
    pack = (
        getattr(binary._pack_rows, "py_func", binary._pack_rows)
        if interpreted
        else binary._pack_rows
    )
    unpack = (
        getattr(binary._unpack_rows, "py_func", binary._unpack_rows)
        if interpreted
        else binary._unpack_rows
    )
    rng = np.random.default_rng(821)
    for width in (0, 1, 63, 64, 65, 127, 128, 129):
        for height in (0, 1, 5):
            backing = rng.integers(0, 2, (2 * height, 2 * width), dtype=np.uint8)
            for matrix in (backing[::2, ::2], backing[::2, ::2][::-1, ::-1]):
                original = matrix.copy()
                result = pack(matrix)
                expected = np.empty((height, (width + 63) // 64), dtype=np.uint64)
                for row in range(height):
                    integer = sum(int(bit) << column for column, bit in enumerate(matrix[row]))
                    for word in range(expected.shape[1]):
                        expected[row, word] = (integer >> (64 * word)) & (2**64 - 1)
                np.testing.assert_array_equal(result, expected)
                np.testing.assert_array_equal(unpack(result, width), original)
                np.testing.assert_array_equal(matrix, original)
                assert not np.shares_memory(matrix, result)
    for dtype, invalid in ((np.int64, -1), (np.uint64, 2)):
        matrix = np.ones((2, 129), dtype=dtype)
        matrix[1, -1] = invalid
        with pytest.raises(ValueError, match="binary"):
            pack(matrix)


def test_packing_memory_failure() -> None:
    """Allocation failure is explicit rather than an uninitialized or partial packed result."""
    pack = getattr(binary._pack_rows, "py_func", binary._pack_rows)
    matrix = np.eye(2, dtype=np.uint8)
    with (
        mock.patch.object(binary.np, "empty", side_effect=MemoryError("packing allocation")),
        pytest.raises(MemoryError, match="packing allocation"),
    ):
        pack(matrix)


####################################################################################################
# independent reference helpers


def _dense_elimination(
    matrix: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    columns: npt.NDArray[np.int_],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.int_]]:
    rows, tags = matrix.copy(), labels.copy()
    pivots: list[int] = []
    for column in columns:
        rank = len(pivots)
        choices = np.flatnonzero(rows[rank:, column])
        if not len(choices):
            continue
        source = rank + int(choices[0])
        rows[[rank, source]] = rows[[source, rank]]
        tags[[rank, source]] = tags[[source, rank]]
        for other in range(len(rows)):
            if other != rank and rows[other, column]:
                rows[other] ^= rows[rank]
                tags[other] ^= tags[rank]
        pivots.append(int(column))
        if len(pivots) == len(rows):
            break
    return rows, tags, np.array(pivots, dtype=int)
