# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared binary algebra, packing and weight helpers."""

from __future__ import annotations

import contextlib
import itertools
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
from llvmlite import ir
from numba import njit, types
from numba.core.registry import cpu_target

import qldpc

from . import _binary as binary
from .conftest import _bitwise_count


def test_binary_matrices_and_nested_bases() -> None:
    """Check matrix conversion and basis ownership independently of distance dispatch."""
    assert binary._as_binary_matrix(np.empty((0, 3))).shape == (0, 3)
    assert binary._as_binary_matrix([], num_cols=3).shape == (0, 3)
    with pytest.raises(ValueError, match="infer the width"):
        binary._as_binary_matrix([])
    with pytest.raises(ValueError, match="two-dimensional"):
        binary._as_binary_matrix(np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="incompatible widths"):
        binary._as_binary_matrix([[1, 0]], num_cols=3)

    matrix = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.uint8)
    labels = np.eye(4, dtype=np.uint8)
    reduced, transformed, pivots = binary._row_reduce_binary(matrix, labels)
    np.testing.assert_array_equal(pivots, [0, 1])
    assert transformed is not None
    np.testing.assert_array_equal(transformed @ matrix % 2, reduced)
    np.testing.assert_array_equal(labels, np.eye(4, dtype=np.uint8))
    np.testing.assert_array_equal(binary._get_independent_rows(matrix), [[1, 0, 1], [0, 1, 1]])
    basis, tags = binary._get_nested_code_basis(matrix, np.array([[1, 0, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(basis, [[1, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(tags, [[0], [1]])


@pytest.mark.parametrize("shape", [(0, 4), (3, 7), (5, 5), (4, 70)])
def test_get_kernel(shape: tuple[int, int]) -> None:
    """The kernel basis is independent, annihilated by the matrix, and of full nullity."""
    matrix = np.random.default_rng(shape[1]).integers(0, 2, shape, dtype=np.uint8)
    matrix[-1:] = matrix[:1]
    kernel = binary._get_kernel(matrix)
    rank = len(binary._get_independent_rows(matrix))
    assert kernel.dtype == np.uint8 and kernel.shape == (shape[1] - rank, shape[1])
    assert not np.any(matrix.astype(int) @ kernel.T % 2)
    assert len(binary._get_independent_rows(kernel)) == len(kernel)


def test_packed_row_weights_do_not_overflow() -> None:
    packed = np.full((2, 5), np.iinfo(np.uint64).max, dtype=np.uint64)
    packed[1, -1] = 0
    weight_func, _ = binary._get_hamming_weight_fn()
    np.testing.assert_array_equal(binary._get_packed_row_weights(packed, weight_func), [320, 256])


def test_riffle_and_symplectic_isometry() -> None:
    symplectic = np.array(list(itertools.product((0, 1), repeat=6)), dtype=np.uint8)
    expected = np.stack((symplectic[:, :3], symplectic[:, 3:]), axis=2).reshape(-1, 6)
    np.testing.assert_array_equal(binary._riffle(symplectic), expected)
    assert np.shape(binary._riffle(np.empty((0, 6)))) == (0, 6)
    with pytest.raises(AssertionError):
        binary._riffle(np.zeros((1, 3)))
    transformed = binary._symplectic_to_hamming(symplectic)
    np.testing.assert_array_equal(
        np.count_nonzero(transformed, axis=1),
        2 * np.count_nonzero(symplectic[:, :3] | symplectic[:, 3:], axis=1),
    )


def test_hamming_weight() -> None:
    """Validate _hamming_weight against a simple reference bit-counting implementation."""
    vals = np.random.randint(0, 2**64, size=(7, 11), dtype=np.uint64)
    expected_weights = _bitwise_count(vals)

    weights = binary._hamming_weight(vals)
    np.testing.assert_array_equal(weights, expected_weights)

    buf, out = np.random.randint(0, 2**64, size=(2, *vals.shape), dtype=vals.dtype)
    weights = binary._hamming_weight(vals, buf=buf, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert out is weights


def test_symplectic_weight() -> None:
    """Validate _symplectic_weight against the reference bit-counting implementation."""
    vals = np.random.randint(0, 2**64, size=(7, 11), dtype=np.uint64)
    weights = binary._symplectic_weight(vals)
    expected_weights = _bitwise_count((vals | (vals >> np.uint64(1))) & 0x5555555555555555)
    np.testing.assert_array_equal(weights, expected_weights)

    buf, out = np.random.randint(0, 2**64, size=(2, *vals.shape), dtype=vals.dtype)
    weights = binary._symplectic_weight(vals, buf=buf, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert out is weights


def test_get_hamming_weight_fn() -> None:
    """_get_hamming_weight_fn selects NumPy bitcount when available and the fallback otherwise."""
    generators = np.random.randint(2, size=(4, 64), dtype=np.uint64)
    weight_fn, nbuf = binary._get_hamming_weight_fn()
    weights_default = weight_fn(generators)

    # The mocked NumPy backend uses a per-word popcount reference when NumPy lacks bitwise_count.
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    with mock.patch.object(np, "bitwise_count", wraps=mock_weight, create=True) as patched:
        weight_fn, nbuf = binary._get_hamming_weight_fn()
        assert weight_fn is not binary._hamming_weight
        assert nbuf == 0

        out = np.empty_like(generators)
        weights = weight_fn(generators, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out
        patched.assert_called_once()

    with mock.patch.object(np, "bitwise_count", None, create=True):
        weight_fn, nbuf = binary._get_hamming_weight_fn()
        assert weight_fn is binary._hamming_weight
        assert nbuf == 1

        out = np.empty_like(generators)
        weights = weight_fn(generators, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out

        buf = np.empty_like(generators)
        weights = weight_fn(generators, buf, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out


def test_get_symplectic_weight_fn() -> None:
    """_get_symplectic_weight_fn uses NumPy bitcount when available and the fallback otherwise."""
    generators = np.random.randint(2, size=(4, 56), dtype=np.uint64)
    weight_fn, nbuf = binary._get_symplectic_weight_fn()
    weights_default = weight_fn(generators)

    # The mocked NumPy backend uses a per-word popcount reference when NumPy lacks bitwise_count.
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # This branch verifies the NumPy popcount backend for packed symplectic words.
    with mock.patch.object(np, "bitwise_count", wraps=mock_weight, create=True) as patched:
        weight_fn, nbuf = binary._get_symplectic_weight_fn()
        assert weight_fn is not binary._hamming_weight
        assert nbuf == 1

        out = np.empty_like(generators)
        weights = weight_fn(generators, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out
        patched.assert_called_once()

        buf = np.empty_like(generators)
        weights = weight_fn(generators, buf, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out

    # This branch verifies the fallback popcount backend for packed symplectic words.
    with mock.patch.object(np, "bitwise_count", None, create=True):
        weight_fn, nbuf = binary._get_symplectic_weight_fn()
        assert weight_fn is binary._symplectic_weight
        assert nbuf == 1

        out = np.empty_like(generators)
        weights = weight_fn(generators, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out

        buf = np.empty_like(generators)
        weights = weight_fn(generators, buf, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out


@pytest.mark.parametrize(
    "base_val",
    [1, 2**64 - 1, int(np.random.randint(2**64, dtype=np.uint64)) | 1],
)
def test_count_trailing_zeros(base_val: int) -> None:
    """_count_trailing_zeros(val << i) == i for several base values across all 128 shift amounts."""
    for i in range(128):
        assert binary._count_trailing_zeros(base_val << i) == i


@pytest.mark.parametrize("width", range(1, 8))
def test_inplace_rowsum(width: int) -> None:
    """_inplace_rowsum reduces each row to its sum and stores the result in the first column."""
    arr = np.random.randint(2**31, size=(10, width), dtype=np.uint64)
    expected = arr.sum(-1)
    actual = binary._inplace_rowsum(arr)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, arr[:, 0])


def test_distance_requires_binary_input() -> None:
    """Distance calculations reject non-binary input rather than silently truncating it.

    The enumeration packs each row into the bits of uint64 words, so a non-binary entry would
    otherwise be reinterpreted and yield a wrong distance with no indication of a problem.
    """
    # A GF(3) generator matrix is rejected because exact distance enumeration supports only GF(2).
    ternary_code = qldpc.codes.classical.HammingCode(3, 3)
    with pytest.raises(ValueError, match=r"only support binary codes.*GF\(3\)"):
        qldpc.codes.distance.get_distance_classical(ternary_code.generator)

    # A plain integer array with a nonbinary entry is rejected by value validation.
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        qldpc.codes.distance.get_distance_classical(np.array([[1, 2, 0], [0, 1, 1]]))
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        qldpc.codes.distance.get_distance_quantum(
            np.array([[1, 1, 0, 0]]), np.array([[0, 0, 2, 0]]), homogeneous=True
        )

    # A binary generator matrix is accepted with either a plain integer dtype or a GF(2) dtype.
    assert qldpc.codes.distance.get_distance_classical(np.array([[1, 1, 1]])) == 3
    assert qldpc.codes.distance.get_distance_classical(qldpc.codes.HammingCode(4).generator) == 3


####################################################################################################
# native bit counting, packing and row reduction


@njit
def _popcounts(values: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
    counts = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        counts[index] = binary._popcount_word(value)
    return counts


def test_popcount_word() -> None:
    rng = np.random.default_rng(78)
    values = np.concatenate(
        [
            np.array([0, 1, 2**63, 2**64 - 1], dtype=np.uint64),
            rng.integers(0, 2**64, 100, np.uint64),
        ]
    )
    expected = [int(value).bit_count() for value in values]
    np.testing.assert_array_equal(_popcounts(values), expected)
    # The Python implementation calls the Python fallback, and lets coverage trace the helper.
    np.testing.assert_array_equal(_popcounts.py_func(values), expected)


def test_popcount_lowering() -> None:
    """Exercise the compiler extension directly, since coverage cannot trace compiled code."""
    typer = binary._type_popcount_word(cpu_target.typing_context)
    assert typer(types.uint64) == types.int64
    assert typer(types.int64) is None
    word = ir.IntType(64)
    function = ir.Function(ir.Module(), ir.FunctionType(word, [word]), "popcount")
    builder = ir.IRBuilder(function.append_basic_block())
    arguments: list[ir.Value] = [function.args[0]]
    signature = types.int64(types.uint64)
    lowered = binary._lower_popcount_word(cpu_target.target_context, builder, signature, arguments)
    assert "llvm.ctpop.i64" in str(lowered)


@pytest.mark.parametrize("interpreted", [False, True])
def test_row_reduction_matches_dense_elimination(interpreted: bool) -> None:
    """Compare with byte-valued elimination for partial, repeated and multiword columns."""
    kernels = ("_pack_bits", "_unpack_bits", "_row_reduce_packed")
    rng = np.random.default_rng(720)
    with (
        mock.patch.multiple(binary, **{name: getattr(binary, name).py_func for name in kernels})
        if interpreted
        else contextlib.nullcontext()
    ):
        for width in (0, 1, 7, 64, 65, 130):
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
                        actual = binary._row_reduce_binary(matrix, labels, columns=columns)
                        expected = _dense_elimination(matrix, labels, columns)
                        for result, reference in zip(actual, expected, strict=True):
                            np.testing.assert_array_equal(result, reference)
                    np.testing.assert_array_equal(matrix, original)
                    np.testing.assert_array_equal(labels, original_labels)
                    packed = binary._pack_bits(matrix)
                    for row in range(height):
                        integer = sum(int(bit) << column for column, bit in enumerate(matrix[row]))
                        words = [
                            (integer >> (64 * word)) & (2**64 - 1)
                            for word in range(len(packed[row]))
                        ]
                        np.testing.assert_array_equal(packed[row], words)
                    np.testing.assert_array_equal(binary._unpack_bits(packed, width), matrix)
    for columns in (np.array([-1]), np.array([2])):
        with pytest.raises(IndexError, match="out of range"):
            binary._row_reduce_binary(np.eye(2, dtype=np.uint8), columns=columns)


def _dense_elimination(
    matrix: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    columns: npt.NDArray[np.int_],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.int_]]:
    """Reference elimination on byte-valued rows."""
    rows, tags = matrix.copy(), labels.copy()
    pivots: list[int] = []
    for column in columns:
        rank = len(pivots)
        if rank == len(rows):
            break
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
    return rows, tags, np.array(pivots, dtype=int)
