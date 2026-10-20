# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared binary algebra, packing and weight helpers."""

from __future__ import annotations

import itertools
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

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


def test_rows_to_ints_endianness() -> None:
    """_rows_to_ints produces the same bit order as np.packbits."""
    # The packed-row helper stores each row's bits in the same per-byte order as np.packbits.
    bits = np.random.randint(2, size=(10, 120))
    ints = binary._rows_to_ints(bits, dtype=np.uint8)
    assert ints.shape == (10, 15)

    expected = np.packbits(bits).reshape(10, 15)
    np.testing.assert_array_equal(ints, expected)


@pytest.mark.parametrize("dtype", [int, np.uint64, np.uint8, np.int16])
def test_rows_to_ints(dtype: npt.DTypeLike) -> None:
    """_rows_to_ints packs binary rows into integer words of the requested dtype."""
    bits = np.random.randint(2, size=(10, 93))
    ints = binary._rows_to_ints(bits, dtype=dtype)

    nbits = 8 * np.dtype(dtype).itemsize
    expected_words_per_row = int(np.ceil(93 / nbits))
    assert ints.shape == (10, expected_words_per_row)
    assert ints.dtype == np.dtype(dtype)

    np.testing.assert_array_equal(_bitwise_count(ints).sum(-1), bits.sum(-1))

    for indices in np.ndindex(ints.shape):
        i = indices[-1] * nbits
        packed_bits = bits[indices[:-1]][i : i + nbits]
        bitstr = "".join(map(str, packed_bits))
        assert np.binary_repr(ints[indices], len(bitstr)) == bitstr

    # The packed-row helper preserves leading array axes while packing the last bit axis.
    bits = bits.reshape(5, 1, 2, -1)
    np.testing.assert_array_equal(
        binary._rows_to_ints(bits, dtype=dtype),
        ints.reshape(5, 1, 2, -1),
    )

    # The packed-row helper preserves nonpacked axes when packing a nonfinal bit axis.
    bits = bits.swapaxes(0, 3)
    np.testing.assert_array_equal(
        binary._rows_to_ints(bits, dtype=dtype, axis=0),
        ints.reshape(5, 1, 2, -1).swapaxes(0, 3),
    )

    bits = np.zeros((11, 0), dtype=dtype)
    ints = binary._rows_to_ints(bits, dtype=dtype)
    np.testing.assert_array_equal(binary._rows_to_ints(bits, dtype=dtype, axis=0), bits)


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
