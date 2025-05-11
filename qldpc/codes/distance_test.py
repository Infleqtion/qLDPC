from __future__ import annotations

import importlib
import itertools
from collections.abc import Iterable
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

import qldpc


def _bitwise_count(val: npt.ArrayLike, out: object = None) -> npt.NDArray[np.uint]:
    """Simplistic implementation of `bitwise_count` used to validate optimized variants."""
    if isinstance(val, Iterable):
        return np.array(list(map(_bitwise_count, val)))

    _ = out
    val = np.asarray(val)
    nbits = 8 * val.itemsize
    return ((val >> np.arange(nbits, dtype=val.dtype)) & 1).sum()


def test_hamming_weight() -> None:
    vals = np.random.randint(0, 2**64, size=(7, 11), dtype=np.uint)
    weights = qldpc.codes.distance._hamming_weight(vals)
    expected_weights = _bitwise_count(vals)
    np.testing.assert_array_equal(weights, expected_weights)

    buf, out = np.random.randint(0, 2**64, size=(2, *vals.shape), dtype=vals.dtype)
    weights = qldpc.codes.distance.hamming_weight(vals, buf=buf, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert out is weights


def test_symplectic_weight() -> None:
    vals = np.random.randint(0, 2**64, size=(7, 11), dtype=np.uint)
    weights = qldpc.codes.distance._symplectic_weight(vals)
    expected_weights = _bitwise_count((vals | (vals >> np.uint(1))) & 0x5555555555555555)
    np.testing.assert_array_equal(weights, expected_weights)

    buf, out = np.random.randint(0, 2**64, size=(2, *vals.shape), dtype=vals.dtype)
    weights = qldpc.codes.distance.symplectic_weight(vals, buf=buf, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert out is weights


def test_function_selection() -> None:
    stabilizers = np.random.randint(2, size=(4, 56), dtype=np.uint)
    logical_ops = np.random.randint(2, size=(3, 56), dtype=np.uint)

    weights = qldpc.codes.distance.hamming_weight(stabilizers)
    symplectic_weights = qldpc.codes.distance.symplectic_weight(stabilizers)
    dist = qldpc.codes.distance.get_distance_quantum(
        logical_ops, stabilizers, block_size=3, homogeneous=True
    )
    symplectic_dist = qldpc.codes.distance.get_distance_quantum(
        logical_ops, stabilizers, block_size=3, homogeneous=False
    )

    try:
        with mock.patch("numpy.bitwise_count", qldpc.codes.distance._hamming_weight, create=True):
            importlib.reload(qldpc.codes.distance)

            assert qldpc.codes.distance.hamming_weight is not qldpc.codes.distance._hamming_weight
            assert (
                qldpc.codes.distance.symplectic_weight
                is not qldpc.codes.distance._symplectic_weight
            )

            np.testing.assert_array_equal(qldpc.codes.distance.hamming_weight(stabilizers), weights)
            np.testing.assert_array_equal(
                qldpc.codes.distance.symplectic_weight(stabilizers), symplectic_weights
            )
            assert dist == qldpc.codes.distance.get_distance_quantum(
                logical_ops, stabilizers, block_size=3, homogeneous=True
            )
            assert symplectic_dist == qldpc.codes.distance.get_distance_quantum(
                logical_ops, stabilizers, block_size=3, homogeneous=False
            )

        with mock.patch("numpy.bitwise_count", None, create=True):
            importlib.reload(qldpc.codes.distance)

            assert qldpc.codes.distance.hamming_weight is qldpc.codes.distance._hamming_weight
            assert qldpc.codes.distance.symplectic_weight is qldpc.codes.distance._symplectic_weight

            np.testing.assert_array_equal(qldpc.codes.distance.hamming_weight(stabilizers), weights)
            np.testing.assert_array_equal(
                qldpc.codes.distance.symplectic_weight(stabilizers), symplectic_weights
            )
            assert dist == qldpc.codes.distance.get_distance_quantum(
                logical_ops, stabilizers, block_size=3, homogeneous=True
            )
            assert symplectic_dist == qldpc.codes.distance.get_distance_quantum(
                logical_ops, stabilizers, block_size=3, homogeneous=False
            )

    finally:
        importlib.reload(qldpc.codes.distance)


@pytest.mark.parametrize(
    "base_val",
    (
        1,
        2**64 - 1,
        int(np.random.randint(2**63)) | 1,
        int(np.random.randint(2**64, dtype=np.uint)) | 1,
    ),
)
def test_count_trailing_zeros(base_val: int | np.integer[npt.NBitBase]) -> None:
    for i in range(64):
        assert qldpc.codes.distance.count_trailing_zeros(base_val << i) == i


def test_rows_to_ints_endianness() -> None:
    # Compare bit order to that used by `np.packbits`
    bits = np.random.randint(2, size=(10, 120))
    ints = qldpc.codes.distance.rows_to_ints(bits, dtype=np.uint8)
    assert ints.shape == (10, 15)

    expected = np.packbits(bits).reshape(10, 15)
    np.testing.assert_array_equal(ints, expected)


@pytest.mark.parametrize("dtype", [int, np.uint, np.uint8, np.int16])
def test_rows_to_ints(dtype: npt.DTypeLike) -> None:
    bits = np.random.randint(2, size=(10, 93))
    ints = qldpc.codes.distance.rows_to_ints(bits, dtype=dtype)

    nbits = 8 * np.dtype(dtype).itemsize
    expected_words_per_row = int(np.ceil(93 / nbits))
    assert ints.shape == (10, expected_words_per_row)
    assert ints.dtype == np.dtype(dtype)

    np.testing.assert_array_equal(_bitwise_count(ints).sum(-1), bits.sum(-1))

    for indices in np.ndindex(ints.shape):
        i = indices[-1] * nbits
        packed_bits = bits[*indices[:-1]][i : i + nbits]
        bitstr = "".join(map(str, packed_bits))
        assert np.binary_repr(ints[*indices], len(bitstr)) == bitstr

    # Pack array with more dimensions
    bits = bits.reshape(5, 1, 2, -1)
    np.testing.assert_array_equal(
        qldpc.codes.distance.rows_to_ints(bits, dtype=dtype),
        ints.reshape(5, 1, 2, -1),
    )

    # Pack along a different axis
    bits = bits.swapaxes(0, 3)
    np.testing.assert_array_equal(
        qldpc.codes.distance.rows_to_ints(bits, dtype=dtype, axis=0),
        ints.reshape(5, 1, 2, -1).swapaxes(0, 3),
    )

    bits = np.zeros((11, 0), dtype=dtype)
    ints = qldpc.codes.distance.rows_to_ints(bits, dtype=dtype)
    np.testing.assert_array_equal(
        qldpc.codes.distance.rows_to_ints(bits, dtype=dtype, axis=0), bits
    )


@pytest.mark.parametrize("block_size", range(1, 14))
def test_get_distance_classical(block_size: int) -> None:
    generators = np.random.randint(2, size=(9, 137))

    # Intercept `hamming_weight` calls to check that every nontrivial combination of generators
    # is observed exactly once
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_hamming_weight(
        arr: npt.NDArray[np.uint],
        buf: npt.NDArray[np.uint] | None = None,
        out: npt.NDArray[np.uint] | None = None,
    ) -> npt.NDArray[np.uint]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._hamming_weight(arr, buf=buf, out=out)

    with mock.patch("qldpc.codes.distance.hamming_weight", _mock_hamming_weight):
        distance = qldpc.codes.distance.get_distance_classical(generators, block_size=block_size)

    int_generators = qldpc.codes.distance.rows_to_ints(generators)
    expected_bitstrings = [
        tuple(np.bitwise_xor.reduce(np.vstack(gens)).tolist())
        for n in range(1, len(generators) + 1)
        for gens in itertools.combinations(int_generators, n)
    ]

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(expected_bitstrings)

    expected_distance = _bitwise_count(observed_bitstrings).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("block_size", range(1, 14))
def test_get_distance_quantum(block_size: int) -> None:
    stabilizers = np.random.randint(2, size=(8, 97))
    logical_ops = np.random.randint(2, size=(5, 97))

    # Intercept `hamming_weight` calls to check that every combination of stabilizers and at least
    # one logical op is observed exactly once
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_hamming_weight(
        arr: npt.NDArray[np.uint],
        buf: npt.NDArray[np.uint] | None = None,
        out: npt.NDArray[np.uint] | None = None,
    ) -> npt.NDArray[np.uint]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._hamming_weight(arr, buf=buf, out=out)

    with mock.patch("qldpc.codes.distance.hamming_weight", _mock_hamming_weight):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, homogeneous=True, block_size=block_size
        )

    int_stabilizers = qldpc.codes.distance.rows_to_ints(stabilizers)
    int_logical_ops = qldpc.codes.distance.rows_to_ints(logical_ops)
    expected_bitstrings = [
        tuple(np.bitwise_xor.reduce(np.vstack(stabs + ops)).tolist())
        for ns in range(len(stabilizers) + 1)
        for stabs in itertools.combinations(int_stabilizers, ns)
        for nl in range(1, len(logical_ops) + 1)
        for ops in itertools.combinations(int_logical_ops, nl)
    ]

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(expected_bitstrings)

    expected_distance = _bitwise_count(observed_bitstrings).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("block_size", range(1, 14))
def test_get_distance_quantum_symplectic(block_size: int) -> None:
    stabilizers = np.random.randint(2, size=(3, 98))
    logical_ops = np.random.randint(2, size=(5, 98))

    # Intercept `symplectic_weight` calls to check that every combination of stabilizers and at
    # least one logical op is observed exactly once
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_symplectic_weight(
        arr: npt.NDArray[np.uint],
        buf: npt.NDArray[np.uint] | None = None,
        out: npt.NDArray[np.uint] | None = None,
    ) -> npt.NDArray[np.uint]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._symplectic_weight(arr, buf=buf, out=out)

    with mock.patch("qldpc.codes.distance.symplectic_weight", _mock_symplectic_weight):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, homogeneous=False, block_size=block_size
        )

    int_stabilizers = qldpc.codes.distance.rows_to_ints_symplectic(stabilizers)
    int_logical_ops = qldpc.codes.distance.rows_to_ints_symplectic(logical_ops)
    expected_bitstrings = np.array(
        [
            np.bitwise_xor.reduce(np.vstack(stabs + ops))
            for ns in range(len(stabilizers) + 1)
            for stabs in itertools.combinations(int_stabilizers, ns)
            for nl in range(1, len(logical_ops) + 1)
            for ops in itertools.combinations(int_logical_ops, nl)
        ]
    )

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(map(tuple, expected_bitstrings))

    vals = (expected_bitstrings | (expected_bitstrings >> 1)) & 0x5555555555555555
    expected_distance = _bitwise_count(vals).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.classical.HammingCode(4), 3),
        (qldpc.codes.classical.RepetitionCode(3), 3),
        (qldpc.codes.classical.RepetitionCode(8), 8),
        (qldpc.codes.classical.RingCode(8), 8),
    ],
)
def test_get_distance_classical_known_codes(
    code: qldpc.codes.ClassicalCode, expected_distance: int
) -> None:
    distance = qldpc.codes.distance.get_distance_classical(code.generator)
    assert distance == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.quantum.C4Code(), 2),
        (qldpc.codes.quantum.C6Code(), 2),
        (qldpc.codes.quantum.SteaneCode(), 3),
        (qldpc.codes.quantum.SurfaceCode(3), 3),
        (qldpc.codes.quantum.SurfaceCode(4), 4),
        (qldpc.codes.quantum.ToricCode(4), 4),
    ],
)
def test_get_distance_quantum_css_codes(code: qldpc.codes.CSSCode, expected_distance: int) -> None:
    distance_x = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(qldpc.math.Pauli.X),
        code.get_stabilizer_ops(qldpc.math.Pauli.X),
        homogeneous=True,
    )
    distance_z = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(qldpc.math.Pauli.Z),
        code.get_stabilizer_ops(qldpc.math.Pauli.Z),
        homogeneous=True,
    )
    distance_all = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=True,
    )
    assert min(distance_x, distance_z, distance_all) == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.quantum.FiveQubitCode(), 3),
    ],
)
def test_get_distance_quantum_noncss_codes(
    code: qldpc.codes.QuditCode, expected_distance: int
) -> None:
    distance = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=False,
    )
    assert distance == expected_distance
