# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact-distance helpers in qldpc.codes.distance."""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y, z

import qldpc


def _bitwise_count(
    val: npt.ArrayLike, out: npt.NDArray[np.uint64] | None = None
) -> npt.NDArray[np.uint64]:
    """Return reference per-word bit counts for arrays accepted by optimized popcount helpers."""
    val = np.asarray(val)
    nbits = 8 * val.itemsize

    if out is None:
        out = np.empty_like(val)

    for indices in np.ndindex(val.shape):
        ival = int(val[indices])
        out[indices] = sum(ival >> i & 1 for i in range(nbits))

    return out


def test_hamming_weight() -> None:
    """Validate _hamming_weight against a simple reference bit-counting implementation."""
    vals = np.random.randint(0, 2**64, size=(7, 11), dtype=np.uint64)
    expected_weights = _bitwise_count(vals)

    weights = qldpc.codes.distance._hamming_weight(vals)
    np.testing.assert_array_equal(weights, expected_weights)

    buf, out = np.random.randint(0, 2**64, size=(2, *vals.shape), dtype=vals.dtype)
    weights = qldpc.codes.distance._hamming_weight(vals, buf=buf, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert out is weights


def test_symplectic_weight() -> None:
    """Validate _symplectic_weight against the reference bit-counting implementation."""
    vals = np.random.randint(0, 2**64, size=(7, 11), dtype=np.uint64)
    weights = qldpc.codes.distance._symplectic_weight(vals)
    expected_weights = _bitwise_count((vals | (vals >> np.uint64(1))) & 0x5555555555555555)
    np.testing.assert_array_equal(weights, expected_weights)

    buf, out = np.random.randint(0, 2**64, size=(2, *vals.shape), dtype=vals.dtype)
    weights = qldpc.codes.distance._symplectic_weight(vals, buf=buf, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert out is weights


def test_get_hamming_weight_fn() -> None:
    """_get_hamming_weight_fn selects NumPy bitcount when available and the fallback otherwise."""
    generators = np.random.randint(2, size=(4, 64), dtype=np.uint64)
    weight_fn, nbuf = qldpc.codes.distance._get_hamming_weight_fn()
    weights_default = weight_fn(generators)

    # The mocked NumPy backend uses a per-word popcount reference when NumPy lacks bitwise_count.
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    with mock.patch.object(np, "bitwise_count", wraps=mock_weight, create=True) as patched:
        weight_fn, nbuf = qldpc.codes.distance._get_hamming_weight_fn()
        assert weight_fn is not qldpc.codes.distance._hamming_weight
        assert nbuf == 0

        out = np.empty_like(generators)
        weights = weight_fn(generators, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out
        patched.assert_called_once()

    with mock.patch.object(np, "bitwise_count", None, create=True):
        weight_fn, nbuf = qldpc.codes.distance._get_hamming_weight_fn()
        assert weight_fn is qldpc.codes.distance._hamming_weight
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
    weight_fn, nbuf = qldpc.codes.distance._get_symplectic_weight_fn()
    weights_default = weight_fn(generators)

    # The mocked NumPy backend uses a per-word popcount reference when NumPy lacks bitwise_count.
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # This branch verifies the NumPy popcount backend for packed symplectic words.
    with mock.patch.object(np, "bitwise_count", wraps=mock_weight, create=True) as patched:
        weight_fn, nbuf = qldpc.codes.distance._get_symplectic_weight_fn()
        assert weight_fn is not qldpc.codes.distance._hamming_weight
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
        weight_fn, nbuf = qldpc.codes.distance._get_symplectic_weight_fn()
        assert weight_fn is qldpc.codes.distance._symplectic_weight
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
        assert qldpc.codes.distance._count_trailing_zeros(base_val << i) == i


@pytest.mark.parametrize("width", range(1, 8))
def test_inplace_rowsum(width: int) -> None:
    """_inplace_rowsum reduces each row to its sum and stores the result in the first column."""
    arr = np.random.randint(2**31, size=(10, width), dtype=np.uint64)
    expected = arr.sum(-1)
    actual = qldpc.codes.distance._inplace_rowsum(arr)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, arr[:, 0])


def test_rows_to_ints_endianness() -> None:
    """_rows_to_ints produces the same bit order as np.packbits."""
    # The packed-row helper stores each row's bits in the same per-byte order as np.packbits.
    bits = np.random.randint(2, size=(10, 120))
    ints = qldpc.codes.distance._rows_to_ints(bits, dtype=np.uint8)
    assert ints.shape == (10, 15)

    expected = np.packbits(bits).reshape(10, 15)
    np.testing.assert_array_equal(ints, expected)


@pytest.mark.parametrize("dtype", [int, np.uint64, np.uint8, np.int16])
def test_rows_to_ints(dtype: npt.DTypeLike) -> None:
    """_rows_to_ints packs binary rows into integer words of the requested dtype."""
    bits = np.random.randint(2, size=(10, 93))
    ints = qldpc.codes.distance._rows_to_ints(bits, dtype=dtype)

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
        qldpc.codes.distance._rows_to_ints(bits, dtype=dtype),
        ints.reshape(5, 1, 2, -1),
    )

    # The packed-row helper preserves nonpacked axes when packing a nonfinal bit axis.
    bits = bits.swapaxes(0, 3)
    np.testing.assert_array_equal(
        qldpc.codes.distance._rows_to_ints(bits, dtype=dtype, axis=0),
        ints.reshape(5, 1, 2, -1).swapaxes(0, 3),
    )

    bits = np.zeros((11, 0), dtype=dtype)
    ints = qldpc.codes.distance._rows_to_ints(bits, dtype=dtype)
    np.testing.assert_array_equal(
        qldpc.codes.distance._rows_to_ints(bits, dtype=dtype, axis=0), bits
    )


def _get_random_full_rank_matrix(
    rng: np.random.Generator, rows: int, cols: int
) -> npt.NDArray[np.uint8]:
    """Return a random full-row-rank binary matrix with the requested row and column counts."""
    while True:
        matrix = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)
        if len(qldpc.codes.distance._get_independent_rows(matrix)) == rows:
            return matrix


@pytest.fixture
def force_information_sets() -> Iterator[None]:
    """Route Brouwer-Zimmermann searches through information sets instead of exhaustive search."""
    with mock.patch.multiple(
        "qldpc.codes.distance",
        _exhaustive_is_cheaper=mock.Mock(return_value=False),
        _brute_force_is_cheaper=mock.Mock(return_value=False),
    ):
        yield


def test_binary_row_reduction_and_information_sets() -> None:
    """Binary elimination transforms quotient labels and retains the residual rank."""
    matrix = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    labels = np.eye(2, dtype=np.uint8)
    reduced, reduced_labels, pivots = qldpc.codes.distance._row_reduce_binary(matrix, labels)
    np.testing.assert_array_equal(reduced, [[1, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(reduced_labels, [[1, 1], [1, 0]])
    np.testing.assert_array_equal(pivots, [0, 1])

    basis = np.hstack(
        [
            np.eye(3, dtype=np.uint8),
            np.eye(3, dtype=np.uint8),
            np.array([[1, 0], [0, 1], [0, 0]], dtype=np.uint8),
        ]
    )
    information_sets = qldpc.codes.distance._get_information_set_generators(
        basis, np.eye(3, dtype=np.uint8)
    )
    assert [rank for _, _, rank, _ in information_sets] == [3, 3, 2]
    assert not qldpc.codes.distance._get_information_set_generators(
        np.zeros((1, 3), dtype=np.uint8), None
    )

    basis = np.array(
        [
            [1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    information_sets = qldpc.codes.distance._get_information_set_generators(basis, None)
    assert [rank for _, _, rank, _ in information_sets] == [3, 3]


@pytest.mark.parametrize("interpreted", [False, True])
def test_brouwer_zimmermann_scan(interpreted: bool) -> None:
    """Chunked kernel calls enumerate each combination of basis rows once.

    The chunked kernel calls preserve packed row XORs and multiword labels.
    """
    scan = qldpc.codes.distance._scan_brouwer_zimmermann
    if interpreted:
        scan = getattr(scan, "py_func", scan)
    rng = np.random.default_rng(120)

    # The scan input uses full-rank, partial-rank, and rank-zero information sets.
    # The last two information sets store labels only in their second packed word.
    for dimension in range(1, 7):
        generators = rng.integers(0, 2**64, (3, dimension, 5), dtype=np.uint64)
        labels = np.zeros((3, dimension, 2), dtype=np.uint64)
        labels[1:, :, 1] = rng.integers(0, 2, (2, dimension), dtype=np.uint64)
        ranks = np.array([dimension, dimension // 2, 0], dtype=np.int64)
        widths = np.array([5, 2, 1], dtype=np.int64)
        label_widths = np.array([0, 2, 2], dtype=np.int64)
        for weight in range(1, dimension + 1):
            support = np.zeros(weight, dtype=np.int64)
            prefixes = np.zeros((weight + 1, 3, 5), dtype=np.uint64)
            depth, best, expected = 0, 1000, 1000
            for indices in itertools.combinations(range(dimension), weight):
                # Update the expected running minimum from the current combination of basis rows.
                for index in range(3):
                    words = np.bitwise_xor.reduce(generators[index, indices, : widths[index]])
                    eligible = not label_widths[index] or np.any(
                        np.bitwise_xor.reduce(labels[index, indices], axis=0)
                    )
                    candidate = sum(int(word).bit_count() for word in words)
                    candidate += sum(row < ranks[index] for row in indices)
                    if eligible:
                        expected = min(expected, candidate)

                # Scan one basis-row combination and compare the running minimum and kernel state.
                best, depth, visited = scan(
                    generators,
                    labels,
                    ranks,
                    widths,
                    label_widths,
                    support,
                    prefixes,
                    depth,
                    best,
                    0,
                    1,
                )
                assert visited == 1 and best == expected
                # The last selected row index has advanced in preparation for the next chunk.
                np.testing.assert_array_equal(support[:-1], indices[:-1])
                assert support[-1] == indices[-1] + 1
                for index in range(3):
                    np.testing.assert_array_equal(
                        prefixes[weight, index, : widths[index]],
                        np.bitwise_xor.reduce(generators[index, indices, : widths[index]]),
                    )

            # One more kernel call reports completion after all basis-row combinations are visited.
            best, depth, visited = scan(
                generators,
                labels,
                ranks,
                widths,
                label_widths,
                support,
                prefixes,
                depth,
                best,
                0,
                1,
            )
            assert (best, depth, visited) == (expected, -1, 0)


def test_select_brouwer_zimmermann_information_sets() -> None:
    """Partial sets are retained only when they strengthen the active certificate."""
    generators = np.empty((10, 0), dtype=np.uint8)
    pivots = np.empty(0, dtype=int)
    information_sets = [
        qldpc.codes.distance._InformationSet(generators, None, rank, pivots) for rank in (10, 6, 2)
    ]
    selected = qldpc.codes.distance._select_brouwer_zimmermann_information_sets(
        information_sets,
        dimension=10,
        upper_bound=6,
        weight_divisor=1,
    )
    assert [rank for _, _, rank, _ in selected] == [10, 6]

    selected = qldpc.codes.distance._select_brouwer_zimmermann_information_sets(
        information_sets,
        dimension=10,
        upper_bound=12,
        weight_divisor=1,
    )
    assert [rank for _, _, rank, _ in selected] == [10, 6, 2]

    selected = qldpc.codes.distance._select_brouwer_zimmermann_information_sets(
        information_sets[1:],
        dimension=10,
        upper_bound=6,
        weight_divisor=1,
    )
    assert selected == information_sets[1:]


def test_brouwer_zimmermann_search_state() -> None:
    """A search state handles partial-rank information sets.

    The search state handles quotient labels, cutoffs, and lower-bound updates.
    """
    # The input is one rank-zero information set with two basis rows.
    # The two basis rows have Hamming weights 2 and 1, both outside the excluded subcode.
    search = qldpc.codes.distance._BrouwerZimmermannSearch(
        np.array([[[3], [2]]], dtype=np.uint64),
        np.ones((1, 2, 1), dtype=np.uint64),
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        np.array([1], dtype=np.int64),
        cutoff=1,
        batch_size=1,
        weight_divisor=1,
        best=3,
        lower_bound=0,
    )

    # An interrupted coefficient-weight layer leaves the lower-bound certificate unchanged.
    with (
        mock.patch("qldpc.codes.distance._scan_brouwer_zimmermann", side_effect=KeyboardInterrupt),
        pytest.raises(KeyboardInterrupt),
    ):
        search.advance()
    assert (search.lower_bound, search.next_weight) == (0, 1)

    # Reaching the distance cutoff returns before the layer updates the certificate.
    assert search.advance() == 1
    assert (search.lower_bound, search.next_weight) == (0, 1)
    assert search.advance() == 1

    # Without an effective distance cutoff, each call finalizes one coefficient-weight layer.
    search.cutoff = -(2**100)
    assert search.advance() == 1
    assert (search.lower_bound, search.next_weight) == (0, 2)
    assert search.advance() == 1
    assert (search.lower_bound, search.next_weight) == (1, 3)


def test_brouwer_zimmermann_setup_dispatches() -> None:
    """Brouwer-Zimmermann setup can dispatch to exhaustive search and shared cutoffs."""
    basis = np.array(
        [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    information_set = [qldpc.codes.distance._InformationSet(basis, None, 3, np.array([0, 1, 2]))]
    with (
        mock.patch("qldpc.codes.distance._exhaustive_is_cheaper", return_value=False),
        mock.patch(
            "qldpc.codes.distance._get_information_set_generators",
            return_value=information_set,
        ),
        mock.patch("qldpc.codes.distance._brute_force_is_cheaper", return_value=True),
        mock.patch(
            "qldpc.codes.distance._get_distance_quantum_brute_force",
            return_value=2,
        ),
    ):
        prepared = qldpc.codes.distance._prepare_brouwer_zimmermann_search(
            basis,
            None,
            cutoff=0,
            block_size=2,
            weight_divisor=1,
            upper_bound=4,
        )
    assert prepared == 2

    search = mock.Mock()
    search.finished = False
    search.advance.return_value = 1
    with (
        mock.patch(
            "qldpc.codes.distance._get_brouwer_zimmermann_initial_upper_bound",
            return_value=3,
        ),
        mock.patch(
            "qldpc.codes.distance._prepare_brouwer_zimmermann_search",
            return_value=search,
        ),
    ):
        distance = qldpc.codes.distance._get_distance_brouwer_zimmermann_many(
            [(basis, None)],
            cutoff=1,
            block_size=2,
        )
    assert distance == 1


def test_css_brouwer_zimmermann_special_inputs() -> None:
    """The joint CSS helper handles empty, trivial, and nontrivial quotient sectors."""
    empty = np.empty((0, 4), dtype=np.uint8)
    assert (
        qldpc.codes.get_distance_css_brouwer_zimmermann(
            [(empty, empty)],
            cutoff=0,
        )
        == 4
    )
    assert (
        qldpc.codes.get_distance_css_brouwer_zimmermann(
            [([[1, 0]], [[1, 0]])],
            cutoff=0,
        )
        == 0
    )

    with mock.patch(
        "qldpc.codes.distance._get_distance_brouwer_zimmermann_many",
        return_value=2,
    ) as mock_search:
        distance = qldpc.codes.get_distance_css_brouwer_zimmermann(
            [([[1, 1]], [])],
            cutoff=0,
            upper_bound=3,
        )
    assert distance == 2
    assert mock_search.call_args.kwargs["upper_bound"] == 3


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_random_cross_checks() -> None:
    """Information-set search agrees with exhaustive enumeration on random nested codes."""
    rng = np.random.default_rng(118)
    for length in range(5, 10):
        basis = _get_random_full_rank_matrix(rng, min(5, length - 1), length)
        expected = qldpc.codes.get_distance_classical(basis, cutoff=0, method="brute_force")
        assert qldpc.codes.get_distance_classical(basis, cutoff=0) == expected

        stabilizers, logical_ops = basis[:2], basis[2:]
        expected = qldpc.codes.get_distance_quantum(
            logical_ops, stabilizers, cutoff=0, homogeneous=True, method="brute_force"
        )
        assert (
            qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=0, homogeneous=True)
            == expected
        )

    for num_qubits in range(3, 7):
        basis = _get_random_full_rank_matrix(rng, min(6, 2 * num_qubits - 1), 2 * num_qubits)
        stabilizers, logical_ops = basis[:2], basis[2:]
        expected = qldpc.codes.get_distance_quantum(
            logical_ops, stabilizers, cutoff=0, method="brute_force"
        )
        assert qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=0) == expected


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_shared_search() -> None:
    """A shared Brouwer-Zimmermann search returns the minimum distance across nested codes."""
    rng = np.random.default_rng(119)
    basis_a = _get_random_full_rank_matrix(rng, 5, 8)
    basis_b = _get_random_full_rank_matrix(rng, 5, 9)
    labels_b = np.vstack(
        [
            np.zeros((2, 3), dtype=np.uint8),
            np.eye(3, dtype=np.uint8),
        ]
    )

    expected_a = qldpc.codes.get_distance_classical(
        basis_a,
        cutoff=0,
        method="brute_force",
    )
    expected_b = qldpc.codes.get_distance_quantum(
        basis_b[2:],
        basis_b[:2],
        cutoff=0,
        homogeneous=True,
        method="brute_force",
    )
    actual = qldpc.codes.distance._get_distance_brouwer_zimmermann_many(
        [(basis_a, None), (basis_b, labels_b)],
        cutoff=0,
        block_size=2,
    )
    assert actual == min(expected_a, expected_b)


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_information_set_search() -> None:
    """Information-set search excludes stabilizers, finds logical products, and honors cutoffs."""
    # The Hamming-weight-one stabilizer is lighter than every logical operator.
    # With block_size=0, some Brouwer-Zimmermann batches contain only that stabilizer.
    distance = qldpc.codes.get_distance_quantum(
        [[1, 1, 1, 1]], [[1, 0, 0, 0]], homogeneous=True, cutoff=0, block_size=0
    )
    assert distance == 3

    # The lightest logical operator is the product of both logical generators.
    distance = qldpc.codes.get_distance_quantum(
        [[1, 1, 1, 0], [1, 1, 0, 1]], [], homogeneous=True, cutoff=0
    )
    assert distance == 2

    # Every generator in every information set has Hamming weight at least three.
    # Certifying the weight-two codeword requires searching two-generator combinations.
    generators = [
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    ]
    assert qldpc.codes.get_distance_classical(generators, cutoff=0) == 2

    # Both reduced generators have Hamming weight three.
    # Their weight-two sum reaches the distance cutoff.
    assert qldpc.codes.get_distance_classical([[1, 0, 1, 1], [0, 1, 1, 1]], cutoff=2) == 2

    # Row reduction exposes a Hamming-weight-one generator before any search runs.
    assert qldpc.codes.get_distance_classical([[1, 1, 0], [0, 1, 1], [1, 1, 1]], cutoff=1) == 1


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_without_certifying_bound() -> None:
    """Brouwer-Zimmermann returns its best codeword after exhausting every coefficient weight."""
    with mock.patch("qldpc.codes.distance._get_information_set_generators", return_value=[]):
        assert qldpc.codes.get_distance_classical(np.eye(2, dtype=np.uint8), cutoff=0) == 1


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_subsystem_code() -> None:
    """Information-set search treats the gauge operators of a subsystem code as trivial."""
    code = qldpc.codes.QuditCode(qldpc.codes.BaconShorCode(3).matrix, is_subsystem_code=True)
    assert code.get_distance_exact() == 3


@pytest.mark.usefixtures("force_information_sets")
@pytest.mark.parametrize("length", [64, 65, 257])
def test_brouwer_zimmermann_packed_widths(length: int) -> None:
    """Information-set weights are exact across uint64 words and for weights above 255."""
    generators = np.ones((1, length), dtype=np.uint8)
    assert qldpc.codes.get_distance_classical(generators, cutoff=0) == length


def test_brouwer_zimmermann_exhaustive_dispatch() -> None:
    """Exhaustive enumeration is chosen when estimated to be cheaper than information sets."""
    assert qldpc.codes.distance._exhaustive_is_cheaper(20, 21)
    assert qldpc.codes.distance._exhaustive_is_cheaper(21, 4000)
    assert not qldpc.codes.distance._exhaustive_is_cheaper(21, 22)
    assert qldpc.codes.distance._brute_force_is_cheaper(
        dimension=4, num_logical_rows=2, ranks=[4], upper_bound=4, weight_divisor=1
    )
    assert not qldpc.codes.distance._brute_force_is_cheaper(
        dimension=4, num_logical_rows=4, ranks=[4], upper_bound=4, weight_divisor=1
    )
    assert not qldpc.codes.distance._brute_force_is_cheaper(
        dimension=20, num_logical_rows=20, ranks=[20, 20], upper_bound=3, weight_divisor=2
    )

    # Dependent logical rows skip the top-level independent-row exhaustive path.
    # The exhaustive dispatcher uses the reduced quotient basis instead.
    distance = qldpc.codes.get_distance_quantum(
        [[1, 1, 1], [1, 1, 1]], [], homogeneous=True, cutoff=0
    )
    assert distance == 3

    # After information-set setup, _brute_force_is_cheaper can dispatch to exhaustive enumeration.
    with mock.patch("qldpc.codes.distance._exhaustive_is_cheaper", return_value=False):
        assert qldpc.codes.get_distance_classical(qldpc.codes.HammingCode(3).generator) == 3


def test_brouwer_zimmermann_special_inputs() -> None:
    """Brouwer-Zimmermann validates matrix shapes and preserves quotient edge cases."""
    assert qldpc.codes.get_distance_classical(np.empty((0, 4), dtype=np.uint8), cutoff=0) == 4
    assert (
        qldpc.codes.get_distance_quantum(
            np.empty((0, 4), dtype=np.uint8),
            np.empty((0, 4), dtype=np.uint8),
            homogeneous=True,
        )
        == 4
    )
    assert qldpc.codes.get_distance_quantum([[1, 0]], [[1, 0]], homogeneous=True) == 0

    with pytest.raises(ValueError, match="incompatible widths"):
        qldpc.codes.get_distance_quantum([[1, 0, 0]], [[1, 0]], homogeneous=True)
    with pytest.raises(ValueError, match="even number of columns"):
        qldpc.codes.get_distance_quantum([[1, 0, 0]], [], homogeneous=False)
    with pytest.raises(ValueError, match="even number of columns"):
        qldpc.codes.get_distance_quantum(
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint8),
            homogeneous=False,
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        qldpc.codes.get_distance_classical(np.zeros((1, 1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="infer the width"):
        qldpc.codes.get_distance_classical([])


def test_brouwer_zimmermann_reduces_enumeration_work() -> None:
    """BZ examines far fewer candidates than brute force for Hamming(5).

    BZ also examines far fewer candidates for the even-weight extension of Hamming(5).
    """
    generators = qldpc.codes.HammingCode(5).generator
    num_candidates = 0
    original_scan = qldpc.codes.distance._scan_brouwer_zimmermann

    def counting_scan(*args: Any) -> tuple[int, int, int]:
        nonlocal num_candidates
        best, depth, visited = original_scan(*args)
        num_candidates += visited * len(args[2])
        return best, depth, visited

    def counting_weight(
        arr: npt.NDArray[np.uint64],
        buf: npt.NDArray[np.uint64] | None = None,
        out: npt.NDArray[np.uint64] | None = None,
    ) -> npt.NDArray[np.uint64]:
        nonlocal num_candidates
        num_candidates += len(arr)
        return qldpc.codes.distance._hamming_weight(arr, buf=buf, out=out)

    with (
        mock.patch(
            "qldpc.codes.distance._get_hamming_weight_fn", return_value=(counting_weight, 1)
        ),
        mock.patch("qldpc.codes.distance._scan_brouwer_zimmermann", side_effect=counting_scan),
    ):
        assert qldpc.codes.get_distance_classical(generators, cutoff=0) == 3
        assert 1000 * num_candidates < 2 ** len(generators)
        extended = np.column_stack([generators, generators.sum(axis=1)])
        num_candidates = 0
        assert qldpc.codes.get_distance_classical(extended, cutoff=0) == 4
    assert num_candidates < len(generators) ** 2


def test_brouwer_zimmermann_initial_upper_bound_excludes_stabilizers() -> None:
    """Exclude stabilizer rows before packing candidate witnesses."""
    helpers = qldpc.codes.distance
    basis = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    labels = np.array([[0], [1]], dtype=np.uint8)
    with mock.patch.object(helpers, "_rows_to_ints", wraps=helpers._rows_to_ints) as pack:
        assert helpers._get_brouwer_zimmermann_initial_upper_bound(basis, labels) == 2
    np.testing.assert_array_equal(pack.call_args.args[0], basis[1:])


def test_brouwer_zimmermann_overlap_certificates() -> None:
    """Check pivot coverage and label transformations against exhaustive codeword weights."""
    helpers = qldpc.codes.distance

    # Check the coverage formula independently of the family-selection heuristic.
    assert (
        helpers._get_brouwer_zimmermann_lower_bound(
            41, [41] * 3, completed_weight=4, weight_divisor=2, max_coverage=2
        )
        == 8
    )
    assert (
        helpers._get_brouwer_zimmermann_lower_bound(
            3, [3] * 4, completed_weight=2, weight_divisor=1, max_coverage=4
        )
        == 3
    )

    # Give each basis row a distinct label so its transformed coefficients remain observable.
    basis = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis = helpers._get_independent_rows(basis)
    labels = np.eye(len(basis), dtype=np.uint8)
    information_sets = helpers._get_information_set_generators(basis, labels)
    with mock.patch.object(
        helpers, "_row_reduce_binary", wraps=helpers._row_reduce_binary
    ) as reduce:
        selected = helpers._get_overlapping_information_sets(
            basis, labels, information_sets, upper_bound=3, weight_divisor=1, minimum_work=0
        )
    assert selected is not None
    family, coverage = selected
    loads = np.bincount(np.concatenate([item.pivots for item in family]), minlength=basis.shape[1])
    assert coverage == loads.max() and coverage > 1

    # This fixture reaches ideal coverage without duplicate attempts or unused extra bases.
    assert reduce.call_count == len(family) - 1
    for item in family:
        assert item.labels is not None
        np.testing.assert_array_equal(item.generators, item.labels @ basis % 2)
        np.testing.assert_array_equal(item.generators[:, item.pivots], np.eye(len(basis)))

    # Exhaust all words still unsearched in every basis and check their physical-weight bound.
    for bits in itertools.product((0, 1), repeat=len(basis)):
        word = np.asarray(bits) @ basis % 2
        for completed in (1, 2):
            if all(np.count_nonzero(word[item.pivots]) > completed for item in family):
                assert np.count_nonzero(word) >= helpers._get_brouwer_zimmermann_lower_bound(
                    len(basis),
                    [item.rank for item in family],
                    completed_weight=completed,
                    weight_divisor=1,
                    max_coverage=coverage,
                )


def test_brouwer_zimmermann_overlap_setup_guards() -> None:
    """Reject unhelpful coverage and terminate repeated attempts at the same pivot set."""
    helpers = qldpc.codes.distance

    # Nearly complete disjoint coverage leaves no cheaper family within the size budget.
    basis = np.hstack([np.eye(25, dtype=np.uint8)] * 4)
    shortened = basis[:, :-1]
    assert (
        helpers._get_overlapping_information_sets(
            shortened,
            None,
            helpers._get_information_set_generators(shortened, None),
            upper_bound=26,
            weight_divisor=1,
        )
        is None
    )

    # Repeatedly rediscovering the same pivots must terminate without selecting a family.
    basis = basis[:, :40]
    first = helpers._get_information_set_generators(basis, None)[0]
    with mock.patch.object(
        helpers,
        "_row_reduce_binary",
        return_value=(first.generators, first.labels, first.pivots),
    ):
        assert (
            helpers._get_overlapping_information_sets(
                basis, None, [first], upper_bound=8, weight_divisor=1
            )
            is None
        )


@pytest.mark.parametrize("cutoff", [0, 2])
def test_brouwer_zimmermann_overlap_setup_certification(cutoff: int) -> None:
    """A replacement basis can meet the cutoff or close the lower bound during setup."""
    helpers = qldpc.codes.distance
    basis = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    generators, labels, pivots = helpers._row_reduce_binary(basis, columns=np.array([0, 2]))
    original = helpers._InformationSet(basis, None, 2, np.array([0, 1]))
    alternative = helpers._InformationSet(generators, labels, 2, pivots)
    with (
        mock.patch.object(helpers, "_exhaustive_is_cheaper", return_value=False),
        mock.patch.object(helpers, "_get_information_set_generators", return_value=[original]),
        mock.patch.object(
            helpers, "_get_overlapping_information_sets", return_value=([alternative], 1)
        ),
        mock.patch.object(helpers, "_brute_force_is_cheaper", side_effect=AssertionError),
    ):
        assert qldpc.codes.get_distance_classical(basis, cutoff=cutoff) == 2


def test_brouwer_zimmermann_overlap_qc() -> None:
    """The default selector must certify QC64 with substantially less enumeration."""
    code = qldpc.codes.QCCode(
        {x: 4, y: 4, z: 2},
        1 + x + y + x * y * z,
        1 + x * y + x**2 * y**3 * z + x**3 * y**2,
    )
    helpers = qldpc.codes.distance
    original_scan = helpers._scan_brouwer_zimmermann
    evaluated = 0

    def scan(*args: Any) -> tuple[int, int, int]:
        nonlocal evaluated
        best, depth, visited = original_scan(*args)
        evaluated += visited * len(args[2])
        return best, depth, visited

    with mock.patch.object(helpers, "_scan_brouwer_zimmermann", side_effect=scan):
        assert code.forget_distance().get_distance_exact("X", cutoff=0) == 8
    assert 0 < evaluated < 1_000_000


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_overlap_improves_quotient_witness() -> None:
    """A real overlap search improves its row witnesses and matches exhaustive distance."""
    helpers = qldpc.codes.distance

    # Shorten a systematic BCH(31,21) basis to [26,16] and isolate a one-bit quotient.
    basis = helpers._get_independent_rows(
        np.asarray(qldpc.codes.BCHCode(31, 21).generator, dtype=np.uint8)
    )[5:, 5:]
    logicals = basis[13:14]
    stabilizers = np.delete(basis, 13, axis=0)
    expected = qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=True, cutoff=0, method="brute_force"
    )

    # Keep real family construction but bypass its large-work threshold for this small oracle.
    original = helpers._get_overlapping_information_sets

    def select(
        *args: Any, **kwargs: Any
    ) -> tuple[list[qldpc.codes.distance._InformationSet], int] | None:
        selected = original(*args, **kwargs, minimum_work=0)
        assert selected is not None and selected[1] > 1
        assert (
            min(
                helpers._get_brouwer_zimmermann_initial_upper_bound(item.generators, item.labels)
                for item in selected[0]
            )
            > expected
        )
        return selected

    with mock.patch.object(
        helpers, "_get_overlapping_information_sets", side_effect=select
    ) as select_mock:
        assert (
            qldpc.codes.get_distance_quantum(logicals, stabilizers, homogeneous=True, cutoff=0)
            == expected
            == 5
        )
    select_mock.assert_called()


@pytest.mark.parametrize("block_size", [1, 3, 13])
def test_get_distance_classical(block_size: int) -> None:
    """get_distance_classical visits every nontrivial XOR combination of generators exactly once."""
    generators = np.random.randint(2, size=(9, 137))

    # The mocked Hamming-weight function records every packed nonzero generator combination once.
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_hamming_weight(
        arr: npt.NDArray[np.uint64],
        buf: npt.NDArray[np.uint64] | None = None,
        out: npt.NDArray[np.uint64] | None = None,
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._hamming_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance._get_hamming_weight_fn", return_value=(_mock_hamming_weight, 0)
    ):
        distance = qldpc.codes.distance.get_distance_classical(
            generators, block_size=block_size, method="brute_force"
        )

    int_generators = qldpc.codes.distance._rows_to_ints(generators)
    expected_bitstrings = [
        tuple(np.bitwise_xor.reduce(np.vstack(gens)).tolist())
        for n in range(1, len(generators) + 1)
        for gens in itertools.combinations(int_generators, n)
    ]

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(expected_bitstrings)

    observed_array = np.array(observed_bitstrings, dtype=np.uint64)
    expected_distance = _bitwise_count(observed_array).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("block_size", [1, 3, 13])
def test_get_distance_quantum(block_size: int) -> None:
    """get_distance_quantum visits every stabilizer + logical-op combination exactly once.

    Uses Hamming (homogeneous) weight mode.
    """
    stabilizers = np.random.randint(2, size=(8, 97))
    logical_ops = np.random.randint(2, size=(5, 97))

    # The mocked Hamming-weight function records every nonzero logical combination combined
    # with every stabilizer combination, including the empty one.
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_hamming_weight(
        arr: npt.NDArray[np.uint64],
        buf: npt.NDArray[np.uint64] | None = None,
        out: npt.NDArray[np.uint64] | None = None,
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._hamming_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance._get_hamming_weight_fn", return_value=(_mock_hamming_weight, 0)
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops,
            stabilizers,
            homogeneous=True,
            block_size=block_size,
            method="brute_force",
        )

    int_stabilizers = qldpc.codes.distance._rows_to_ints(stabilizers)
    int_logical_ops = qldpc.codes.distance._rows_to_ints(logical_ops)
    expected_bitstrings = [
        tuple(np.bitwise_xor.reduce(np.vstack(stabs + ops)).tolist())
        for ns in range(len(stabilizers) + 1)
        for stabs in itertools.combinations(int_stabilizers, ns)
        for nl in range(1, len(logical_ops) + 1)
        for ops in itertools.combinations(int_logical_ops, nl)
    ]

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(expected_bitstrings)

    observed_array = np.array(observed_bitstrings, dtype=np.uint64)
    expected_distance = _bitwise_count(observed_array).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("block_size", [1, 3, 13])
def test_get_distance_quantum_symplectic(block_size: int) -> None:
    """get_distance_quantum visits every stabilizer + logical-op combination exactly once.

    Uses symplectic weight mode.
    """
    stabilizers = np.random.randint(2, size=(3, 98))
    logical_ops = np.random.randint(2, size=(5, 98))

    # The mocked symplectic-weight function records every nonzero logical combination combined
    # with every stabilizer combination, including the empty one.
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_symplectic_weight(
        arr: npt.NDArray[np.uint64],
        buf: npt.NDArray[np.uint64] | None = None,
        out: npt.NDArray[np.uint64] | None = None,
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._symplectic_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance._get_symplectic_weight_fn", return_value=(_mock_symplectic_weight, 0)
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops,
            stabilizers,
            homogeneous=False,
            block_size=block_size,
            method="brute_force",
        )

    int_stabilizers = qldpc.codes.distance._rows_to_ints(qldpc.codes.distance._riffle(stabilizers))
    int_logical_ops = qldpc.codes.distance._rows_to_ints(qldpc.codes.distance._riffle(logical_ops))
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


def test_get_distance_classical_methods() -> None:
    """get_distance_classical dispatches to NumPy bitcount or the fallback as available."""
    generators = np.random.randint(2, size=(6, 56), dtype=np.uint64)
    distance_default = qldpc.codes.distance.get_distance_classical(generators, block_size=3)

    # The mocked NumPy backend uses a per-word popcount reference when NumPy lacks bitwise_count.
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # This branch verifies the NumPy popcount backend for packed Hamming words.
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._hamming_weight",
            wraps=qldpc.codes.distance._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_classical(generators, block_size=3)
        bitcount.assert_called()
        fallback.assert_not_called()
        assert distance == distance_default

    # This branch verifies the fallback popcount backend for packed Hamming words.
    with (
        mock.patch("numpy.bitwise_count", None, create=True),
        mock.patch(
            "qldpc.codes.distance._hamming_weight",
            wraps=qldpc.codes.distance._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_classical(generators, block_size=3)
        fallback.assert_called()
        assert distance == distance_default


def test_distance_backend_validation() -> None:
    """Distance backend and exact-method selectors are restricted to documented values."""
    for backend in ["auto", "decoder", "gap", "sqetch"]:
        qldpc.codes.validate_distance_backend(backend)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown distance backend"):
        qldpc.codes.validate_distance_backend("other")  # type: ignore[arg-type]

    for method in ["brouwer_zimmermann", "brute_force"]:
        qldpc.codes.validate_distance_method(method)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown distance method"):
        qldpc.codes.validate_distance_method("other")  # type: ignore[arg-type]


def test_get_distance_empty_stabilizers_symplectic() -> None:
    """A symplectic-weight distance with no stabilizers returns the min logical operator weight.

    With an empty ``stabilizers`` argument there is nothing to interleave, and the X/Z riffling
    still handles the empty input.
    """
    # The symplectic vector [1, 0 | 0, 0] is an X operator on qubit 0 with symplectic weight 1.
    distance = qldpc.codes.distance.get_distance_quantum(
        [[1, 0, 0, 0]], [], homogeneous=False, method="brute_force"
    )
    assert distance == 1


def test_get_distance_small_block_size() -> None:
    """A block_size smaller than the packed word count still yields the correct distance.

    When the columns span more than ``64 * (block_size + 1)`` bits, no operators are vectorized and
    the calculation proceeds by the sequential sweep alone.
    """
    generators = np.zeros((2, 200), dtype=np.uint64)
    generators[0] = 1  # First generator has Hamming weight 200.
    generators[1, :100] = 1  # Second generator has Hamming weight 100.
    assert qldpc.codes.distance.get_distance_classical(generators, block_size=1) == 100


def test_cutoff_early_exit() -> None:
    """cutoff makes get_distance_* return early once an operator of weight <= cutoff is found.

    Exercises all three early-exit return paths: the pre-loop vectorized block, the logical-op
    sweep, and the nested stabilizer sweep.
    """
    # The pre-loop vectorized block contains a codeword with Hamming weight 1.
    assert (
        qldpc.codes.distance.get_distance_classical([[1, 0, 0, 0]], cutoff=1, method="brute_force")
        == 1
    )

    # The logical-operator sweep with block_size=1 finds a generator with Hamming weight 1.
    generators = [[1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1]]
    assert (
        qldpc.codes.distance.get_distance_classical(
            generators, block_size=1, cutoff=1, method="brute_force"
        )
        == 1
    )

    # The stabilizer sweep finds a Hamming-weight-1 operator only after XORing a stabilizer row.
    logical_ops = [[0, 1, 1, 1, 1]]
    stabilizers = [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]
    distance = qldpc.codes.distance.get_distance_quantum(
        logical_ops,
        stabilizers,
        block_size=1,
        cutoff=1,
        homogeneous=True,
        method="brute_force",
    )
    assert distance == 1


def test_get_distance_quantum_methods() -> None:
    """Homogeneous get_distance_quantum dispatches to NumPy bitcount or the fallback."""
    stabilizers = np.random.randint(2, size=(4, 56), dtype=np.uint64)
    logical_ops = np.random.randint(2, size=(3, 56), dtype=np.uint64)
    distance_default = qldpc.codes.distance.get_distance_quantum(
        logical_ops, stabilizers, block_size=3, homogeneous=True
    )

    # The mocked NumPy backend uses a per-word popcount reference when NumPy lacks bitwise_count.
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # This branch verifies the NumPy popcount backend for homogeneous quantum packed words.
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._hamming_weight",
            wraps=qldpc.codes.distance._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, block_size=3, homogeneous=True
        )
        bitcount.assert_called()
        fallback.assert_not_called()
        assert distance == distance_default

    # This branch verifies the fallback popcount backend for homogeneous quantum packed words.
    with (
        mock.patch("numpy.bitwise_count", None, create=True),
        mock.patch(
            "qldpc.codes.distance._hamming_weight",
            wraps=qldpc.codes.distance._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, block_size=3, homogeneous=True
        )
        fallback.assert_called()
        assert distance == distance_default


def test_get_distance_quantum_methods_symplectic() -> None:
    """Symplectic get_distance_quantum dispatches to NumPy bitcount or the fallback."""
    stabilizers = np.random.randint(2, size=(4, 56), dtype=np.uint64)
    logical_ops = np.random.randint(2, size=(3, 56), dtype=np.uint64)
    distance_default = qldpc.codes.distance.get_distance_quantum(
        logical_ops,
        stabilizers,
        block_size=3,
        homogeneous=False,
        method="brute_force",
    )

    # The mocked NumPy backend uses a per-word popcount reference when NumPy lacks bitwise_count.
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # This branch verifies the NumPy popcount backend for symplectic quantum packed words.
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._symplectic_weight",
            wraps=qldpc.codes.distance._symplectic_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops,
            stabilizers,
            block_size=3,
            homogeneous=False,
            method="brute_force",
        )
        bitcount.assert_called()
        fallback.assert_not_called()
        assert distance == distance_default

    # This branch verifies the fallback popcount backend for symplectic quantum packed words.
    with (
        mock.patch("numpy.bitwise_count", None, create=True),
        mock.patch(
            "qldpc.codes.distance._symplectic_weight",
            wraps=qldpc.codes.distance._symplectic_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops,
            stabilizers,
            block_size=3,
            homogeneous=False,
            method="brute_force",
        )
        fallback.assert_called()
        assert distance == distance_default


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
    """Classical distance matches known values for Hamming, repetition, and ring codes."""
    distance = qldpc.codes.distance.get_distance_classical(code.generator)
    assert distance == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance


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
    """Quantum distance matches known values for C4, C6, Steane, and small surface/toric codes."""
    distance_x = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(qldpc.objects.Pauli.X),
        code.get_stabilizer_ops(qldpc.objects.Pauli.X),
        homogeneous=True,
    )
    distance_z = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(qldpc.objects.Pauli.Z),
        code.get_stabilizer_ops(qldpc.objects.Pauli.Z),
        homogeneous=True,
    )
    distance_all = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=True,
    )
    assert min(distance_x, distance_z, distance_all) == expected_distance

    # CSSCode.forget_distance clears the X-sector and Z-sector distance caches before recomputation.
    # Assigning only CSSCode._distance would leave cached sector distances available to the method.
    code.forget_distance()
    assert code.get_distance_exact() == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.quantum.FiveQubitCode(), 3),
    ],
)
def test_get_distance_quantum_noncss_codes(
    code: qldpc.codes.QuditCode, expected_distance: int
) -> None:
    """Quantum distance matches known values for small non-CSS codes."""
    distance = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=False,
    )
    assert distance == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance


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
