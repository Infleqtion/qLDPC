# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact-distance helpers in qldpc.codes.distance."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y, z

import qldpc

from . import _binary as binary
from . import brouwer_zimmermann as bz
from .conftest import _get_random_full_rank_matrix


def test_binary_row_reduction_and_information_sets() -> None:
    """Binary elimination transforms quotient labels and retains the residual rank."""
    matrix = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    labels = np.eye(2, dtype=np.uint8)
    reduced, reduced_labels, pivots = binary._row_reduce_binary(matrix, labels)
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
    information_sets = bz._get_information_set_generators(basis, np.eye(3, dtype=np.uint8))
    assert [rank for _, _, rank, _ in information_sets] == [3, 3, 2]
    assert not bz._get_information_set_generators(np.zeros((1, 3), dtype=np.uint8), None)

    basis = np.array(
        [
            [1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    information_sets = bz._get_information_set_generators(basis, None)
    assert [rank for _, _, rank, _ in information_sets] == [3, 3]


@pytest.mark.parametrize("interpreted", [False, True])
def test_brouwer_zimmermann_scan(interpreted: bool) -> None:
    """Chunked kernel calls enumerate each combination of basis rows once.

    The chunked kernel calls preserve packed row XORs and multiword labels.
    """
    scan = bz._scan_brouwer_zimmermann
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


@pytest.mark.parametrize("ranks, upper_bound", [([10, 6, 2], 6), ([10, 6, 2], 12), ([6, 2], 6)])
def test_select_brouwer_zimmermann_information_sets(ranks: list[int], upper_bound: int) -> None:
    """Selecting search bases does not increase estimated enumeration work."""
    generators = np.empty((10, 0), dtype=np.uint8)
    pivots = np.empty(0, dtype=int)
    information_sets = [bz._InformationSet(generators, None, rank, pivots) for rank in ranks]
    selected = bz._select_brouwer_zimmermann_information_sets(
        information_sets,
        dimension=10,
        upper_bound=upper_bound,
        weight_divisor=1,
    )
    assert selected
    assert bz._get_brouwer_zimmermann_work(
        10,
        [item.rank for item in selected],
        upper_bound=upper_bound,
        weight_divisor=1,
    ) <= bz._get_brouwer_zimmermann_work(
        10,
        ranks,
        upper_bound=upper_bound,
        weight_divisor=1,
    )


def test_brouwer_zimmermann_search_state() -> None:
    """A search state handles partial-rank information sets.

    The search state handles quotient labels, cutoffs, and lower-bound updates.
    """
    # The input is one rank-zero information set with two basis rows.
    # The two basis rows have Hamming weights 2 and 1, both outside the excluded subcode.
    search = bz._BrouwerZimmermannSearch(
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
        mock.patch(
            "qldpc.codes.distance.brouwer_zimmermann._scan_brouwer_zimmermann",
            side_effect=KeyboardInterrupt,
        ),
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
    information_set = [bz._InformationSet(basis, None, 3, np.array([0, 1, 2]))]
    with (
        mock.patch(
            "qldpc.codes.distance.brouwer_zimmermann._exhaustive_is_cheaper", return_value=False
        ),
        mock.patch(
            "qldpc.codes.distance.brouwer_zimmermann._get_information_set_generators",
            return_value=information_set,
        ),
        mock.patch(
            "qldpc.codes.distance.brouwer_zimmermann._brute_force_is_cheaper", return_value=True
        ),
        mock.patch(
            "qldpc.codes.distance.brouwer_zimmermann._get_distance_quantum_brute_force",
            return_value=2,
        ),
    ):
        prepared = bz._prepare_brouwer_zimmermann_search(
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
            "qldpc.codes.distance.brouwer_zimmermann._get_brouwer_zimmermann_initial_upper_bound",
            return_value=3,
        ),
        mock.patch(
            "qldpc.codes.distance.brouwer_zimmermann._prepare_brouwer_zimmermann_search",
            return_value=search,
        ),
    ):
        distance = bz._get_distance_brouwer_zimmermann_many(
            [(basis, None)],
            cutoff=1,
            block_size=2,
        )
    assert distance == 1


def test_brouwer_zimmermann_setup_cutoffs() -> None:
    """Preparation can reach the cutoff from initial rows or a newly systematic row."""
    basis = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    assert (
        bz._prepare_brouwer_zimmermann_search(
            basis, None, cutoff=3, block_size=2, weight_divisor=1, upper_bound=4
        )
        == 3
    )
    with mock.patch.object(bz, "_exhaustive_is_cheaper", return_value=False):
        assert (
            bz._prepare_brouwer_zimmermann_search(
                basis, None, cutoff=2, block_size=2, weight_divisor=1, upper_bound=4
            )
            == 2
        )


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
    actual = bz._get_distance_brouwer_zimmermann_many(
        [(basis_a, None), (basis_b, labels_b)],
        cutoff=0,
        block_size=2,
    )
    assert actual == min(expected_a, expected_b)


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_sector_certificates() -> None:
    """A completed sector's certificate must not finish a different sector's search."""
    generators = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis_a = binary._get_independent_rows(
        np.column_stack([generators, np.bitwise_xor.reduce(generators, axis=1)])
    )
    basis_b = binary._get_independent_rows(
        np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
    )
    expected = [
        qldpc.codes.get_distance_classical(basis, cutoff=0, method="brute_force")
        for basis in (basis_a, basis_b)
    ]
    assert expected == [4, 2]

    searches: list[bz._BrouwerZimmermannSearch] = []
    advanced: set[int] = set()
    original_prepare = bz._prepare_brouwer_zimmermann_search
    original_advance = bz._BrouwerZimmermannSearch.advance

    def prepare(*args: Any, **kwargs: Any) -> int | bz._BrouwerZimmermannSearch:
        result = original_prepare(*args, **kwargs)
        if isinstance(result, bz._BrouwerZimmermannSearch):
            searches.append(result)
        return result

    def advance(search: bz._BrouwerZimmermannSearch) -> int:
        advanced.add(id(search))
        return original_advance(search)

    # Exercise real layer advancement, rather than two results certified during preparation.
    with (
        mock.patch.object(bz, "_prepare_brouwer_zimmermann_search", side_effect=prepare),
        mock.patch.object(bz._BrouwerZimmermannSearch, "advance", advance),
    ):
        actual = bz._get_distance_brouwer_zimmermann_many(
            [(basis_a, None), (basis_b, None)], cutoff=0, block_size=2
        )
    assert actual == min(expected)
    assert len(searches) == 2
    assert {id(search) for search in searches} <= advanced


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_without_certifying_bound() -> None:
    """Brouwer-Zimmermann returns its best codeword after exhausting every coefficient weight."""
    with mock.patch(
        "qldpc.codes.distance.brouwer_zimmermann._get_information_set_generators", return_value=[]
    ):
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
    assert bz._exhaustive_is_cheaper(20, 21)
    assert bz._exhaustive_is_cheaper(21, 4000)
    assert not bz._exhaustive_is_cheaper(21, 22)
    assert bz._brute_force_is_cheaper(
        dimension=4, num_logical_rows=2, ranks=[4], upper_bound=4, weight_divisor=1
    )
    assert not bz._brute_force_is_cheaper(
        dimension=4, num_logical_rows=4, ranks=[4], upper_bound=4, weight_divisor=1
    )
    assert not bz._brute_force_is_cheaper(
        dimension=20, num_logical_rows=20, ranks=[20, 20], upper_bound=3, weight_divisor=2
    )

    # Dependent logical rows skip the top-level independent-row exhaustive path.
    # The exhaustive dispatcher uses the reduced quotient basis instead.
    distance = qldpc.codes.get_distance_quantum(
        [[1, 1, 1], [1, 1, 1]], [], homogeneous=True, cutoff=0
    )
    assert distance == 3

    # After information-set setup, _brute_force_is_cheaper can dispatch to exhaustive enumeration.
    with mock.patch(
        "qldpc.codes.distance.brouwer_zimmermann._exhaustive_is_cheaper", return_value=False
    ):
        assert qldpc.codes.get_distance_classical(qldpc.codes.HammingCode(3).generator) == 3


@pytest.mark.parametrize("extended", [False, True])
def test_brouwer_zimmermann_reduces_enumeration_work(
    extended: bool, get_candidate_count: Callable[[], int]
) -> None:
    """Hamming(5) and its even-weight extension need far less work than exhaustive search."""
    generators = np.asarray(qldpc.codes.HammingCode(5).generator, dtype=np.uint8)
    if extended:
        generators = np.column_stack([generators, np.bitwise_xor.reduce(generators, axis=1)])
    assert qldpc.codes.get_distance_classical(generators, cutoff=0) == (4 if extended else 3)
    assert 1000 * get_candidate_count() < 2 ** len(generators)


def test_brouwer_zimmermann_initial_upper_bound_excludes_stabilizers() -> None:
    """The initial upper bound comes from eligible rows, not lighter stabilizers."""
    basis = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    labels = np.array([[0], [1]], dtype=np.uint8)
    assert bz._get_brouwer_zimmermann_initial_upper_bound(basis, labels) == 2


def test_brouwer_zimmermann_overlap_certificates() -> None:
    """Check pivot coverage and label transformations against exhaustive codeword weights."""

    # Label each row separately so its row operations can be checked.
    basis = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis = binary._get_independent_rows(basis)
    labels = np.eye(len(basis), dtype=np.uint8)
    information_sets = bz._get_information_set_generators(basis, labels)
    selected = bz._get_overlapping_information_sets(
        basis, labels, information_sets, upper_bound=3, weight_divisor=1, minimum_work=0
    )
    assert selected is not None
    family, coverage = selected
    loads = np.bincount(np.concatenate([item.pivots for item in family]), minlength=basis.shape[1])
    assert coverage >= loads.max() > 1
    for item in family:
        assert item.labels is not None
        np.testing.assert_array_equal(item.generators, item.labels @ basis % 2)
        np.testing.assert_array_equal(item.generators[:, item.pivots], np.eye(len(basis)))

    # Check the Hamming-weight bound for every word still unsearched in all bases.
    for bits in itertools.product((0, 1), repeat=len(basis)):
        word = np.asarray(bits) @ basis % 2
        for completed in (1, 2):
            if all(np.count_nonzero(word[item.pivots]) > completed for item in family):
                assert np.count_nonzero(word) >= bz._get_brouwer_zimmermann_lower_bound(
                    len(basis),
                    [item.rank for item in family],
                    completed_weight=completed,
                    weight_divisor=1,
                    max_coverage=coverage,
                )


def test_brouwer_zimmermann_overlap_setup_guards() -> None:
    """Reject unhelpful coverage and terminate repeated attempts at the same pivot set."""

    # Nearly complete disjoint coverage leaves no cheaper family within the size budget.
    basis = np.hstack([np.eye(25, dtype=np.uint8)] * 4)
    shortened = basis[:, :-1]
    assert (
        bz._get_overlapping_information_sets(
            shortened,
            None,
            bz._get_information_set_generators(shortened, None),
            upper_bound=26,
            weight_divisor=1,
        )
        is None
    )

    # Repeatedly rediscovering the same pivots must terminate without selecting a family.
    basis = basis[:, :40]
    first = bz._get_information_set_generators(basis, None)[0]
    with mock.patch.object(
        bz,
        "_row_reduce_binary",
        return_value=(first.generators, first.labels, first.pivots),
    ):
        assert (
            bz._get_overlapping_information_sets(
                basis, None, [first], upper_bound=8, weight_divisor=1
            )
            is None
        )


@pytest.mark.parametrize("cutoff", [0, 2])
def test_brouwer_zimmermann_overlap_setup_certification(cutoff: int) -> None:
    """A replacement basis can meet the cutoff or close the lower bound during setup."""
    basis = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    generators, labels, pivots = binary._row_reduce_binary(basis, columns=np.array([0, 2]))
    original = bz._InformationSet(basis, None, 2, np.array([0, 1]))
    alternative = bz._InformationSet(generators, labels, 2, pivots)
    with (
        mock.patch.object(bz, "_exhaustive_is_cheaper", return_value=False),
        mock.patch.object(bz, "_get_information_set_generators", return_value=[original]),
        mock.patch.object(bz, "_get_overlapping_information_sets", return_value=([alternative], 1)),
        mock.patch.object(bz, "_brute_force_is_cheaper", side_effect=AssertionError),
    ):
        assert qldpc.codes.get_distance_classical(basis, cutoff=cutoff) == 2


def test_brouwer_zimmermann_overlap_qc(get_candidate_count: Callable[[], int]) -> None:
    """The default QC64 X-sector search stays within a broad candidate-work budget."""
    code = qldpc.codes.QCCode(
        {x: 4, y: 4, z: 2},
        1 + x + y + x * y * z,
        1 + x * y + x**2 * y**3 * z + x**3 * y**2,
    )
    assert code.forget_distance().get_distance_exact("X", cutoff=0) == 8
    assert get_candidate_count() < 1_000_000


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_overlap_improves_quotient_witness() -> None:
    """Compare witness-improving overlap enumeration with an exhaustive quotient oracle."""

    # Shorten a systematic BCH(31,21) basis to [26,16] and isolate a one-bit quotient.
    basis = binary._get_independent_rows(
        np.asarray(qldpc.codes.BCHCode(31, 21).generator, dtype=np.uint8)
    )[5:, 5:]
    logicals = basis[13:14]
    stabilizers = np.delete(basis, 13, axis=0)
    expected = qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=True, cutoff=0, method="brute_force"
    )

    # Lower only the work threshold so this small oracle exercises production family construction.
    original = bz._get_overlapping_information_sets

    def select(*args: Any, **kwargs: Any) -> tuple[list[bz._InformationSet], int] | None:
        selected = original(*args, **kwargs, minimum_work=0)
        assert selected is not None and selected[1] > 1
        assert (
            min(
                bz._get_brouwer_zimmermann_initial_upper_bound(item.generators, item.labels)
                for item in selected[0]
            )
            > expected
        )
        return selected

    with mock.patch.object(
        bz, "_get_overlapping_information_sets", side_effect=select
    ) as select_mock:
        assert (
            qldpc.codes.get_distance_quantum(logicals, stabilizers, homogeneous=True, cutoff=0)
            == expected
            == 5
        )
    select_mock.assert_called()


@pytest.fixture
def get_candidate_count(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Count row-weight evaluations and scanned supports across BZ and exhaustive routes."""
    num_candidates = 0
    original_scan = bz._scan_brouwer_zimmermann

    def counting_scan(*args: Any) -> tuple[int, int, int]:
        nonlocal num_candidates
        best, depth, visited = original_scan(*args)
        num_candidates += visited * len(args[2])
        return best, depth, visited

    bitwise_count = np.bitwise_count

    def counting_bitwise_count(
        arr: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
    ) -> npt.NDArray[np.uint64]:
        nonlocal num_candidates
        num_candidates += len(arr)
        return bitwise_count(arr, out=out)

    # Row weights and exhaustive enumeration count every packed word with np.bitwise_count.
    monkeypatch.setattr(np, "bitwise_count", counting_bitwise_count)
    monkeypatch.setattr(bz, "_scan_brouwer_zimmermann", counting_scan)
    return lambda: num_candidates
