# SPDX-License-Identifier: Apache-2.0

"""Tests for bounded kernel enumeration and exact Pauli-coset weight decisions."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pytest
import stim

from ._bounded_weight_search import (
    BoundedKernelSearch,
    PauliCosetSearch,
    _iter_fixed_weight_supports,
    _stream_fixed_weight_supports,
)

Metric = Literal["pauli", "css"]

# A 10-vertex graph state with a syndrome (1019) whose minimum weight is 3, found by searching
# thresholds in increasing order for a recomputation that once discarded a proved bound.
GRAPH_10_EDGES = (
    (0, 2), (0, 3), (0, 6), (0, 7), (1, 8), (2, 5), (2, 8), (3, 4),
    (3, 6), (3, 7), (4, 6), (4, 7), (4, 9), (5, 6), (6, 8), (7, 8),
)  # fmt: skip


def _graph_state(
    num_qubits: int, edges: Sequence[tuple[int, int]], phase: bool = False
) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] of a graph state, with S gates on every qubit if ``phase``.

    Generator j acts as X (or Y) on qubit j, so the X parts of the generators form the identity.
    """
    circuit = stim.Circuit()
    circuit.append("H", range(num_qubits))
    for edge in edges:
        circuit.append("CZ", edge)
    if phase:
        circuit.append("S", range(num_qubits))
    generators = stim.Tableau.from_circuit(circuit).to_stabilizers()
    return np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)


def _class_weights(
    matrix: npt.NDArray[np.uint8], syndrome: int, metric: Metric
) -> npt.NDArray[np.int_]:
    """Weights of every error with this syndrome, for stabilizers whose X parts are the identity.

    Z on the set syndrome bits has the syndrome, and all other such errors are its products with
    the 2**n stabilizers, so this enumeration is independent of the code under test.
    """
    n = len(matrix)
    assert np.array_equal(matrix[:, :n], np.eye(n))
    base = np.concatenate([np.zeros(n, dtype=int), (syndrome >> np.arange(n)) & 1])
    subsets = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    errors = (base + subsets @ matrix.astype(int)) % 2
    x, z = errors[:, :n], errors[:, n:]
    return (x | z).sum(axis=1) if metric == "pauli" else np.maximum(x.sum(axis=1), z.sum(axis=1))


def _pauli_weight(pauli: int, num_qubits: int, metric: Metric) -> int:
    """Weight of a packed Pauli, with X in the low bits and Z in the next block."""
    x, z = pauli & ((1 << num_qubits) - 1), pauli >> num_qubits
    return (x | z).bit_count() if metric == "pauli" else max(x.bit_count(), z.bit_count())


def _ghz_blocks(num_blocks: int) -> tuple[npt.NDArray[np.uint8], int]:
    """Stabilizers of independent GHZ blocks, and the syndrome of X on each block's last qubit.

    That syndrome flips two Z-type stabilizers in every block, which needs an X component on the
    block, so every equivalent error has X weight at least ``num_blocks``.  Weight ``num_blocks``
    is achieved, so it is the exact minimum in both metrics.
    """
    generators = [
        stim.PauliString("_" * (3 * block) + local + "_" * (3 * (num_blocks - block - 1)))
        for block in range(num_blocks)
        for local in ("XXX", "Z_Z", "_ZZ")
    ]
    witness = stim.PauliString("__X" * num_blocks)
    syndrome = sum((not witness.commutes(gg)) << jj for jj, gg in enumerate(generators))
    matrix = np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)
    return matrix, syndrome


def _kernel_words(checks: npt.NDArray[np.uint8], max_weight: int) -> set[int]:
    """Nonzero words of weight at most ``max_weight`` in the kernel of ``checks``."""
    width = checks.shape[1]
    words = np.arange(1, 1 << width)
    bits = (words[:, None] >> np.arange(width)) & 1
    in_kernel = ~np.any((bits @ checks.T.astype(int)) % 2, axis=1)
    return set(map(int, words[in_kernel & (bits.sum(axis=1) <= max_weight)]))


def test_kernel_search_matches_brute_force() -> None:
    """Every low-weight kernel word is yielded, for both storage layouts and early stopping."""
    rng = np.random.default_rng(431)
    cases = [
        rng.integers(2, size=(height, width), dtype=np.uint8)
        for width in range(1, 10)
        for height in (0, 1, width // 2, width, width + 1)
    ]
    # Twelve rows tie sixteen columns into groups of four, so no kernel word has weight below four.
    structured = np.zeros((12, 16), dtype=np.uint8)
    for row in range(12):
        structured[row, row % 4] = structured[row, row + 4] = 1
    cases.append(structured)
    for checks in cases:
        width = checks.shape[1]
        for max_weight in (1, 3, 4, width):
            search = BoundedKernelSearch(checks, max_weight, batch_size=2)
            assert set(search.words()) == _kernel_words(checks, max_weight)


def test_kernel_search_packs_words_across_machine_words() -> None:
    """The even-weight code of length 130 has exactly one word per pair of columns."""
    search = BoundedKernelSearch(np.ones((1, 130), dtype=np.uint8), 2)
    expected = {(1 << i) | (1 << j) for i, j in itertools.combinations(range(130), 2)}
    assert set(search.words()) == expected


def test_fixed_weight_supports_cover_all_combinations() -> None:
    """Supports are exactly the combinations, whatever the batch size."""
    for dimension, weight in itertools.product(range(1, 8), range(1, 8)):
        if weight <= dimension:
            batches = _iter_fixed_weight_supports(dimension, weight, batch_size=3)
            supports = [tuple(support) for batch in batches for support in batch]
            assert sorted(supports) == list(itertools.combinations(range(dimension), weight))


def test_fixed_weight_supports_stream_in_lexicographic_order() -> None:
    """Streaming, used when supports are too many to rank in 64 bits, yields every combination."""
    batches = _stream_fixed_weight_supports(5, 2, batch_size=3)
    supports = [tuple(support) for batch in batches for support in batch]
    assert supports == list(itertools.combinations(range(5), 2))
    # C(200, 20) exceeds 2**63, so the public iterator streams; take only its first batch.
    first = next(_iter_fixed_weight_supports(200, 20, batch_size=3))
    expected = itertools.islice(itertools.combinations(range(200), 20), 3)
    assert first.tolist() == [list(support) for support in expected]


@pytest.mark.parametrize("metric", ["pauli", "css"])
def test_decisions_match_exhaustive_minima(metric: Metric) -> None:
    """Decisions agree with exhaustively computed minima for shuffled queries and a tiny cache."""
    rng = np.random.default_rng(182)
    for num_qubits in range(1, 7):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        matrix = _graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        minima = [
            int(_class_weights(matrix, syndrome, metric).min())
            for syndrome in range(1 << num_qubits)
        ]
        queries = list(itertools.product(range(1 << num_qubits), range(num_qubits + 1)))
        oracle = PauliCosetSearch(matrix, metric=metric, cache_size=3, batch_size=2)
        for index in rng.permutation(len(queries)):
            syndrome, threshold = queries[index]
            decision = oracle.decide(syndrome, threshold)
            assert decision.within_bound == (minima[syndrome] <= threshold)
            assert oracle.syndrome(decision.representative) == syndrome
            assert decision.lower_bound <= minima[syndrome]
            if decision.within_bound:
                assert _pauli_weight(decision.representative, num_qubits, metric) <= threshold
            else:
                assert decision.lower_bound > threshold


def test_css_weight_counts_the_larger_sector() -> None:
    """X0 Z1 on |0+> has CSS weight one but physical weight two."""
    matrix = np.array([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
    assert not PauliCosetSearch(matrix, metric="pauli").decide(3, 1).within_bound
    css = PauliCosetSearch(matrix, metric="css")
    assert css.decide(3, 1).within_bound
    assert not css.decide(3, 0).within_bound


def test_css_weight_minimizes_both_sectors_jointly() -> None:
    """On a four-qubit path graph state, Z^4 is equivalent to pure X and to pure Z, but not to both.

    The X part and the Z part can each be made zero, yet no one error has CSS weight below two.
    """
    matrix = _graph_state(4, [(0, 1), (1, 2), (2, 3)])
    assert _class_weights(matrix, 15, "css").min() == 2
    oracle = PauliCosetSearch(matrix, metric="css")
    assert not oracle.decide(15, 1).within_bound
    assert oracle.decide(15, 2).within_bound


@pytest.mark.parametrize("metric", ["pauli", "css"])
def test_global_bound_after_threshold_queries(metric: Metric) -> None:
    """Four GHZ blocks need weight four, so a bound of five must never be reported or cached."""
    matrix, syndrome = _ghz_blocks(4)
    oracle = PauliCosetSearch(matrix, metric=metric)
    below = oracle.decide(syndrome, 3)
    at_minimum = oracle.decide(syndrome, 4)
    below_again = oracle.decide(syndrome, 3)
    assert not below.within_bound and below.lower_bound == 4
    assert at_minimum.within_bound
    assert oracle.syndrome(at_minimum.representative) == syndrome
    assert not below_again.within_bound


def test_recomputation_keeps_proved_bounds() -> None:
    """A later answer for a syndrome never forgets an earlier proved lower bound."""
    matrix = _graph_state(10, GRAPH_10_EDGES)
    assert _class_weights(matrix, 1019, "pauli").min() == 3
    oracle = PauliCosetSearch(matrix)
    first = oracle.decide(1019, 2)
    second = oracle.decide(1019, 3)
    third = oracle.decide(1019, 2)
    assert not first.within_bound and first.lower_bound == 3
    assert second.within_bound and second.lower_bound >= first.lower_bound
    assert not third.within_bound and third.lower_bound == 3


def test_bound_with_rank_deficient_information_set() -> None:
    """Qubit 3 is isolated, so one information set is rank-deficient; its bound is still exact."""
    edges = [(0, 1), (0, 6), (0, 7), (1, 2), (1, 7), (2, 6), (2, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    matrix = _graph_state(8, edges)
    assert _class_weights(matrix, 11, "pauli").min() == 3
    decision = PauliCosetSearch(matrix).decide(11, 2)
    assert not decision.within_bound and decision.lower_bound == 3


def test_exhausted_search_gives_the_exact_minimum() -> None:
    """On |00000>, flipping every qubit has CSS weight five, which only exhaustion can prove."""
    zero_state = np.hstack([np.zeros((5, 5), dtype=np.uint8), np.eye(5, dtype=np.uint8)])
    oracle = PauliCosetSearch(zero_state, metric="css")
    decision = oracle.decide(31, 4)
    assert not decision.within_bound and decision.lower_bound == 5
    assert oracle.decide(31, 5).within_bound


@pytest.mark.parametrize("metric", ["pauli", "css"])
def test_wide_states_span_several_packed_words(metric: Metric) -> None:
    """Eleven GHZ blocks use 66 (CSS) or 99 (Pauli) bits, more than one 64-bit word."""
    num_blocks = 11
    matrix, syndrome = _ghz_blocks(num_blocks)
    oracle = PauliCosetSearch(matrix, metric=metric)
    if metric == "pauli":  # the CSS lower bound is too slow to prove at this width
        below = oracle.decide(syndrome, num_blocks - 1)
        assert not below.within_bound and below.lower_bound == num_blocks
    found = oracle.decide(syndrome, num_blocks)
    assert found.within_bound
    assert oracle.syndrome(found.representative) == syndrome
    assert _pauli_weight(found.representative, 3 * num_blocks, metric) == num_blocks


@pytest.mark.parametrize("metric", ["pauli", "css"])
def test_candidate_weights_above_one_byte_are_exact(metric: Metric) -> None:
    """Weights above 255 must not wrap around in a one-byte sum."""
    n = 257
    zero_state = np.hstack([np.zeros((n, n), dtype=np.uint8), np.eye(n, dtype=np.uint8)])
    oracle = PauliCosetSearch(zero_state, metric=metric)
    width = (3 * n + 63) // 64 if metric == "pauli" else (2 * n + 63) // 64
    all_ones = np.full((1, width), np.iinfo(np.uint64).max, dtype=np.uint64)
    expected = 64 * width // 2 if metric == "pauli" else n
    assert oracle._candidate_weights(all_ones).tolist() == [expected]


def test_invalid_arguments_are_rejected() -> None:
    """Malformed searches and queries raise ValueError."""
    with pytest.raises(ValueError):
        BoundedKernelSearch([[1]], -1)
    with pytest.raises(ValueError):
        BoundedKernelSearch([[1]], 1, batch_size=0)
    # Not 2n columns, dependent rows, and anticommuting rows.
    for stabilizers in ([[1]], [[0, 0]], [[1, 0, 0, 0], [0, 0, 1, 0]]):
        with pytest.raises(ValueError):
            PauliCosetSearch(stabilizers)
    with pytest.raises(ValueError):
        PauliCosetSearch([[1, 0]], cache_size=0)
    with pytest.raises(ValueError):
        PauliCosetSearch([[1, 0]], batch_size=0)
    with pytest.raises(ValueError):
        PauliCosetSearch([[1, 0]], metric=cast(Metric, "invalid"))
    oracle = PauliCosetSearch([[1, 0]])
    with pytest.raises(ValueError):
        oracle.decide(2, 1)
    with pytest.raises(ValueError):
        oracle.decide(0, -1)
