# SPDX-License-Identifier: Apache-2.0

"""Independent regression oracles for bounded kernel and affine Pauli searches."""

from __future__ import annotations

import itertools
from typing import Literal
from unittest import mock

import numpy as np
import pytest
import stim

from . import bounded


def test_bounded_kernel_search() -> None:
    """Independent enumeration checks coverage, nonlinear filtering, and both storage paths."""
    rng = np.random.default_rng(431)
    for width in range(1, 10):
        for height in (0, 1, width // 2, width, width + 1):
            checks = rng.integers(2, size=(height, width), dtype=np.uint8)
            for budget in (1, 3, width):
                search = bounded.BoundedKernelSearch(checks, budget, batch_size=2)
                actual = set(search.words())
                expected = {
                    word
                    for word in range(1, 1 << width)
                    if word.bit_count() <= budget
                    and not np.any(checks @ [(word >> jj) & 1 for jj in range(width)] % 2)
                }
                assert actual == expected
                # Harmless initial rows must not create an eligible-word upper bound.
                assert {word for word in actual if word % 3 == 0} == {
                    word for word in expected if word % 3 == 0
                }
    with pytest.raises(ValueError, match="nonnegative"):
        bounded.BoundedKernelSearch([[1]], -1)
    with pytest.raises(ValueError, match="positive"):
        bounded.BoundedKernelSearch([[1]], 1, batch_size=0)


def test_bounded_kernel_information_bound() -> None:
    """Disjoint full information sets certify all repetition words at coefficient depth one."""
    checks = np.zeros((12, 16), dtype=np.uint8)
    for jj in range(12):
        checks[jj, jj % 4] = checks[jj, jj + 4] = 1
    search = bounded.BoundedKernelSearch(checks, 3)
    assert list(search.words()) == []
    assert search.ranks == (4, 4, 4, 4)
    assert search.coefficient_depth == 1
    assert search.lower_bound == 8
    assert search.generated == 16 < sum(1 for _ in itertools.combinations(range(16), 3))
    wider = bounded.BoundedKernelSearch(checks, 4)
    assert min(word.bit_count() for word in wider.words()) == 4 < wider.lower_bound
    # Check packing across multiple 64-bit words without dense high-rate generators.
    search = bounded.BoundedKernelSearch(np.ones((1, 130), dtype=int), 2)
    assert len(set(search.words())) == 130 * 129 // 2


@pytest.mark.parametrize("weight", ["pauli", "css"])
def test_pauli_coset_search_exhaustive(weight: Literal["pauli", "css"]) -> None:
    """Compare every small syndrome and threshold with independent stabilizer multiplication."""
    rng = np.random.default_rng(182)
    for n in range(1, 7):
        circuit = stim.Circuit()
        circuit.append("H", range(n))
        for jj, kk in itertools.combinations(range(n), 2):
            if rng.integers(2):
                circuit.append("CZ", [jj, kk])
        circuit.append("S", range(n))
        generators = stim.Tableau.from_circuit(circuit).to_stabilizers()
        matrix = np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)
        oracle = bounded.PauliCosetSearch(matrix, weight=weight, cache_size=2, batch_size=2)
        minimum = [n + 1] * (1 << n)
        for pauli in itertools.product(range(4), repeat=n):
            string = stim.PauliString(pauli)
            syndrome = sum((not string.commutes(gg)) << jj for jj, gg in enumerate(generators))
            xs, zs = string.to_numpy()
            value = string.weight if weight == "pauli" else max(int(xs.sum()), int(zs.sum()))
            minimum[syndrome] = min(minimum[syndrome], value)
        for syndrome, minimum_weight in enumerate(minimum):
            for threshold in (n, 0, 1, 2, 3, 0, n):
                decision = oracle.decide(syndrome, threshold)
                assert decision.within_bound == (minimum_weight <= threshold)
                assert oracle.syndrome(decision.representative) == syndrome
                assert decision.lower_bound <= minimum_weight
                if decision.within_bound:
                    string = decision.representative
                    x, z = string & ((1 << n) - 1), string >> n
                    actual = (
                        (x | z).bit_count()
                        if weight == "pauli"
                        else max(x.bit_count(), z.bit_count())
                    )
                    assert actual <= threshold
                else:
                    assert decision.lower_bound > threshold
        assert oracle.preparations <= 1
        assert len(oracle._cache) <= 2
        assert oracle.cache_hits


def test_pauli_coset_bounds_and_validation() -> None:
    """Affine zero coefficients, exact negatives, cached thresholds, and Y weight."""
    matrix = np.hstack([np.zeros((5, 5), dtype=np.uint8), np.eye(5, dtype=np.uint8)])
    oracle = bounded.PauliCosetSearch(matrix, batch_size=3)
    assert not oracle.decide(3, 1).within_bound
    assert not oracle.decide(31, 2).within_bound
    assert oracle.preparations == 1
    assert not oracle.decide(31, 3).within_bound
    assert oracle.decide(31, 5).within_bound
    assert oracle.preparations == 1
    assert oracle.decide(0, 0).representative == 0
    y_state = bounded.PauliCosetSearch([[1, 1]])
    assert y_state.decide(1, 1).within_bound
    for invalid_matrix, match in (
        ([[1]], "complete"),
        ([[0, 0]], "independent"),
        ([[1, 0, 0, 0], [0, 0, 1, 0]], "commute"),
    ):
        with pytest.raises(ValueError, match=match):
            bounded.PauliCosetSearch(invalid_matrix)
    with pytest.raises(ValueError, match="positive"):
        bounded.PauliCosetSearch([[1, 0]], cache_size=0)
    with pytest.raises(ValueError, match="range"):
        oracle.decide(32, 2)
    with pytest.raises(ValueError, match="negative"):
        oracle.decide(0, -1)


def test_affine_partial_information_set() -> None:
    """A partial pivot set proves absence after depth one when full sets alone cannot."""
    circuit = stim.Circuit("""
        H 0 1 2 3 4 5 6 7
        CZ 0 1 0 6 0 7 1 2 1 7 2 6 2 7 4 5 4 6 5 7 6 7
    """)
    generators = stim.Tableau.from_circuit(circuit).to_stabilizers()
    matrix = np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)
    oracle = bounded.PauliCosetSearch(matrix)
    decision = oracle.decide(11, 2)
    assert not decision.within_bound and decision.lower_bound == 3
    assert oracle.ranks == (8, 8, 7)
    assert oracle.coefficient_depth == 1
    representative = stim.PauliString.from_numpy(
        xs=np.array([(decision.representative >> jj) & 1 for jj in range(8)], dtype=bool),
        zs=np.array([(decision.representative >> (jj + 8)) & 1 for jj in range(8)], dtype=bool),
    )
    assert sum((not representative.commutes(gg)) << jj for jj, gg in enumerate(generators)) == 11
    minimum = 8
    for mask in range(256):
        candidate = representative.copy()
        for jj, generator in enumerate(generators):
            if (mask >> jj) & 1:
                candidate *= generator
        minimum = min(minimum, candidate.weight)
    assert minimum == 3


@pytest.mark.parametrize("batch_size", [1, 7, 4096])
@pytest.mark.parametrize("weight", ["pauli", "css"])
def test_affine_global_bound_and_cache(batch_size: int, weight: Literal["pauli", "css"]) -> None:
    """Four nonzero GHZ-block syndromes need exactly four qubits, not unseen bound five."""
    generators = [
        stim.PauliString("_" * (3 * block) + local + "_" * (9 - 3 * block))
        for block in range(4)
        for local in ("XXX", "Z_Z", "_ZZ")
    ]
    witness = stim.PauliString("__X__X__X__X")
    syndrome = sum((not witness.commutes(gg)) << jj for jj, gg in enumerate(generators))
    assert syndrome == 3510 and witness.weight == 4
    # Each disjoint block has a nonzero syndrome, requiring at least one affected qubit.
    assert all((syndrome >> (3 * block)) & 7 for block in range(4))
    matrix = np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)
    oracle = bounded.PauliCosetSearch(matrix, weight=weight, batch_size=batch_size)
    negative = oracle.decide(syndrome, 3)
    assert not negative.within_bound
    assert negative.lower_bound == 4
    assert oracle.ranks == ((12, 12, 12) if weight == "pauli" else (12, 8))
    assert oracle.coefficient_depth == (2 if weight == "pauli" else 5)
    generated = oracle.generated
    at_minimum = oracle.decide(syndrome, 4)
    assert at_minimum.within_bound and at_minimum.lower_bound == 4
    assert oracle.generated == generated
    assert oracle.syndrome(at_minimum.representative) == syndrome
    # The positive decision must not erase the proved bound that already answers threshold three.
    hits = oracle.cache_hits
    repeated = oracle.decide(syndrome, 3)
    assert not repeated.within_bound and repeated.lower_bound == 4
    assert oracle.cache_hits == hits + 1 and oracle.generated == generated
    fresh = bounded.PauliCosetSearch(matrix, weight=weight).decide(syndrome, 4)
    assert fresh.within_bound and fresh.lower_bound <= 4


def test_cache_keeps_proved_bound_after_recompute() -> None:
    """A positive recompute at a higher threshold must not discard the bound for a lower one."""
    n = 10
    # A graph state: each stabilizer is X on one vertex and Z on its neighbors.
    graph_state = [
        "X_ZZ__ZZ__",
        "_X______Z_",
        "Z_X__Z__Z_",
        "Z__XZ_ZZ__",
        "___ZX_ZZ_Z",
        "__Z__XZ___",
        "Z__ZZZX_Z_",
        "Z__ZZ__XZ_",
        "_ZZ___ZZX_",
        "____Z____X",
    ]
    matrix = np.array(
        [np.concatenate(stim.PauliString(gg).to_numpy()) for gg in graph_state], dtype=np.uint8
    )
    oracle = bounded.PauliCosetSearch(matrix)
    syndrome = 1019

    # The first search keeps a heavy representative, so only its proved bound answers threshold two.
    negative = oracle.decide(syndrome, 2)
    assert not negative.within_bound and negative.lower_bound == 3
    assert oracle._weight(negative.representative) == 4
    positive = oracle.decide(syndrome, 3)
    assert positive.within_bound and positive.lower_bound == 3
    generated, hits = oracle.generated, oracle.cache_hits
    repeated = oracle.decide(syndrome, 2)
    assert not repeated.within_bound and repeated.lower_bound == 3
    assert oracle.cache_hits == hits + 1 and oracle.generated == generated

    # Brute force over every stabilizer product confirms the exact minimum.
    rows = [sum(int(bit) << jj for jj, bit in enumerate(row)) for row in matrix]
    assert oracle.syndrome(positive.representative) == syndrome
    weights = []
    for mask in range(1 << n):
        word = positive.representative
        for jj, row in enumerate(rows):
            if (mask >> jj) & 1:
                word ^= row
        weights.append(oracle._weight(word))
    assert min(weights) == 3


@pytest.mark.parametrize("cache_size", [1, 2, 4096])
def test_affine_query_order_boundaries(cache_size: int) -> None:
    """Global bounds stay valid across threshold equality, query order, and cache eviction."""
    generators = [
        stim.PauliString("_" * (3 * block) + local + "_" * (9 - 3 * block))
        for block in range(4)
        for local in ("XXX", "Z_Z", "_ZZ")
    ]
    matrix = np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)
    # The eight local syndromes have one-qubit representatives I,Z0,X0,Y0,X1,Y1,X2,Y2.
    local_representatives = ("___", "Z__", "X__", "Y__", "_X_", "_Y_", "__X", "__Y")
    rng = np.random.default_rng(713)
    local_syndromes: list[tuple[int, ...]] = [(0, 0, 0, 0), (6, 6, 6, 6)]
    local_syndromes += [tuple(map(int, row)) for row in rng.integers(8, size=(24, 4))]
    for reverse in (False, True):
        oracle = bounded.PauliCosetSearch(matrix, cache_size=cache_size)
        for local in local_syndromes:
            syndrome = sum(value << (3 * block) for block, value in enumerate(local))
            minimum = sum(value != 0 for value in local)
            witness = stim.PauliString("".join(local_representatives[value] for value in local))
            assert witness.weight == minimum
            assert (
                sum((not witness.commutes(gg)) << jj for jj, gg in enumerate(generators))
                == syndrome
            )
            thresholds = (max(0, minimum - 1), minimum, minimum + 1, 0, minimum)
            for threshold in reversed(thresholds) if reverse else thresholds:
                decision = oracle.decide(syndrome, threshold)
                assert decision.within_bound == (minimum <= threshold)
                assert 0 <= decision.lower_bound <= minimum
                assert oracle.syndrome(decision.representative) == syndrome
                pauli = decision.representative
                weight = ((pauli | (pauli >> 12)) & ((1 << 12) - 1)).bit_count()
                assert weight >= minimum
                if decision.within_bound:
                    assert weight <= threshold
                else:
                    assert threshold < decision.lower_bound
            # Eviction must only cause recomputation, never a different mathematical decision.
            oracle.decide(syndrome ^ 1, 12)
            oracle.decide(syndrome ^ 2, 12)
            decision = oracle.decide(syndrome, minimum)
            assert decision.within_bound and decision.lower_bound <= minimum
        assert len(oracle._cache) <= cache_size


def test_css_weight_one_and_fixed_metric() -> None:
    """X0 Z1 has CSS weight one but physical weight two on |0+>, even at threshold zero."""
    matrix = np.array([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
    physical = bounded.PauliCosetSearch(matrix)
    css = bounded.PauliCosetSearch(matrix, weight="css")
    assert physical.weight == "pauli" and css.weight == "css"
    assert not physical.decide(3, 1).within_bound
    assert css.decide(3, 1).within_bound
    assert css.decide(3, 0).lower_bound == 1
    assert css.decide(3, 1).within_bound
    with pytest.raises(AttributeError):
        object.__setattr__(css, "weight", "pauli")
    assert css.weight == "css"
    assert len(css._x_singles) <= 3 and len(css._z_singles) <= 3


def test_css_joint_non_css_state() -> None:
    """Invertible graph-state X/Z projections can each vanish, but not in one representative."""
    circuit = stim.Circuit("H 0 1 2 3\nCZ 0 1 1 2 2 3")
    generators = stim.Tableau.from_circuit(circuit).to_stabilizers()
    error = stim.PauliString("ZZZZ")
    syndrome = sum((not error.commutes(gg)) << jj for jj, gg in enumerate(generators))
    assert syndrome == 15
    min_x = min_z = joint = 4
    for mask in range(16):
        representative = error.copy()
        for jj, gg in enumerate(generators):
            if (mask >> jj) & 1:
                representative *= gg
        xs, zs = representative.to_numpy()
        x, z = int(xs.sum()), int(zs.sum())
        min_x, min_z, joint = min(min_x, x), min(min_z, z), min(joint, max(x, z))
    assert min_x == min_z == 0 and joint == 2
    matrix = np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)
    oracle = bounded.PauliCosetSearch(matrix, weight="css")
    assert not oracle.decide(syndrome, 1).within_bound
    assert oracle.decide(syndrome, 2).within_bound


@pytest.mark.parametrize("cache_size", [1, 3, 4096])
def test_css_cache_boundaries_and_exhaustion(cache_size: int) -> None:
    """Independent mixed-product minima check query order, equality, eviction, and exhaustion."""
    n = 6
    matrix = np.zeros((n, 2 * n), dtype=np.uint8)
    for jj in range(n):
        matrix[jj, jj + n if jj < 3 else jj] = 1
    for reverse in (False, True):
        oracle = bounded.PauliCosetSearch(matrix, weight="css", cache_size=cache_size)
        for syndrome in range(64):
            minimum = max((syndrome & 7).bit_count(), (syndrome >> 3).bit_count())
            thresholds = (max(0, minimum - 1), minimum, minimum + 1, 0, minimum)
            for threshold in reversed(thresholds) if reverse else thresholds:
                decision = oracle.decide(syndrome, threshold)
                assert decision.within_bound == (minimum <= threshold)
                assert 0 <= decision.lower_bound <= minimum
                assert oracle.syndrome(decision.representative) == syndrome
                if not decision.within_bound:
                    assert threshold < decision.lower_bound
            oracle.decide(syndrome ^ 1, 6)
            assert oracle.decide(syndrome, minimum).within_bound
        assert len(oracle._cache) <= cache_size and oracle.preparations == 1
    zero_state = np.hstack([np.zeros((5, 5), dtype=np.uint8), np.eye(5, dtype=np.uint8)])
    oracle = bounded.PauliCosetSearch(zero_state, weight="css")
    zero_state[:] = 0  # Preparation owns its generators, even before lazy information-set setup.
    decision = oracle.decide(31, 4)
    assert not decision.within_bound and decision.lower_bound == 5
    assert oracle.coefficient_depth == 5
    assert oracle.decide(31, 5).within_bound


@pytest.mark.parametrize("weight", ["pauli", "css"])
@pytest.mark.parametrize("n", [130, 257])
def test_wide_packed_sectors(n: int, weight: Literal["pauli", "css"]) -> None:
    """Unaligned sector boundaries and weights above 255 retain the actual metric weight."""
    matrix = np.hstack([np.zeros((n, n), dtype=np.uint8), np.eye(n, dtype=np.uint8)])
    oracle = bounded.PauliCosetSearch(matrix, weight=weight)
    decision = oracle.decide((1 << n) - 1, n)
    assert decision.within_bound
    assert oracle._weight(decision.representative) == n
    assert oracle.preparations == 1


@pytest.mark.parametrize("weight", ["invalid"])
def test_invalid_coset_weight(weight: Literal["pauli", "css"]) -> None:
    with pytest.raises(ValueError, match="weight"):
        bounded.PauliCosetSearch([[1, 0]], weight=weight)


def test_fixed_weight_supports() -> None:
    """Witness enumeration retains every support and the large-count streaming fallback."""
    for dimension in range(1, 8):
        for weight in range(1, dimension + 1):
            actual = bounded._iter_fixed_weight_supports(
                dimension,
                weight,
                batch_size=3,
            )
            actual_supports = {tuple(support) for batch in actual for support in batch}
            expected_supports = set(itertools.combinations(range(dimension), weight))
            assert actual_supports == expected_supports

    with mock.patch(
        "qldpc.codes.distance.bounded.math.comb",
        return_value=np.iinfo(np.int64).max + 1,
    ):
        batches = bounded._iter_fixed_weight_supports(3, 2, batch_size=2)
        assert [batch.tolist() for batch in batches] == [
            [[0, 1], [0, 2]],
            [[1, 2]],
        ]
