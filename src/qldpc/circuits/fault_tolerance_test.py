# SPDX-License-Identifier: Apache-2.0

"""Independent small oracles for the bounded-spread verifier."""

from __future__ import annotations

import itertools
from unittest import mock

import numpy as np
import pytest
import stim

from qldpc import circuits
from qldpc.codes.distance import BoundedKernelSearch, PauliCosetSearch

from . import fault_tolerance


def _exhaustive(flags: list[int], syndromes: list[int], n: int, budget: int) -> bool:
    """Product-zero target: effective weight is syndrome Hamming weight."""
    for size in range(1, min(budget, len(flags)) + 1):
        for support in itertools.combinations(range(len(flags)), size):
            flag = syndrome = 0
            for jj in support:
                flag ^= flags[jj]
                syndrome ^= syndromes[jj]
            if flag == 0 and syndrome.bit_count() > size:
                return False
    return True


@pytest.mark.parametrize("force_prefix", [False, True])
def test_random_maps(force_prefix: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Minimal-dependency search agrees with all accepted patterns, including flag cancellation."""
    monkeypatch.setattr(fault_tolerance, "_use_prefix_search", mock.Mock(return_value=force_prefix))
    rng = np.random.default_rng(612)
    for n in (3, 5):
        matrix = np.hstack([np.zeros((n, n), dtype=int), np.eye(n, dtype=int)])
        for rank in range(5):
            for _ in range(12):
                flags = list(map(int, rng.integers(1 << rank, size=8)))
                syndromes = list(map(int, rng.integers(1 << n, size=8)))
                for budget in (1, 2, 3, 5):
                    stats = circuits.StatePrepSpreadStatistics()
                    failure = fault_tolerance._certify_maps(
                        flags, syndromes, PauliCosetSearch(matrix), budget, stats
                    )
                    assert (failure is None) == _exhaustive(flags, syndromes, n, budget)
                    if failure is not None:
                        flag = syndrome = 0
                        for jj in failure.fault_indices:
                            flag ^= flags[jj]
                            syndrome ^= syndromes[jj]
                        assert flag == 0
                        assert 1 <= failure.fault_count <= budget
                        assert syndrome.bit_count() >= failure.effective_weight_lower_bound
                        assert failure.effective_weight_lower_bound > failure.fault_count
                        assert failure.output_syndrome == tuple(
                            (syndrome >> jj) & 1 for jj in range(n)
                        )
                        assert (
                            PauliCosetSearch(matrix).syndrome(
                                sum(
                                    int(stim.PauliString(failure.output_pauli)[jj] in (1, 2)) << jj
                                    for jj in range(n)
                                )
                                | sum(
                                    int(stim.PauliString(failure.output_pauli)[jj] in (2, 3))
                                    << (jj + n)
                                    for jj in range(n)
                                )
                            )
                            == syndrome
                        )


def test_minimal_prefix_oracle() -> None:
    """Independent accepted-subset enumeration checks canonical closure and original-ID mapping."""
    rng = np.random.default_rng(518)
    for width in range(10):
        for flag_bits in range(1, 5):
            flags = list(map(int, rng.integers(1, 1 << flag_bits, size=width)))
            syndromes = list(map(int, rng.integers(16, size=width)))
            originals = list(map(int, rng.permutation(np.arange(width) * 7 + 100)))
            mechanisms = list(zip(flags, syndromes, originals, strict=True))
            for budget in range(1, 6):
                expected: dict[tuple[int, ...], int] = {}
                accepted: list[int] = []
                for size in range(1, min(width, budget) + 1):
                    for support in itertools.combinations(range(width), size):
                        flag = syndrome = 0
                        for jj in support:
                            flag ^= flags[jj]
                            syndrome ^= syndromes[jj]
                        if flag:
                            continue
                        word = sum(1 << jj for jj in support)
                        if not any(previous & word == previous for previous in accepted):
                            expected[tuple(originals[jj] for jj in support)] = syndrome
                        accepted.append(word)
                stats = circuits.StatePrepSpreadStatistics()
                actual = list(fault_tolerance._iter_minimal_patterns(mechanisms, budget, stats))
                assert dict(actual) == expected
                assert len(actual) == len(expected)
                assert stats.generated == stats.prefix_extensions + len(actual)
                assert stats.independent_prefixes <= stats.prefix_extensions


def test_minimal_prefix_is_not_full_kernel() -> None:
    """The separate minimal search must not weaken the full-kernel iterator's contract."""
    flags = [1, 2, 3, 4, 8, 12]
    mechanisms = [(flag, 1, jj) for jj, flag in enumerate(flags)]
    stats = circuits.StatePrepSpreadStatistics()
    minimal = list(fault_tolerance._iter_minimal_patterns(mechanisms, 6, stats))
    assert {support for support, _ in minimal} == {(0, 1, 2), (3, 4, 5)}
    checks = np.array([[(flag >> jj) & 1 for flag in flags] for jj in range(4)], dtype=int)
    assert set(BoundedKernelSearch(checks, 6).words()) == {0b000111, 0b111000, 0b111111}
    wide = [(1 << 80, 1, 90), (1 << 130, 2, 10), ((1 << 80) ^ (1 << 130), 4, 50)]
    assert list(fault_tolerance._iter_minimal_patterns(wide, 3, stats)) == [((90, 10, 50), 7)]


@pytest.mark.parametrize(
    "columns,rank,budget,expected",
    [
        (45, 4, 3, True),
        (93, 5, 2, True),
        (381, 7, 3, True),
        (21, 3, 4, True),
        (30, 4, 5, True),
        (35, 5, 5, True),
        (21, 3, 2, False),
        (21, 3, 3, False),
        (24, 16, 3, False),
        (41, 20, 3, False),
        (49, 24, 3, True),
        (100, 3, 1, False),
    ],
)
def test_prefix_crossover(columns: int, rank: int, budget: int, expected: bool) -> None:
    assert fault_tolerance._use_prefix_search(columns, rank, budget) == expected
    assert fault_tolerance._use_prefix_search(columns, rank, budget, independent_flags=True) == (
        budget >= 2 and columns > 2 * rank
    )


@pytest.mark.parametrize("damaging", [False, True])
def test_adaptive_prefix_circuit(damaging: bool) -> None:
    """Real compiled DEMs use canonical closure, retain fault costs, and expose valid witnesses."""
    circuit = stim.Circuit()
    circuit.append("I", range(9))
    for flag in range(1, 16):
        for syndrome in (3, 5, 6):
            targets = [stim.target_x(jj) for jj in range(3) if (syndrome >> jj) & 1]
            targets += [stim.target_x(5 + jj) for jj in range(4) if (flag >> jj) & 1]
            circuit.append("CORRELATED_ERROR", targets, 0.001)
    if damaging:
        circuit.append("CORRELATED_ERROR", [stim.target_x(jj) for jj in (3, 4, 5)], 0.001)
    circuit += stim.Circuit("H 0 1 2\nS 0 2\nX 0")
    circuit.append("M", range(5, 9))
    for jj in range(4):
        circuit.append("DETECTOR", [stim.target_rec(jj - 4)])
    result = circuits.certify_state_prep_spread(
        circuit, output_qubits=[2, 0, 4, 1, 3], post_select=range(4), max_faults=3
    )
    assert result.certified != damaging
    assert any("Y" in gg for gg in result.target_stabilizers)
    assert any(gg.startswith("-") for gg in result.target_stabilizers)
    stats = result.statistics
    assert stats.search_method == "independent_prefix" and stats.outer_ranks == ()
    failure = result.counterexample
    if damaging:
        assert failure is not None and failure.fault_count in (2, 3)
        assert failure.fault_count < failure.effective_weight_lower_bound <= 4
        errors = [
            instruction
            for instruction in result.detector_error_model
            if instruction.type == "error"
        ]
        events: set[int] = set()
        for jj in failure.fault_indices:
            events.symmetric_difference_update(target.val for target in errors[jj].targets_copy())
        assert not events.intersection(range(4))
        generators = [stim.PauliString(gg) for gg in result.target_stabilizers]
        representative = stim.PauliString(failure.output_pauli)
        assert (
            tuple(int(not representative.commutes(gg)) for gg in generators)
            == failure.output_syndrome
        )
        assert sum(failure.output_syndrome) == 4
    else:
        # Even-parity three-bit syndromes have weight <=2; no singleton is accepted.
        assert failure is None
        assert stats.prefix_extensions == 1035 and stats.independent_prefixes == 990
        assert stats.generated == 2025
        assert stats.generated < 11521
        assert stats.minimal_candidates == 990


def test_adaptive_retains_strong_information_sets() -> None:
    """Low-rate accepted codes keep the original multiple-information-set path."""
    flags = [
        flag
        for group in range(8)
        for flag in (1 << (2 * group), 1 << (2 * group + 1), 3 << (2 * group))
    ]
    n = 5
    matrix = np.hstack([np.zeros((n, n), dtype=int), np.eye(n, dtype=int)])
    stats = circuits.StatePrepSpreadStatistics()
    with mock.patch.object(
        fault_tolerance, "_iter_minimal_patterns", side_effect=AssertionError("Wrong search")
    ):
        assert (
            fault_tolerance._certify_maps(flags, [3, 6, 28] * 8, PauliCosetSearch(matrix), 3, stats)
            is None
        )
    assert stats.search_method == "brouwer_zimmermann"
    assert stats.outer_ranks == (8, 8, 8) and stats.coefficient_depth == 1
    assert stats.generated == 24 and stats.prefix_extensions == 0


@pytest.mark.parametrize("budget", [2, 3, 6, 30])
def test_independent_flag_types_reduce_to_pairs(budget: int) -> None:
    """Acceptance has even multiplicity in each independent flag type at every budget."""
    flags = [flag for flag in (3, 5, 8, 16, 1 << 130) for _ in range(3)]
    n = 3
    matrix = np.hstack([np.zeros((n, n), dtype=int), np.eye(n, dtype=int)])
    stats = circuits.StatePrepSpreadStatistics()
    assert (
        fault_tolerance._certify_maps(flags, [3, 5, 6] * 5, PauliCosetSearch(matrix), budget, stats)
        is None
    )
    assert stats.flag_rank == 5 and stats.effective_budget == 2
    assert stats.search_method == "independent_prefix"
    assert stats.prefix_extensions == stats.independent_prefixes == 15
    assert stats.minimal_candidates == 15 and stats.generated == 30
    assert _exhaustive(flags, [3, 5, 6] * 5, n, budget)


def test_independent_flag_types_failure() -> None:
    """Pair-only reduction still returns a genuine accepted violation with original IDs."""
    flags = [flag for flag in (3, 5, 8) for _ in range(3)]
    n = 4
    matrix = np.hstack([np.zeros((n, n), dtype=int), np.eye(n, dtype=int)])
    stats = circuits.StatePrepSpreadStatistics()
    failure = fault_tolerance._certify_maps(
        flags, [1, 14, 0, 1, 2, 3, 1, 2, 3], PauliCosetSearch(matrix), 7, stats
    )
    assert failure is not None and failure.fault_count == 2
    assert failure.fault_indices == (0, 1)
    assert 2 < failure.effective_weight_lower_bound <= 4
    assert failure.output_syndrome == (1, 1, 1, 1)
    assert stats.search_method == "independent_prefix" and stats.effective_budget == 2


def test_dependency_reductions() -> None:
    """Genuine triples, decomposable pairs, duplicate joint symptoms, and budget boundaries."""
    n = 4
    matrix = np.hstack([np.zeros((n, n), dtype=int), np.eye(n, dtype=int)])
    for flags, syndromes, budget, expected in (
        ([1, 2, 3], [1, 2, 12], 2, True),
        ([1, 2, 3], [1, 2, 12], 3, False),
        ([1, 1], [1, 14], 3, False),
        ([0, 0, 0], [1, 2, 4], 4, True),
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 4, 8, 12], 4, False),
        ([1, 1, 2, 2, 4, 4], [1, 2, 1, 2, 1, 2], 4, True),
        ([1, 1, 1, 0], [1, 1, 1, 0], 4, True),
        ([0, 0], [0, 0], 4, True),
    ):
        stats = circuits.StatePrepSpreadStatistics()
        result = fault_tolerance._certify_maps(
            flags, syndromes, PauliCosetSearch(matrix), budget, stats
        )
        assert (result is None) == expected
    assert stats.generated == 0
    # Harmless systematic rows do not bound the first violating eligible word.
    n = 6
    matrix = np.hstack([np.zeros((n, n), dtype=int), np.eye(n, dtype=int)])
    flags = [1, 2, 4, 3, 5]
    syndromes = [0, 0, 0, 7, 56]
    for budget in (3, 4):
        result = fault_tolerance._certify_maps(
            flags, syndromes, PauliCosetSearch(matrix), budget, circuits.StatePrepSpreadStatistics()
        )
        assert (result is None) == (budget == 3)


def test_steane_mixed_pauli_witness() -> None:
    """An algebraic one-fault witness refutes the full criterion despite the pure-Z result."""
    ideal = stim.Circuit("""
        H 0 2 4
        CX 0 3 2 1 4 5
        CX 0 1 2 6 4 3
        CX 2 5 3 6
        H 7
        CZ 7 1 7 3 7 5
        MX 7
        DETECTOR rec[-1]
    """)
    circuit = circuits.DepolarizingNoiseModel(1e-3).noisy_circuit(ideal)
    result = circuits.certify_state_prep_spread(
        circuit, output_qubits=range(7), max_faults=3, post_select=[0]
    )
    failure = result.counterexample
    assert not result.certified and failure is not None
    assert failure.fault_count == 1
    assert failure.effective_weight_lower_bound == 2
    assert failure.output_syndrome == (1, 1, 0, 1, 0, 0, 0)
    stabilizers = [stim.PauliString(string) for string in result.target_stabilizers]
    representative = stim.PauliString("YZ_____")
    assert (
        tuple(int(not representative.commutes(gg)) for gg in stabilizers) == failure.output_syndrome
    )
    # Complete independent physical weight-zero/one oracle, not the implementation's lookup.
    for qubit in range(7):
        for pauli in "IXYZ":
            candidate = stim.PauliString(7)
            candidate[qubit] = pauli
            assert (
                tuple(int(not candidate.commutes(gg)) for gg in stabilizers)
                != failure.output_syndrome
            )
    errors = [
        instruction for instruction in result.detector_error_model if instruction.type == "error"
    ]
    targets = errors[failure.fault_indices[0]].targets_copy()
    assert {target.val for target in targets} == {1, 2, 4}


def test_ghz_global_counterexample_bound() -> None:
    """An accepted triple affects four GHZ blocks, proving effective weight exactly four."""
    circuit = stim.Circuit()
    for block in range(0, 12, 3):
        circuit.append("H", [block])
        circuit.append("CX", [block, block + 1, block, block + 2])
    circuit.append("CORRELATED_ERROR", [stim.target_x(12)], 0.001)
    circuit.append("CORRELATED_ERROR", [stim.target_x(13)], 0.001)
    circuit.append(
        "CORRELATED_ERROR",
        [stim.target_x(jj) for jj in (2, 5, 8, 11, 12, 13)],
        0.001,
    )
    circuit.append("M", [12, 13])
    circuit.append("DETECTOR", [stim.target_rec(-2)])
    circuit.append("DETECTOR", [stim.target_rec(-1)])
    assert circuits.certify_state_prep_spread(
        circuit, output_qubits=range(12), post_select=[0, 1], max_faults=2
    ).certified
    certificate = circuits.certify_state_prep_spread(
        circuit, output_qubits=range(12), post_select=[0, 1], max_faults=3
    )
    failure = certificate.counterexample
    assert not certificate.certified and failure is not None
    assert failure.fault_indices == (0, 1, 2)
    assert failure.effective_weight_lower_bound == 4
    witness = stim.PauliString("__X__X__X__X")
    generators = [stim.PauliString(gg) for gg in certificate.target_stabilizers]
    assert tuple(int(not witness.commutes(gg)) for gg in generators) == failure.output_syndrome
    # Every block has a nonzero local syndrome, so a representative must touch all four.
    for block in range(4):
        local = stim.PauliString("_" * (3 * block) + "Z_Z" + "_" * (9 - 3 * block))
        assert not witness.commutes(local)
    assert witness.weight == 4
    representative = stim.PauliString(failure.output_pauli)
    assert representative.weight == 4
    assert (
        tuple(int(not representative.commutes(gg)) for gg in generators) == failure.output_syndrome
    )
    errors = [
        instruction
        for instruction in certificate.detector_error_model
        if instruction.type == "error"
    ]
    acceptance: set[int] = set()
    for index in failure.fault_indices:
        acceptance.symmetric_difference_update(
            target.val
            for target in errors[index].targets_copy()
            if target.is_relative_detector_id() and target.val < 2
        )
    assert acceptance == set()


@pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
def test_non_css_signed_frame(pauli: str) -> None:
    """Signed non-CSS state, reordered physical outputs, correlated mechanisms, and ideal signs."""
    circuit = stim.Circuit(f"""
        H 4
        S 4
        CX 4 1
        X 1
        CORRELATED_ERROR(0.01) {pauli}4
    """)
    result = circuits.certify_state_prep_spread(circuit, output_qubits=[4, 1], max_faults=3)
    assert result.certified
    assert result.counterexample is None
    assert len(result.target_stabilizers) == 2
    assert any("Y" in stabilizer for stabilizer in result.target_stabilizers)
    assert any(stabilizer.startswith("-") for stabilizer in result.target_stabilizers)
    assert result.output_qubits == (4, 1)
    assert result.max_faults == 3


def test_correlated_violation() -> None:
    """A two-qubit output error is one correlated fault, not two artificially split faults."""
    result = circuits.certify_state_prep_spread(
        stim.Circuit("I 0 1\nCORRELATED_ERROR(0.01) X0 Y1"),
        output_qubits=[1, 0],
        max_faults=3,
    )
    assert not result.certified
    failure = result.counterexample
    assert failure is not None
    assert failure.fault_indices == (0,)
    assert failure.effective_weight_lower_bound == 2
    assert failure.output_syndrome == (1, 1)


def test_post_selection_pairs() -> None:
    """Individually rejected mechanisms can combine to form an accepted damaging pair."""
    circuit = stim.Circuit("""
        I 0 1 2 3 4
        CORRELATED_ERROR(0.01) X0 X1 X4
        CORRELATED_ERROR(0.01) X2 X3 X4
        M 4
        DETECTOR rec[-1]
    """)
    assert circuits.certify_state_prep_spread(
        circuit, output_qubits=[0, 1, 2, 3], post_select=[0], max_faults=1
    ).certified
    result = circuits.certify_state_prep_spread(
        circuit, output_qubits=[0, 1, 2, 3], post_select=[0], max_faults=3
    )
    assert not result.certified
    assert result.counterexample is not None
    assert result.counterexample.fault_count == 2
    assert result.counterexample.effective_weight_lower_bound >= 3
    assert result.statistics.flag_rank == 1


def test_empty_model() -> None:
    """No errors, repetitions, existing observables, and deterministic nonzero measurement bits."""
    circuit = stim.Circuit("""
        X 0
        REPEAT 2 {
            M 0
            DETECTOR rec[-1]
        }
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    result = circuits.certify_state_prep_spread(
        circuit, output_qubits=[0], post_select=[0, 1], max_faults=4
    )
    assert result.certified
    assert result.statistics.mechanisms == 0
    assert result.target_stabilizers == ("-Z",)


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_invalid_budget(budget: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        circuits.certify_state_prep_spread(
            stim.Circuit("I 0"), output_qubits=[0], max_faults=budget
        )


@pytest.mark.parametrize("outputs", [[], [0, 0], [1], [-1], [True], [0.5]])
def test_invalid_outputs(outputs: list[int]) -> None:
    with pytest.raises(ValueError, match="output_qubits"):
        circuits.certify_state_prep_spread(stim.Circuit("I 0"), output_qubits=outputs, max_faults=1)


def test_invalid_annotations_and_mixed_states() -> None:
    for text in ("H 0\nCX 0 1", "H 0\nM 0"):
        with pytest.raises(ValueError, match="pure"):
            circuits.certify_state_prep_spread(stim.Circuit(text), output_qubits=[0], max_faults=1)
    with pytest.raises(ValueError, match="post_select"):
        circuits.certify_state_prep_spread(
            stim.Circuit("I 0"), output_qubits=[0], max_faults=1, post_select=[0]
        )
    for text in (
        "H 1\nM 1\nDETECTOR rec[-1]",
        "H 1\nM 1\nOBSERVABLE_INCLUDE(0) rec[-1]",
    ):
        with pytest.raises(ValueError, match="non-deterministic"):
            circuits.certify_state_prep_spread(stim.Circuit(text), output_qubits=[0], max_faults=1)
    # A detector cannot be used as an assumed random post-selected branch condition.
    with pytest.raises(ValueError, match="non-deterministic"):
        circuits.certify_state_prep_spread(
            stim.Circuit("H 0\nM 0\nDETECTOR rec[-1]"),
            output_qubits=[0],
            post_select=[0],
            max_faults=1,
        )
