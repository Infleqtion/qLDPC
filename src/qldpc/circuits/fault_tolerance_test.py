# SPDX-License-Identifier: Apache-2.0

"""Independent small oracles for the bounded-spread verifier."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
import stim

from qldpc import circuits
from qldpc.codes.distance import PauliCosetSearch

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


def test_random_maps() -> None:
    """Minimal-dependency search agrees with all accepted patterns, including flag cancellation."""
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
