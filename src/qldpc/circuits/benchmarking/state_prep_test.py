# SPDX-License-Identifier: Apache-2.0

"""Tests for state-preparation diagnostics and sampling tasks."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import circuits, codes, math
from qldpc.objects import Pauli


def test_state_prep_benchmarks() -> None:
    """State preparation circuit benchmarks."""

    # construct a state prep circuit for the Steane code
    code = codes.SteaneCode()
    circuit = stim.Circuit("""
        # non-fault-tolerant |0> state prep
        H 0 2 4
        CX 0 3 2 1 4 5
        CX 0 1 2 6 4 3
        CX 2 5 3 6
        # flag a logical Z representative
        H 7
        CZ 7 1 7 3 7 5
        MX 7
    """)

    noise_model_family = circuits.DepolarizingNoiseModel
    error_rates = np.logspace(-3, -1, 5)

    # observables can be specified by either a symplectic matrix or list of Pauli strings
    observables = code.get_logical_ops(Pauli.Z, symplectic=True)
    string_observables = [math.op_to_string(obs) for obs in observables]

    # build sinter tasks
    tasks = circuits.get_state_prep_diagnostic_tasks(
        code,
        circuit,
        error_rates,
        noise_model_family,
        observables=string_observables,
    )
    for error_rate, task in zip(error_rates, tasks):
        assert task.json_metadata["p"] == error_rate
    assert (
        circuits.get_state_prep_diagnostic_tasks(code, circuit, [0.123], metadata={"p": 999})[
            0
        ].json_metadata["p"]
        == 0.123
    )

    # find observables automatically and post-select on all measurements
    task = circuits.get_state_prep_diagnostic_tasks(
        code,
        circuit,
        error_rates[:1],
        noise_model_family,
        post_select=True,
    )[0]
    diagnostic_circuit, _ = circuits.get_state_prep_diagnostic_circuit(
        code, circuit, add_flags=True
    )
    postselection_array = np.zeros(task.circuit.num_detectors, dtype=int)
    postselection_array[: circuit.num_measurements] = 1
    postselection_mask = np.packbits(postselection_array, bitorder="little")
    assert task.circuit == noise_model_family(error_rates[0]).noisy_circuit(diagnostic_circuit)
    assert np.array_equal(task.postselection_mask, postselection_mask)

    # we can also post-select manually
    circuit.append("DETECTOR", stim.target_rec(-1))
    diagnostic_circuit, _ = circuits.get_state_prep_diagnostic_circuit(code, circuit)
    task = circuits.get_state_prep_diagnostic_tasks(
        code,
        circuit,
        error_rates[:1],
        noise_model_family,
        observables=string_observables,
        post_select=range(circuit.num_measurements),
    )[0]
    assert task.circuit == noise_model_family(error_rates[0]).noisy_circuit(diagnostic_circuit)
    assert np.array_equal(task.postselection_mask, postselection_mask)

    # we can only manually post-select on detectors that are present in the circuit
    with pytest.raises(ValueError, match="selectable prep/flag detectors"):
        circuits.get_state_prep_diagnostic_tasks(
            code,
            circuit,
            error_rates[:1],
            noise_model_family,
            observables=string_observables,
            post_select=[circuit.num_measurements],
        )
    negative_task = circuits.get_state_prep_diagnostic_tasks(
        code,
        circuit,
        error_rates[:1],
        noise_model_family,
        observables=string_observables,
        post_select=[-1],
    )[0]
    assert negative_task.postselection_mask is not None
    selected_bits = np.unpackbits(negative_task.postselection_mask, bitorder="little")
    assert selected_bits[circuit.num_detectors - 1] == 1
    assert np.sum(selected_bits[: circuit.num_detectors]) == 1


def test_state_prep_benchmark_remaps_data_qubits() -> None:
    """Diagnostic validation and measurements honor a non-identity data-qubit frame."""
    code = codes.SteaneCode()
    qubit_ids = circuits.QubitIDs(
        tuple(range(3, 3 + len(code))),
        tuple(range(20, 20 + code.num_checks)),
    )
    prep = circuits.with_remapped_qubits(
        circuits.get_encoding_circuit(code, only_zero=True), qubit_ids.data
    )
    diagnostic, _ = circuits.get_state_prep_diagnostic_circuit(code, prep, qubit_ids=qubit_ids)
    assert diagnostic.num_observables > 0
    assert "Z4" in str(diagnostic)
    assert "Z0" not in str(diagnostic)


def test_pure_logical_state(pytestconfig: pytest.Config) -> None:
    """Identify invalid pure logical states."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.SHPCode(codes.ClassicalCode.random(4, 2, seed=np.random.randint(2**31)))
    encoder = circuits.get_encoding_circuit(code)

    with pytest.raises(ValueError, match="does not prepare"):
        circuits.get_state_prep_diagnostic_circuit(
            code, stim.Circuit(f"X {len(code) - 1}") + encoder
        )

    with pytest.raises(ValueError, match="pure"):
        circuits.get_state_prep_diagnostic_circuit(
            code, stim.Circuit(f"H 0\nCX 0 {len(code)}") + encoder
        )
