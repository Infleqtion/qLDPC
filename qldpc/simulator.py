"""Tools for simulating codes to identify logical error rates

   Copyright 2023 The qLDPC Authors and Infleqtion Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from collections.abc import Sequence

import numpy as np
import stim

from qldpc import codes, objects


def get_syndrome_extraction_circuit(
    code: codes.CSSCode,
    stabilizer_pauli: objects.PauliXZ,
    error_rate: float,
    rounds: int = 1,
    gate_order: Sequence[tuple[int, int]] | None = None,
) -> stim.Circuit:
    """Get the syndrome extraction circuit for a CSS qubit code.

    Args:
        code: the CSS code in question
        stabilizer_pauli: the Pauli type of the stabilizers to measure
        error_rate: the probability of SPAM and two-qubit gate errors
        rounds: the number of syndrome extraction rounds
        gate_order (optional): the order in which to apply syndrome extraction gates
    Returns:
        a stim circuit

    The gate_order is a sequence of tuples (ancilla_qubit_index, data_qubit_index), where the
    ancilla_qubit_index must be in range(code.num_checks), and the data_qubit_index must be in
    range(code.num_qubits).  There should be exactly one tuple for each edge in the Tanner graph of
    the code.
    """
    if not code.field.order == 2:
        raise ValueError("Syndrome extraction circuits only supported for qubit CSS codes.")

    # identify data and ancilla qubits by index
    data_qubits = list(range(code.num_qubits))
    ancillas_x = [len(data_qubits) + qq for qq in range(code.num_checks_x)]
    ancillas_z = [len(data_qubits) + len(ancillas_x) + qq for qq in range(code.num_checks_z)]
    ancillas_xz = ancillas_x + ancillas_z

    if gate_order is None:
        gate_order = [
            (ancilla, data_qubit)
            for ancillas, matrix in [(ancillas_x, code.matrix_x), (ancillas_z, code.matrix_z)]
            for ancilla, row in zip(ancillas, matrix)
            for data_qubit in np.nonzero(row)[0]
        ]
    else:
        # assert that there are the correct number of gates addressing the correct qubits
        num_gates_x = len(np.nonzero(code.matrix_x)[0])
        num_gates_z = len(np.nonzero(code.matrix_z)[0])
        assert len(gate_order) == num_gates_x + num_gates_z
        assert all(
            (
                code.matrix_x[ancilla, data_qubit]
                if ancilla < code.num_checks_x
                else code.matrix_z[ancilla, data_qubit]
            )
            for ancilla, data_qubit in gate_order
        )

    # initialize data qubits
    circuit = stim.Circuit()
    circuit.append(f"R{stabilizer_pauli}", data_qubits)
    circuit.append(f"{~stabilizer_pauli}_ERROR", data_qubits, error_rate)

    # initialize ancillas, deferring SPAM errors until later
    circuit.append("RX", ancillas_xz)

    # construct circuit to extract syndromes once
    single_round_circuit = stim.Circuit()

    single_round_circuit.append("Z_ERROR", ancillas_xz, error_rate)
    for ancilla, data_qubit in gate_order:
        gate = "CX" if ancilla < len(ancillas_x) else "CZ"
        single_round_circuit.append(gate, [ancilla, data_qubit])
        single_round_circuit.append("DEPOLARIZE2", [ancilla, data_qubit], error_rate)

    # noisy syndrome measurement + ancilla reset
    single_round_circuit.append("Z_ERROR", ancillas_xz, error_rate)
    single_round_circuit.append("MRX", ancillas_xz)

    # append first round of syndrome extraction
    circuit += single_round_circuit

    # initial ancilla detectors
    ancilla_recs = {ancilla: -code.num_checks + qq for qq, ancilla in enumerate(ancillas_xz)}
    for ancilla in ancillas_x if stabilizer_pauli is objects.Pauli.X else ancillas_z:
        circuit.append("DETECTOR", stim.target_rec(ancilla_recs[ancilla]))

    # additional rounds of syndrome extraction
    if rounds > 1:
        repeat_circuit = single_round_circuit.copy()
        for ancilla in ancillas_xz:
            recs = [ancilla_recs[ancilla], ancilla_recs[ancilla] - code.num_checks]
            repeat_circuit.append("DETECTOR", [stim.target_rec(rec) for rec in recs])
        circuit += (rounds - 1) * repeat_circuit

    # measure out data qubits
    circuit.append(f"{~stabilizer_pauli}_ERROR", data_qubits, error_rate)
    circuit.append(f"M{stabilizer_pauli}", data_qubits)

    # check stabilizer parity
    data_qubit_recs = [-code.num_qubits + qq for qq in data_qubits]
    ancilla_recs = {
        ancilla: -code.num_qubits - code.num_checks + qq for qq, ancilla in enumerate(ancillas_xz)
    }
    ancillas = ancillas_x if stabilizer_pauli is objects.Pauli.X else ancillas_z
    matrix = code.matrix_x if stabilizer_pauli is objects.Pauli.X else code.matrix_z
    for ancilla, row in zip(ancillas, matrix):
        recs = [ancilla_recs[ancilla]] + [data_qubit_recs[qq] for qq in np.nonzero(row)[0]]
        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in recs])

    # check logical observable parity
    for logical_op_idx, logical_op in enumerate(code.get_logical_ops(stabilizer_pauli)):
        recs = [stim.target_rec(data_qubit_recs[qq]) for qq in np.nonzero(logical_op)[0]]
        circuit.append("OBSERVABLE_INCLUDE", recs, logical_op_idx)

    return circuit
