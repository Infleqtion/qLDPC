"""Tools for constructing syndrome extraction circuits.

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

import numpy as np
import stim

from qldpc import codes, objects


def get_syndrome_extraction_circuit(
    code: codes.CSSCode, stabilizer_pauli: objects.PauliXZ, error_rate: float, rounds: int = 1
) -> stim.Circuit:
    """Get the syndrome extraction circuit for a CSS qubit code."""
    if not code.field.order == 2:
        raise ValueError("Syndrome extraction circuits only supported for qubit CSS codes.")

    circuit = stim.Circuit()

    data_qubits = list(range(code.num_qubits))
    ancillas_x = [len(data_qubits) + qq for qq in range(code.num_checks_x)]
    ancillas_z = [len(data_qubits) + len(ancillas_x) + qq for qq in range(code.num_checks_z)]
    ancillas_xz = ancillas_x + ancillas_z

    # initialize data qubits
    circuit.append(f"R{~stabilizer_pauli}", data_qubits)
    circuit.append(f"{stabilizer_pauli}_ERROR", data_qubits, error_rate)

    # initialize ancillas, deferring SPAM errors until later
    circuit.append("RX", ancillas_xz)

    # construct circuit to extract syndromes once
    single_round_circuit = stim.Circuit()

    single_round_circuit.append("Z_ERROR", ancillas_xz, error_rate)
    for ancillas, matrix, pauli in [
        (ancillas_x, code.matrix_x, "X"),
        (ancillas_z, code.matrix_z, "Z"),
    ]:
        for ancilla, row in zip(ancillas, matrix):
            for data_qubit in np.nonzero(row)[0]:
                single_round_circuit.append(f"C{pauli}", [ancilla, data_qubit])
                single_round_circuit.append("DEPOLARIZE2", [ancilla, data_qubit], error_rate)

    # noisy syndrome measurement + ancilla reset
    single_round_circuit.append("Z_ERROR", ancillas_xz, error_rate)
    single_round_circuit.append("MRX", ancillas_xz)

    # append first round of syndrome extraction
    circuit += single_round_circuit

    # initial ancilla detectors
    for index in range(code.num_checks):
        circuit.append("DETECTOR", stim.target_rec(-index - 1))

    # additional rounds of syndrome extraction
    if rounds > 1:
        repeat_circuit = single_round_circuit.copy()
        for index in range(code.num_checks):
            repeat_circuit.append(
                "DETECTOR", [stim.target_rec(-1 - index), stim.target_rec(-1 - code.num_checks)]
            )
        circuit += (rounds - 1) * repeat_circuit

    # measure out data qubits
    circuit.append(f"{~stabilizer_pauli}_ERROR", data_qubits, error_rate)
    circuit.append(f"M{stabilizer_pauli}", data_qubits)

    # check stabilizer parity
    data_qubit_recs = [-code.num_qubits + qq for qq in data_qubits]
    ancilla_recs = [-code.num_qubits - code.num_checks + qq for qq in ancillas_xz]
    matrix = code.matrix_x if stabilizer_pauli is objects.Pauli.X else code.matrix_z
    for ancilla_rec, row in zip(ancilla_recs, matrix):
        recs = [ancilla_rec] + [data_qubit_recs[qq] for qq in np.nonzero(row)[0]]
        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in recs])

    # check logical observable parity
    for logical_op_idx, logical_op in enumerate(code.get_logical_ops(stabilizer_pauli)):
        recs = [stim.target_rec(data_qubit_recs[qq]) for qq in np.nonzero(logical_op)[0]]
        circuit.append("OBSERVABLE_INCLUDE", recs, logical_op_idx)

    return circuit


code = codes.SurfaceCode(3, 3, rotated=True)
circuit = get_syndrome_extraction_circuit(code, objects.Pauli.Z, error_rate=1e-3, rounds=3)
print(circuit)
exit()

