"""Tools for constructing useful circuits

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

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import stim

from qldpc import codes
from qldpc.objects import Pauli


def restrict_to_qubits(func: Callable[[...], stim.Circuit]) -> Callable[[...], stim.Circuit]:
    """Restrict a circuit constructor to qubit-based codes."""

    @functools.wraps(func)
    def qubit_func(*args: object, **kwargs: object) -> stim.Circuit:
        if any(isinstance(arg, codes.QuditCode) and arg.field.order != 2 for arg in args):
            raise ValueError("Circuit methods are only supported for qubit codes.")
        return func(*args, **kwargs)

    return qubit_func


def op_to_string(op: npt.NDArray[np.int_], flip_xz: bool = False) -> stim.PauliString:
    """Convert an integer array that represents a Pauli string into a stim.PauliString."""
    assert len(op) % 2 == 0
    num_qubits = len(op) // 2
    paulis = []
    for qubit in range(num_qubits):
        val_x = int(op[qubit])
        val_z = int(op[qubit + num_qubits])
        pauli = Pauli((val_x, val_z))
        paulis.append(str(pauli if not flip_xz else ~pauli))
    return stim.PauliString("".join(paulis))


@restrict_to_qubits
def get_ecoding_tableau(
    code: codes.QuditCode, string: stim.PauliString | None = None
) -> stim.Circuit:
    """Tableau to encode a logical all-|0> state of the given code.

    If provided a Pauli string, prepare a logical +1 eigenstate of that logical Pauli string.
    """
    string = string if isinstance(string, stim.PauliString) else stim.PauliString(code.dimension)
    if len(string) < code.dimension:
        string += stim.PauliString(len(string) - code.dimension)
    assert len(string) == code.dimension

    # identify logical operators that stabilize our target state
    logical_ops = code.get_logical_ops()
    logical_stabs = []
    for qubit, pauli in enumerate(string):
        if pauli == 1:
            logical_op = logical_ops[0, qubit]
        elif pauli == 3 or pauli == 0:
            logical_op = logical_ops[1, qubit]
        else:
            assert pauli == 2
            logical_op = logical_ops[0, qubit] + logical_ops[1, qubit]
        logical_stabs.append(op_to_string(logical_op))

    code_stabs = [op_to_string(row, flip_xz=True) for row in code.matrix]
    return stim.Tableau.from_stabilizers(logical_stabs + code_stabs, allow_redundant=True)


@restrict_to_qubits
def get_ecoding_circuit(
    code: codes.QuditCode, string: stim.PauliString = stim.PauliString()
) -> stim.Tableau:
    """Circuit to encode a logical all-|0> state of the given code.

    If provided a Pauli string, prepare a logical +1 eigenstate of that logical Pauli string.
    """
    return get_ecoding_tableau(code, string).to_circuit()


@restrict_to_qubits
def get_swap_transversal_circuit(
    code: codes.QuditCode, circuit: stim.Circuit | stim.Tableau
) -> stim.Circuit | None:
    """Get a SWAP-transversal implementation of a logical circuit (or tablaeau), if it exists."""
    # extend the circuit to act on code.num_qubits qubits, acting with the identity on ancillas
    circuit = circuit if isinstance(circuit, stim.Circuit) else circuit.to_circuit()
    for qubit in range(circuit.num_qubits, code.num_qubits):
        circuit.append("I", qubit)

    circuit.append("SWAP", [2, 3])
    print(circuit)
    print()

    # construct the physical tableau for the desired operation
    logical_tableau = circuit.to_tableau()
    encoding_tableau = get_ecoding_tableau(code)
    tableau = encoding_tableau.inverse().then(logical_tableau).then(encoding_tableau)
    print()
    print(tableau)

    # identify the qubit permutation matrix
    x2x, x2z, z2x, z2z, x_signs, z_signs = tableau.to_numpy()
    perm_mat = (x2x | x2z | z2x | z2z).T
    ones = np.ones(code.num_qubits)
    if not (perm_mat.sum(0) == ones).all() or not (perm_mat.sum(1) == ones).all():
        # a SWAP-transversal implementation of this circuit does not exist
        return None

    print()
    print(perm_mat.astype(int))

    # apply identify swaps
    output_circuit = stim.Circuit()
    current_loc_to_qubit = np.arange(circuit.num_qubits)
    desired_qubit_to_loc = np.argmax(perm_mat, axis=0)
    desired_loc_to_qubit = np.argmax(perm_mat, axis=1)
    for old_loc in range(circuit.num_qubits):
        while (qubit := current_loc_to_qubit[old_loc]) != desired_loc_to_qubit[old_loc]:
            new_loc = desired_qubit_to_loc[qubit]
            output_circuit.append("SWAP", [old_loc, new_loc])
            current_loc_to_qubit[[old_loc, new_loc]] = current_loc_to_qubit[[new_loc, old_loc]]

    print()
    print(output_circuit)
    print()
    print(output_circuit.to_tableau())
    return
