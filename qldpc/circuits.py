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
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import stim

from qldpc import codes
from qldpc.objects import Pauli


def restrict_to_qubits(func: Callable[..., stim.Circuit]) -> Callable[..., stim.Circuit]:
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
    code: codes.QuditCode, string: stim.PauliString | None = None
) -> stim.Tableau:
    """Circuit to encode a logical all-|0> state of the given code.

    If provided a Pauli string, prepare a logical +1 eigenstate of that logical Pauli string.
    """
    return get_ecoding_tableau(code, string).to_circuit()
