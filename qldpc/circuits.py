"""Tools for constructing circuits

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
from qldpc.objects import Pauli, PauliXZ


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
        pauli = Pauli.I
        if op[qubit]:
            pauli *= Pauli.X
        if op[qubit + num_qubits]:
            pauli *= Pauli.Z
        paulis.append(str(pauli if not flip_xz else ~pauli))
    return stim.PauliString("".join(paulis))


@restrict_to_qubits
def prep(code: codes.QuditCode, pauli: PauliXZ = Pauli.Z) -> stim.Circuit:
    """Circuit to prepare a logical +1 eigenstate of all logical operators of one type."""
    stabilizers = [op_to_string(row, flip_xz=True) for row in code.matrix]
    operators = [op_to_string(op) for op in code.get_logical_ops(pauli)]
    tableau = stim.Tableau.from_stabilizers(stabilizers + operators, allow_redundant=True)
    return tableau.to_circuit()


@restrict_to_qubits
def steane_syndrome_extraction(
    code: codes.QuditCode, error_prob: float = 0, num_cycles: int = 1
) -> stim.Circuit:
    """Noisy circuit to perform one or more Steane-type syndrome extraction cycles."""


# code = codes.FiveQubitCode()
# circuit = prep(code, Pauli.Z)
# print(circuit)
