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

import collections
import functools
from collections.abc import Callable, Collection

import numpy as np
import numpy.typing as npt
import stim

from qldpc import abstract, codes
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
    """Convert an integer array that represents a Pauli string into a stim.PauliString.

    The (first, second) half the array indicates the support of (X, Z) Paulis, unless flip_xz=True.
    """
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
def get_encoding_tableau(
    code: codes.QuditCode, string: stim.PauliString | None = None
) -> stim.Circuit:
    """Tableau to prepare a logical all-|0> state of a code from an all-|0> state of its qubits.

    If provided a Pauli string, prepare a logical +1 eigenstate of that logical Pauli string.
    """
    string = string if isinstance(string, stim.PauliString) else stim.PauliString(code.dimension)
    assert len(string) == code.dimension

    # identify logical operators that stabilize our target state
    logical_ops = code.get_logical_ops()
    logical_stabs = []
    for qubit, pauli in enumerate(string):
        if pauli == 1:  # X
            logical_op = logical_ops[0, qubit]
        elif pauli == 3 or pauli == 0:  # Z or I
            logical_op = logical_ops[1, qubit]
        else:
            assert pauli == 2  # Y
            logical_op = logical_ops[0, qubit] + logical_ops[1, qubit]
        logical_stabs.append(op_to_string(logical_op))

    code_stabs = [op_to_string(row, flip_xz=True) for row in code.matrix]
    return stim.Tableau.from_stabilizers(logical_stabs + code_stabs, allow_redundant=True)


@restrict_to_qubits
def get_encoding_circuit(
    code: codes.QuditCode, string: stim.PauliString | None = None
) -> stim.Tableau:
    """Circuit to prepare a logical all-|0> state of a code from an all-|0> state of its qubits.

    If provided a Pauli string, prepare a logical +1 eigenstate of that logical Pauli string.
    """
    return get_encoding_tableau(code, string).to_circuit()


@restrict_to_qubits
def get_transversal_ops(
    code: codes.QuditCode,
    local_gates: Collection[str] = ("H", "S"),
    *,
    allow_swaps: bool = True,
    remove_redundancies: bool = True,
) -> tuple[list[stim.Tableau], list[stim.Circuit]]:
    """Logical tableaus and physical circuits for transversal logical Clifford gates of a code.

    Here local_gates must be a subset of {"H", "S", "SQRT_X"}, and if allow_swaps is True, then SWAP
    gates are considered "free" (transversal).

    Transversal logical Clifford gates are identified via the code automorphism methods in
    https://arxiv.org/abs/2409.18175.
    """
    group_aut = get_transversal_automorphism_group(code, local_gates, allow_swaps=allow_swaps)

    logical_tableaus = []
    physical_circuits = []
    for generator in group_aut.generators:
        logical_tableau, physical_circuit = _get_transversal_automorphism_data(
            code, local_gates, generator
        )
        if not remove_redundancies or not (
            _is_pauli_tableau(logical_tableau) or logical_tableau in logical_tableaus
        ):
            logical_tableaus.append(logical_tableau)
            physical_circuits.append(physical_circuit)

    return logical_tableaus, physical_circuits


def _is_pauli_tableau(tableau: stim.Tableau) -> bool:
    """Does this Tableau represent a Pauli operator?"""
    identity_mat = np.identity(len(tableau), dtype=bool)
    x2x, x2z, z2x, z2z, *_ = tableau.to_numpy()
    return (
        np.array_equal(identity_mat, x2x)
        and np.array_equal(identity_mat, z2z)
        and not np.any(x2z)
        and not np.any(z2x)
    )


@restrict_to_qubits
def get_transversal_automorphism_group(
    code: codes.QuditCode, local_gates: Collection[str] = ("H", "S"), *, allow_swaps: bool = True
) -> abstract.Group:
    """Get the transversal automorphism group of a QuditCode, using the methods of arXiv.2409.18175.

    The transversal automorphism group of a QuditCode is the group of logical Clifford operations
    that can be implemented transversally with a given local gate set.

    Here local_gates must be a subset of {"H", "S", "SQRT_X"}, and if allow_swaps is True, then SWAP
    gates are considered "free" (transversal).

    Uses the methods of https://arxiv.org/abs/2409.18175.
    """
    local_gates = _standardize_local_gates(local_gates)

    # compute the automorphism group of the "augmented" code for a transversal gate set
    matrix_z = code.matrix.reshape(code.num_checks, 2, len(code))[:, 0, :]
    matrix_x = code.matrix.reshape(code.num_checks, 2, len(code))[:, 1, :]
    if not local_gates or local_gates == {"H"}:
        # swapping sectors = swapping Z <--> X
        augmented_matrix = np.hstack([matrix_z, matrix_x])
    elif local_gates == {"S"}:
        # swapping sectors = swapping X <--> Y
        augmented_matrix = np.hstack([matrix_z, matrix_z + matrix_x])
    elif local_gates == {"SQRT_X"}:
        # swapping sectors = swapping Y <--> Z
        augmented_matrix = np.hstack([matrix_z + matrix_x, matrix_x])
    else:
        # we have a complete local Clifford gate set that can arbitrarily permute Pauli ops
        augmented_matrix = np.hstack([matrix_z, matrix_x, matrix_z + matrix_x])

    code_checks = codes.ClassicalCode(augmented_matrix)
    group_checks = code_checks.get_automorphism_group()

    # identify the group of augmented code transformations generated by the gate set
    num_sectors = augmented_matrix.shape[1] // len(code)
    column_perms = []
    if allow_swaps:
        # allow swapping qudits, which swaps corresponding columns in all "sectors" concurrently
        column_perms += [
            [(ss * len(code) + qq, ss * len(code) + qq + 1) for ss in range(num_sectors)]
            for qq in range(len(code) - 1)
        ]
    if local_gates:
        # local gates can permute "sectors" arbitrarily and independently on any qudit
        column_perms += [
            [(ss * len(code) + qq, (ss + 1) * len(code) + qq)]
            for ss in range(num_sectors - 1)
            for qq in range(len(code))
        ]
    group_gates = abstract.Group(*map(abstract.GroupMember, column_perms))

    # intersect the groups above to find the group generated by a SWAP-transversal gate set
    group_checks_sympy = group_checks.to_sympy()
    group_gates_sympy = group_gates.to_sympy()
    group_aut_sympy = group_checks_sympy.subgroup_search(group_gates_sympy.contains)
    return abstract.Group.from_sympy(group_aut_sympy, field=code.field.order)


@restrict_to_qubits
def _get_transversal_automorphism_data(
    code: codes.QuditCode,
    local_gates: Collection[str],
    automorphism: abstract.GroupMember,
) -> tuple[stim.Tableau, stim.Circuit]:
    """Logical tableau and physical circuit for a transversal automorphism of a code.

    Here local_gates must be the same as that used to construct the automorphism group.
    """
    # construct a circuit with the desired action modulo destabilizers
    physical_circuit = stim.Circuit()
    physical_circuit += _get_pauli_permutation_circuit(code, local_gates, automorphism)
    physical_circuit += _get_swap_circuit(code, automorphism)

    # make sure that the physical circuit acts on all physial qubits
    if physical_circuit.num_qubits < len(code):
        physical_circuit.append("I", len(code) - 1)

    # Determine the effect of physical_circuit on "decoded" qubits, for which
    # logicals, stabilizers, and destabilizers are single-qubit Paulis.
    encoder = get_encoding_tableau(code)
    decoder = encoder.inverse()
    decoded_tableau = encoder.then(physical_circuit.to_tableau()).then(decoder)

    # Identify Pauli corrections to the circuit: a product of destabilizers whose correspoding
    # stabilizers change sign under the physical_circuit.
    decoded_correction = "_" * code.dimension
    for aa in range(code.dimension, len(code)):
        decoded_stabilizer = stim.PauliString("_" * aa + "Z" + "_" * (len(code) - aa - 1))
        decoded_string = decoded_stabilizer.after(decoded_tableau, targets=range(len(code)))
        decoded_correction += "_" if decoded_string.sign == -1 else "X"
    correction = stim.PauliString(decoded_correction).after(encoder, targets=range(len(code)))

    # prepend the Pauli correction to the circuit
    correction_circuit = stim.Circuit()
    for pauli in ["X", "Y", "Z"]:
        for index in correction.pauli_indices(pauli):
            correction_circuit.append(pauli, index)
    physical_circuit = correction_circuit + physical_circuit

    # Identify the logical tableau implemented by the physical circuit, which is simply
    # the "upper left" block of the decoded tableau.
    decoded_tableau = encoder.then(physical_circuit.to_tableau()).then(decoder)
    x2x, x2z, z2x, z2z, x_signs, z_signs = decoded_tableau.to_numpy()
    logical_tableau = stim.Tableau.from_numpy(
        x2x=x2x[: code.dimension, : code.dimension],
        x2z=x2z[: code.dimension, : code.dimension],
        z2x=z2x[: code.dimension, : code.dimension],
        z2z=z2z[: code.dimension, : code.dimension],
        x_signs=x_signs[: code.dimension],
        z_signs=z_signs[: code.dimension],
    )

    # sanity checks: the images of stabilizers and logicals do not contain destabilizers
    assert not np.any(z2x[:, code.dimension :])  # stabilizers and Z-type logicals
    assert not np.any(x2x[: code.dimension, code.dimension :])  # X-type logicals

    return logical_tableau, physical_circuit


def _get_swap_circuit(code: codes.QuditCode, automorphism: abstract.GroupMember) -> stim.Circuit:
    """Construct the circuit of SWAPs applied by a transversal automorphism."""
    circuit = stim.Circuit()
    new_locs = [automorphism(qubit) % len(code) for qubit in range(len(code))]
    for cycle in abstract.GroupMember(new_locs).cyclic_form:
        loc_0 = cycle[0]
        for loc_1 in cycle[1:]:
            circuit.append("SWAP", [loc_0, loc_1])
    return circuit


@restrict_to_qubits
def _get_pauli_permutation_circuit(
    code: codes.QuditCode,
    local_gates: Collection[str],
    automorphism: abstract.GroupMember,
) -> stim.Circuit:
    """Construct the circuit of local Pauli permutations applied by a transversal automorphism."""
    local_gates = _standardize_local_gates(local_gates)
    circuit = stim.Circuit()

    if len(local_gates) == 1:
        # there is only one local gate, and all it can do is permute two parity check sectors
        gate = next(iter(local_gates))
        for qubit in range(len(code)):
            if automorphism(qubit) >= len(code):
                circuit.append(gate, qubit)

    elif len(local_gates) > 1:
        # we have a complete local Clifford gate set that can permute Pauli sectors arbitrarily
        gate_targets = collections.defaultdict(list)
        for qubit in range(len(code)):
            pauli_perm = [automorphism(qubit + ss * len(code)) // len(code) for ss in range(3)]
            match pauli_perm:
                case [1, 0, 2]:  # Z <--> X
                    gate_targets["H"].append(qubit)
                case [2, 1, 0]:  # X <--> Y
                    gate_targets["S"].append(qubit)
                case [0, 2, 1]:  # Y <--> Z
                    gate_targets["H_YZ"].append(qubit)
                case [1, 2, 0]:  # ZXY <--> XYZ
                    gate_targets["C_XYZ"].append(qubit)  # pragma: no cover
                case [2, 0, 1]:  # ZXY <--> ZYX
                    gate_targets["C_ZYX"].append(qubit)  # pragma: no cover

        for gate, targets in gate_targets.items():
            circuit.append(gate, sorted(targets))

    return circuit


def _standardize_local_gates(local_gates: Collection[str]) -> set[str]:
    """Standardize a local Clifford gate set."""
    allowed_gates = {"S", "H", "SQRT_X"}
    if not allowed_gates.issuperset(local_gates):
        raise ValueError(
            f"Local Clifford gates (provided: {local_gates}) must be subset of {allowed_gates}"
        )
    return set(local_gates)
