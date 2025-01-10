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
from collections.abc import Callable, Collection, Sequence

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

    The (first, second) half the array indicates the support of (X, Z) Paulis, unless flip_xz==True.
    """
    assert len(op) % 2 == 0
    support_xz = np.array(op, dtype=int).reshape(2, -1)
    if flip_xz:
        support_xz = support_xz[::-1, :]
    paulis = [Pauli((support_xz[0, qq], support_xz[1, qq])) for qq in range(support_xz.shape[1])]
    return stim.PauliString(map(str, paulis))

    num_qubits = len(op) // 2
    paulis = ""
    for qubit in range(num_qubits):
        val_x = int(op[qubit])
        val_z = int(op[qubit + num_qubits])
        pauli = Pauli((val_x, val_z))
        paulis += str(pauli if not flip_xz else ~pauli)
    return stim.PauliString(paulis)


@restrict_to_qubits
def get_encoding_tableau(code: codes.QuditCode) -> stim.Circuit:
    """Tableau to prepare an all-|0> logical state of a code from an all-|0> state of its qubits."""
    logical_ops_z = [op_to_string(op) for op in code.get_logical_ops(Pauli.Z)]
    code_stabs = [op_to_string(row, flip_xz=True) for row in code.matrix]
    return stim.Tableau.from_stabilizers(
        logical_ops_z + code_stabs, allow_redundant=True, allow_underconstrained=False
    )


@restrict_to_qubits
def get_encoding_circuit(code: codes.QuditCode) -> stim.Tableau:
    """Circuit to prepare an all-|0> logical state of a code from an all-|0> state of its qubits."""
    return get_encoding_tableau(code).to_circuit()


@restrict_to_qubits
def get_transversal_ops(
    code: codes.QuditCode,
    local_gates: Collection[str] = ("S", "H", "SWAP"),
    *,
    remove_redundancies: bool = True,
) -> list[tuple[stim.Tableau, stim.Circuit]]:
    """Logical tableaus and physical circuits for transversal logical Clifford gates of a code.

    Here local_gates must be a subset of {"S", "H", "SQRT_X", "SWAP"}.

    Transversal logical Clifford gates are identified via the code automorphism methods in
    https://arxiv.org/abs/2409.18175.
    """
    group_aut = get_transversal_automorphism_group(code, local_gates)

    transversal_ops: list[tuple[stim.Tableau, stim.Circuit]] = []
    for generator in group_aut.generators:
        logical_tableau, physical_circuit = _get_transversal_automorphism_data(
            code, generator, local_gates
        )
        if not remove_redundancies or not (
            _is_pauli_tableau(logical_tableau)
            or any(
                _tableaus_are_equivalent_mod_paulis(logical_tableau, tableau)
                for tableau, _ in transversal_ops
            )
        ):
            transversal_ops.append((logical_tableau, physical_circuit))

    return transversal_ops


def _is_pauli_tableau(tableau: stim.Tableau) -> bool:
    """Does this Tableau represent a Pauli operator?

    If so, it maps Pauli strings to themselves up to sign.
    """
    identity_mat = np.identity(len(tableau), dtype=bool)
    x2x, x2z, z2x, z2z, *_ = tableau.to_numpy()
    return (
        np.array_equal(identity_mat, x2x)
        and np.array_equal(identity_mat, z2z)
        and not np.any(x2z)
        and not np.any(z2x)
    )


def _tableaus_are_equivalent_mod_paulis(tableau_1: stim.Tableau, tableau_2: stim.Tableau) -> bool:
    """Are the two stabilizer tableaus equivalent, up to Paulis?"""
    x2x_1, x2z_1, z2x_1, z2z_1, *_ = tableau_1.to_numpy()
    x2x_2, x2z_2, z2x_2, z2z_2, *_ = tableau_2.to_numpy()
    return (
        np.array_equal(x2x_1, x2x_2)
        and np.array_equal(x2z_1, x2z_2)
        and np.array_equal(z2x_1, z2x_2)
        and np.array_equal(z2z_1, z2z_2)
    )


@restrict_to_qubits
def get_transversal_automorphism_group(
    code: codes.QuditCode, local_gates: Collection[str] = ("S", "H", "SWAP")
) -> abstract.Group:
    """Get the transversal automorphism group of a QuditCode, using the methods of arXiv.2409.18175.

    The transversal automorphism group of a QuditCode is the group of logical Clifford operations
    that can be implemented transversally with a given local gate set.

    Here local_gates must be a subset of {"S", "H", "SQRT_X", "SWAP"}.

    Uses the methods of https://arxiv.org/abs/2409.18175.
    """
    local_gates = _standardize_local_gates(local_gates)
    allow_swaps = "SWAP" in local_gates
    local_gates.discard("SWAP")

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
    group_aut_sympy = group_checks.to_sympy().subgroup_search(group_gates.to_sympy().contains)
    return abstract.Group.from_sympy(group_aut_sympy, field=code.field.order)


@restrict_to_qubits
def _get_transversal_automorphism_data(
    code: codes.QuditCode,
    automorphism: abstract.GroupMember,
    local_gates: Collection[str],
) -> tuple[stim.Tableau, stim.Circuit]:
    """Logical tableau and physical circuit for a transversal automorphism of a code.

    Here local_gates must be the same as that used to construct the automorphism group.
    """
    # construct a circuit with the desired action modulo destabilizers
    physical_circuit = stim.Circuit()
    physical_circuit += _get_pauli_permutation_circuit(code, automorphism, local_gates)
    physical_circuit += _get_swap_circuit(code, automorphism)

    # make sure that the physical circuit acts on all physial qubits
    if physical_circuit.num_qubits < len(code):
        physical_circuit.append("I", len(code) - 1)

    # Determine the effect of physical_circuit on "decoded" qubits, for which
    # logicals, stabilizers, and destabilizers are single-qubit Paulis.
    encoder = get_encoding_tableau(code)
    decoder = encoder.inverse()
    decoded_tableau = encoder.then(physical_circuit.to_tableau()).then(decoder)

    # Prepend Pauli corrections to the circuit: a product of destabilizers whose correspoding
    # stabilizers change sign under the physical_circuit.
    decoded_correction = "_" * code.dimension  # identity on the logical qubits
    for aa in range(code.dimension, len(code)):
        decoded_stabilizer_before = "_" * aa + "Z" + "_" * (len(code) - aa - 1)
        decoded_stabilizer_after = decoded_tableau(stim.PauliString(decoded_stabilizer_before))
        decoded_correction += "_" if decoded_stabilizer_after.sign == 1 else "X"
    correction = encoder(stim.PauliString(decoded_correction))
    physical_circuit = _get_pauli_circuit(correction) + physical_circuit

    # Identify the logical tableau implemented by the physical circuit, which is simply
    # the "upper left" block of the decoded tableau that acts on all logical qubits.
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


def _standardize_local_gates(local_gates: Collection[str]) -> set[str]:
    """Standardize a local Clifford gate set."""
    allowed_gates = {"S", "H", "SQRT_X", "SWAP"}
    if not allowed_gates.issuperset(local_gates):
        raise ValueError(
            f"Local Clifford gates (provided: {local_gates}) must be subset of {allowed_gates}"
        )
    return set(local_gates)


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
    automorphism: abstract.GroupMember,
    local_gates: Collection[str],
) -> stim.Circuit:
    """Construct the circuit of local Pauli permutations applied by a transversal automorphism."""
    local_gates = _standardize_local_gates(local_gates)
    local_gates.discard("SWAP")

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
                    gate_targets["C_ZYX"].append(qubit)  # pragma: no cover
                case [2, 0, 1]:  # ZXY <--> ZYX
                    gate_targets["C_XYZ"].append(qubit)  # pragma: no cover

        for gate, targets in gate_targets.items():
            circuit.append(gate, sorted(targets))

    return circuit


def _get_pauli_circuit(string: stim.PauliString) -> stim.Circuit:
    """Stim circuit to apply a Pauli string."""
    circuit = stim.Circuit()
    for pauli in ["X", "Y", "Z"]:
        if indices := string.pauli_indices(pauli):
            circuit.append(pauli, indices)
    return circuit


@restrict_to_qubits
def get_transversal_circuits(
    code: codes.QuditCode,
    logical_circuits_or_tableaus: Sequence[stim.Circuit | stim.Tableau],
    local_gates: Collection[str] = ("S", "H", "SWAP"),
) -> stim.Circuit | None:
    """Find a transversal physical circuits (if any) to implement given logical Clifford operations.

    Here local_gates must be a subset of {"S", "H", "SQRT_X", "SWAP"}.

    Warning: this method performs a brute-force search over the Clifford automorphisms of a code,
    and thereby generally has exponential runtime.
    """
    physical_circuits = [None] * len(logical_circuits_or_tableaus)

    # convert logical Cliffords into tableaus
    identity = stim.Circuit(f"I {code.dimension - 1}")  # to ensure circuits address all qubits
    logical_tableaus = [
        (
            logical_circuit_or_tableau
            if isinstance(logical_circuit_or_tableau, stim.Tableau)
            else (logical_circuit_or_tableau + identity).to_tableau()
        )
        for logical_circuit_or_tableau in logical_circuits_or_tableaus
    ]

    # compute the group of transversal Cliffords
    group_aut = get_transversal_automorphism_group(code, local_gates)

    # perform a brute-force search for matching Clifford operations
    matching_ops: list[tuple[stim.Tableau, stim.Circuit] | None] = [None] * len(logical_tableaus)
    for automorphism in group_aut.generate():
        tableau, circuit = _get_transversal_automorphism_data(code, automorphism, local_gates)
        for tt, logical_tableau in enumerate(logical_tableaus):
            if matching_ops[tt] is None and _tableaus_are_equivalent_mod_paulis(
                logical_tableau, tableau
            ):
                matching_ops[tt] = tableau, circuit
        if not any(op is None for op in matching_ops):
            break

    # add logical Pauli corrections to matching circuits
    for tt, (logical_tableau, matching_op) in enumerate(zip(logical_tableaus, matching_ops)):
        if matching_op is None:
            continue
        matching_tableau, matching_circuit = matching_op

        correction = code.field([0] * (2 * len(code)))
        *_, x_signs_l, z_signs_l = logical_tableau.to_numpy()
        *_, x_signs_m, z_signs_m = matching_tableau.to_numpy()
        for logical_qubit in range(code.dimension):
            if x_signs_l[logical_qubit] != x_signs_m[logical_qubit]:  # pragma: no cover
                correction = correction + code.get_logical_ops(Pauli.Z)[logical_qubit]
            if z_signs_l[logical_qubit] != z_signs_m[logical_qubit]:  # pragma: no cover
                correction += code.get_logical_ops(Pauli.X)[logical_qubit]
        correction_circuit = _get_pauli_circuit(op_to_string(correction))

        physical_circuits[tt] = correction_circuit + matching_circuit

    return physical_circuits


def get_transversal_circuit(
    code: codes.QuditCode,
    logical_circuit_or_tableau: stim.Circuit | stim.Tableau,
    local_gates: Collection[str] = ("S", "H", "SWAP"),
) -> stim.Circuit | None:
    """Find a transversal physical circuit (if any) to implement a logical Clifford operation.

    Here local_gates must be a subset of {"S", "H", "SQRT_X", "SWAP"}.

    Warning: this method performs a brute-force search over the Clifford automorphisms of a code,
    and thereby generally has exponential runtime.
    """
    return get_transversal_circuits(code, [logical_circuit_or_tableau], local_gates)[0]
