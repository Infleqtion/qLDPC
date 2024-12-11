"""Unit tests for circuits.py

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

import contextlib
import unittest.mock

import numpy as np
import pytest
import stim

from qldpc import circuits, codes, external
from qldpc.objects import Pauli


def test_restriction() -> None:
    """Raise an error for non-qubit codes."""
    code = codes.SurfaceCode(2, field=3)
    with pytest.raises(ValueError, match="only supported for qubit codes"):
        circuits.get_encoding_circuit(code)


def test_pauli_strings() -> None:
    """Stabilizers correctly converted into stim.PauliString objects."""
    code = codes.FiveQubitCode()
    assert all(
        circuits.op_to_string(row, flip_xz=True) == stim.PauliString(stabilizer.replace(" ", ""))
        for row, stabilizer in zip(code.matrix, code.get_stabilizers())
    )


def test_state_prep() -> None:
    """Prepare all-0 logical states of qubit codes."""
    for code in [
        codes.FiveQubitCode(),
        codes.HGPCode(codes.HammingCode(3)),
        codes.HGPCode(codes.ClassicalCode.random(5, 3)),
    ]:
        encoder = circuits.get_encoding_circuit(code)
        simulator = stim.TableauSimulator()
        simulator.do(encoder)

        # the state of the simulator is a +1 eigenstate of code stabilizers
        for row in code.matrix:
            string = circuits.op_to_string(row, flip_xz=True)
            assert simulator.peek_observable_expectation(string) == 1

        # the state of the simulator is a +1 eigenstate of all logical Z operators
        for op in code.get_logical_ops(Pauli.Z):
            string = circuits.op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1


def test_transversal_ops() -> None:
    """Construct SWAP-transversal logical Cliffords of a code."""
    gap_is_installed = external.gap.is_installed()
    code = codes.FiveQubitCode()

    gate_gens = {
        ("SWAP", "S"): [
            [[3, 7], [4, 5], [8, 9]],
            [[1, 5], [2, 7], [3, 9], [6, 8]],
            [[2, 6], [3, 8], [4, 5], [7, 9]],
            [[0, 9, 2, 8], [1, 6], [3, 5, 4, 7]],
            [[0, 7, 3], [1, 9, 8], [2, 4, 5]],
        ],
        ("SWAP", "H"): [
            [[2, 5], [4, 6], [7, 9]],
            [[1, 9], [3, 5], [6, 8]],
            [[1, 8], [4, 7], [6, 9]],
            [[1, 4, 9, 8, 7, 6], [2, 5, 3]],
            [[1, 7], [2, 3], [4, 8]],
            [[0, 3, 1, 4, 9, 5], [2, 8, 6]],
            [[0, 6, 8], [1, 9, 2], [3, 5, 7]],
        ],
        ("SWAP", "SQRT_X"): [
            [[3, 7], [4, 5], [8, 9]],
            [[1, 5], [2, 7], [3, 9], [6, 8]],
            [[2, 6], [3, 8], [4, 5], [7, 9]],
            [[0, 9, 2, 8], [1, 6], [3, 5, 4, 7]],
            [[0, 7, 3], [1, 9, 8], [2, 4, 5]],
        ],
        ("SWAP", "H", "S"): [
            [[0, 1], [7, 10], [9, 11], [12, 14]],
            [[2, 3], [6, 13], [9, 12], [11, 14]],
            [[1, 2], [5, 12], [8, 11], [10, 13]],
            [[1, 8], [4, 7], [6, 9], [12, 13]],
            [[3, 5], [4, 13], [7, 12], [10, 14]],
            [[2, 9], [3, 12], [6, 11], [13, 14]],
            [[2, 6], [3, 12], [4, 7], [5, 10], [9, 11], [13, 14]],
            [[3, 12], [4, 10], [5, 7], [13, 14]],
            [[3, 14], [4, 7], [5, 10], [12, 13]],
        ],
    }
    for local_gates, group_aut_gens in gate_gens.items():
        with (
            unittest.mock.patch("qldpc.external.groups.get_generators", return_value=group_aut_gens)
            if not gap_is_installed
            else contextlib.nullcontext()
        ):
            logical_tableaus, physical_circuits = circuits.get_transversal_ops(code, local_gates)
            assert len(logical_tableaus) == len(physical_circuits) == len(local_gates) - 1

    with pytest.raises(ValueError, match="Local Clifford gates"):
        circuits.get_transversal_automorphism_group(code, ["SQRT_Y"])


def test_finding_circuit(pytestconfig: pytest.Config) -> None:
    """Find a physical circuit for a desired logical Clifford operation."""
    gap_is_installed = external.gap.is_installed()
    np.random.seed(pytestconfig.getoption("randomly_seed") if gap_is_installed else 0)

    # code with randomly permuted qubits
    base_code = codes.FiveQubitCode()
    matrix = base_code.matrix.reshape(-1, len(base_code))
    permutation = np.eye(len(base_code), dtype=int)[np.random.permutation(len(base_code))]
    permuted_matrix = (matrix @ base_code.field(permutation)).reshape(-1, 2 * len(base_code))
    code = codes.QuditCode(permuted_matrix)

    # logical circuit: random single-qubit Clifford recognized by Stim
    logical_op = np.random.choice(
        [
            "X",
            "Y",
            "Z",
            "C_XYZ",
            "C_ZYX",
            "H",
            "H_XY",
            "H_XZ",
            "H_YZ",
            "S",
            "SQRT_X",
            "SQRT_X_DAG",
            "SQRT_Y",
            "SQRT_Y_DAG",
            "SQRT_Z",
            "SQRT_Z_DAG",
            "S_DAG",
        ]
    )
    logical_circuit = stim.Circuit(f"{logical_op} 0")

    # construct physical circuit
    group_aut_gens = [
        [[0, 1], [7, 11], [8, 10], [12, 13]],
        [[2, 3], [5, 6], [7, 10], [8, 11]],
        [[1, 2], [6, 13], [7, 14], [8, 9]],
        [[1, 9], [4, 12], [5, 10], [6, 7]],
        [[3, 13], [4, 7], [6, 12], [11, 14]],
        [[2, 8], [4, 12], [5, 10], [13, 14]],
        [[2, 5], [3, 7], [4, 12], [6, 11], [8, 10], [13, 14]],
        [[3, 7], [4, 13], [6, 11], [12, 14]],
        [[3, 11], [4, 12], [6, 7], [13, 14]],
    ]
    with (
        unittest.mock.patch("qldpc.external.groups.get_generators", return_value=group_aut_gens)
        if not gap_is_installed
        else contextlib.nullcontext()
    ):
        physical_circuit = circuits.maybe_get_transversal_circuit(code, logical_circuit)

    # check that the physical circuit has the correct logical tableau
    encoder = circuits.get_encoding_tableau(code)
    decoder = encoder.inverse()
    decoded_physical_tableau = encoder.then(physical_circuit.to_tableau()).then(decoder)
    x2x, x2z, z2x, z2z, x_signs, z_signs = decoded_physical_tableau.to_numpy()
    reconstructed_logical_tableau = stim.Tableau.from_numpy(
        x2x=x2x[: code.dimension, : code.dimension],
        x2z=x2z[: code.dimension, : code.dimension],
        z2x=z2x[: code.dimension, : code.dimension],
        z2z=z2z[: code.dimension, : code.dimension],
        x_signs=x_signs[: code.dimension],
        z_signs=z_signs[: code.dimension],
    )
    assert logical_circuit.to_tableau() == reconstructed_logical_tableau
