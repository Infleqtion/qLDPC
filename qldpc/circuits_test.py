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

import unittest.mock

import pytest
import stim

from qldpc import abstract, circuits, codes


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
    """Prepare eigenstates of logical Pauli strings."""
    for code in [
        codes.FiveQubitCode(),
        codes.HGPCode(codes.HammingCode(3)),
        codes.HGPCode(codes.ClassicalCode.random(5, 3)),
    ]:
        logical_string = stim.PauliString.random(code.dimension)
        circuit = circuits.get_encoding_circuit(code, logical_string)

        # test stabilizers of the code
        simulator = stim.TableauSimulator()
        simulator.do(circuit)
        for row in code.matrix:
            string = circuits.op_to_string(row, flip_xz=True)
            assert simulator.peek_observable_expectation(string) == 1

        # test the logical string
        simulator.do(circuit.inverse())
        simulator.do(logical_string)
        simulator.do(circuit)
        assert simulator.peek_observable_expectation(string) == 1


def test_transversal_ops() -> None:
    """Construct SWAP-transversal logical Cliffords of a code."""
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
        with unittest.mock.patch(
            "qldpc.codes.ClassicalCode.get_automorphism_group",
            return_value=abstract.Group(*map(abstract.GroupMember, group_aut_gens)),
        ):
            logical_tableaus, physical_circuits = circuits.get_transversal_ops(code, local_gates)
            assert len(logical_tableaus) == len(physical_circuits) == len(local_gates) - 1

    with pytest.raises(ValueError, match="Local Clifford gates"):
        circuits.get_transversal_automorphism_group(code, ["SQRT_Y"])
