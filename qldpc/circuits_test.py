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

import pytest
import stim

from qldpc import circuits, codes, objects


def test_restriction() -> None:
    """Raise an error for non-qubit codes."""
    code = codes.SurfaceCode(2, field=3)
    with pytest.raises(ValueError, match="only supported for qubit codes"):
        circuits.prep(code, objects.Pauli.Z)


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
        circuit = circuits.get_ecoding_circuit(code, logical_string)

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


def test_swap_transversal_circuit() -> None:
    """Test SWAP-transversal implementations of logical operations."""
    circuit = stim.Circuit()
    # circuit.append("CX", [1, 0])
    circuit.append("SWAP", [0, 1])
    circuit.append("H", [0, 1])

    code = codes.CSSCode([[1, 1, 1, 1]], [[1, 1, 1, 1]])
    print(code.get_code_params())
    print()
    print(circuit)
    print()
    circuits.get_swap_transversal_circuit(code, circuit)


test_swap_transversal_circuit()
