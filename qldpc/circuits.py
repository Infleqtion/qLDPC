"""Tools for constructing syndrome extraction circuits

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
import cirq

from qldpc import codes

rep_code = codes.BitCode.ring(3)
code = codes.HGPCode(rep_code)


circuit = cirq.Circuit()

for ancilla in [qubit for qubit in code.graph.nodes if not qubit.is_data]:
    string = {
        cirq.NamedQubit(str(neighbor)): str(code.graph[ancilla][neighbor][codes.Pauli])
        for neighbor in code.graph.successors(ancilla)
    }
    circuit += cirq.H(cirq.NamedQubit(str(ancilla)))
    circuit += cirq.PauliString(string).controlled_by(cirq.NamedQubit(str(ancilla)))
    circuit += cirq.H(cirq.NamedQubit(str(ancilla)))
    print(circuit)
    exit()
