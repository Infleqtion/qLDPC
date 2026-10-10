# SPDX-License-Identifier: Apache-2.0

"""Shared pytest helpers for circuits tests."""

from __future__ import annotations

from typing import Self

import stim

from qldpc.circuits.noise_model.moments import _gate_qubits


class StimCircuitWrapper:
    """A minimal stim-wrapping circuit (like tsim.Circuit) for exercising StimCircuitProtocol.

    It satisfies StimCircuitProtocol (append / __iadd__ / indexing / len) and is
    default-constructible, so circuit-polymorphic functions accept it and return the same type.  The
    ``stim_circuit`` attribute is not required by those functions; it just lets tests read back the
    wrapped circuit to check results.
    """

    def __init__(self, stim_circuit: stim.Circuit | None = None) -> None:
        self.stim_circuit = stim.Circuit() if stim_circuit is None else stim_circuit

    def append(self, *args: object, **kwargs: object) -> None:
        self.stim_circuit.append(*args, **kwargs)

    def __iadd__(self, other: stim.Circuit) -> Self:
        self.stim_circuit += other
        return self

    def __getitem__(self, index: int) -> stim.CircuitInstruction | stim.CircuitRepeatBlock:
        return self.stim_circuit[index]

    def __len__(self) -> int:
        return len(self.stim_circuit)


def circuits_are_equivalent(
    circuit_a: stim.Circuit, circuit_b: stim.Circuit, atol: float = 1e-10
) -> bool:
    """Test equivalence between circuits after standardizing them with a trivial noise model."""
    from qldpc.circuits import NoiseModel

    trivial_noise_model = NoiseModel()
    circuit_a = trivial_noise_model.noisy_circuit(circuit_a)
    circuit_b = trivial_noise_model.noisy_circuit(circuit_b)
    return circuit_a.approx_equals(circuit_b, atol=atol)


def flatten_repeat_blocks(circuit: stim.Circuit, keep_tag: str | None = None) -> stim.Circuit:
    """Flatten the repeat blocks of a circuit, except for those whose tag contains keep_tag."""
    flattened = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock) and (
            keep_tag is None or keep_tag not in instruction.tag
        ):
            body = flatten_repeat_blocks(instruction.body_copy(), keep_tag)
            for _ in range(instruction.repeat_count):
                flattened += body
        else:
            flattened.append(instruction)
    return flattened


def flatten_repeat_blocks_with_seam_ticks(
    circuit: stim.Circuit, immune_op_tag: str
) -> stim.Circuit:
    """Flatten non-immune repeat blocks, inserting TICKs to prevent qubit reuse across their seams.

    A TICK is inserted wherever a gate would reuse a qubit that was used (since the last TICK)
    before the boundary of a repeat block or of one of its iterations.
    """
    flattened = stim.Circuit()
    used_qubits: set[int] = set()
    boundary_qubits: set[int] = set()

    def append(circuit: stim.Circuit) -> None:
        nonlocal boundary_qubits
        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock) and immune_op_tag not in op.tag:
                for _ in range(op.repeat_count):
                    boundary_qubits = used_qubits.copy()
                    append(op.body_copy())
                boundary_qubits = used_qubits.copy()
                continue
            if isinstance(op, stim.CircuitRepeatBlock) or op.name == "TICK":
                used_qubits.clear()
                boundary_qubits.clear()
            elif qubits := _gate_qubits(op):
                if qubits & boundary_qubits:
                    flattened.append("TICK")
                    used_qubits.clear()
                    boundary_qubits.clear()
                used_qubits.update(qubits)
            flattened.append(op)

    append(circuit)
    return flattened
