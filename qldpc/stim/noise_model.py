from __future__ import annotations

import abc
import stim

from qldpc.stim.util import Basis


class NoiseModel(abc.ABC):

    @abc.abstractmethod
    def apply_1q_gates(self, circ: stim.Circuit, gate: str, qubits: list[int]):
        """TODO: doc"""

    @abc.abstractmethod
    def apply_2q_gates(
        self, circ: stim.Circuit, gate: str, qubit_pairs: list[tuple[int, int]]
    ):
        """TODO: doc"""

    @abc.abstractmethod
    def apply_meas(self, circ: stim.Circuit, basis: Basis, qubits: list[int]):
        """TODO: doc"""

    @abc.abstractmethod
    def apply_reset(self, circ: stim.Circuit, basis: Basis, qubits: list[int]):
        """TODO: doc"""


class UniformNoiseModel(NoiseModel):

    def __init__(self, p):
        self.p = p

    def apply_1q_gates(self, circ: stim.Circuit, gate: str, qubits: list[int]):
        """
        Apply single-qubit gates followed by single-qubit errors with probability p.
        """
        circ.append(gate, qubits)
        circ.append("TICK")
        circ.append("DEPOLARIZE1", qubits, self.p)
        circ.append("TICK")

    def apply_2q_gates(
        self, circ: stim.Circuit, gate: str, qubit_pairs: list[tuple[int, int]]
    ):
        """
        Apply two-qubit gates followed by two-qubit errors with probability p.
        """
        flattened_args = [q for pair in qubit_pairs for q in pair]
        circ.append(gate, flattened_args)
        circ.append("TICK")
        circ.append("DEPOLARIZE2", flattened_args, self.p)
        circ.append("TICK")

    def apply_meas(self, circ: stim.Circuit, basis: Basis, qubits: list[int]):
        """
        Measure qubits with error probability p
        """
        if basis == basis.Z:
            error_gate = "X_ERROR"
            meas_gate = "M"
        elif basis == basis.X:
            error_gate = "Z_ERROR"
            meas_gate = "MX"
        else:
            raise ValueError(f"Invalid basis: {basis}")
        circ.append(error_gate, qubits, self.p)
        circ.append("TICK")
        circ.append(meas_gate, qubits)
        circ.append("TICK")

    def apply_reset(self, circ: stim.Circuit, basis: Basis, qubits: list[int]):
        """
        Reset qubits to 0 with error probability p
        """
        if basis == basis.Z:
            reset_gate = "R"
            error_gate = "X_ERROR"
        elif basis == basis.X:
            reset_gate = "RX"
            error_gate = "Z_ERROR"
        else:
            raise ValueError(f"Invalid basis: {basis}")
        circ.append(reset_gate, qubits)
        circ.append("TICK")
        circ.append(error_gate, qubits, self.p)
        circ.append("TICK")
