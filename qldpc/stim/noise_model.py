from __future__ import annotations

import abc
import stim

from qldpc.objects import PauliXZ, Pauli


class NoiseModel(abc.ABC):
    """
    To implement different circuit-level models, a noise model wraps the application of stim operations.
    Subclasses should define expected error behavior for these operations (1q gate, 2q gate, meas, and reset)
    """

    @abc.abstractmethod
    def apply_1q_gates(self, circ: stim.Circuit, gate: str, qubits: list[int]):
        """
        args:
            circ: stim.Circuit
                The circuit to apply gates and errors to
            gate: str
                The single qubit gate being applied
            qubits: list[int]
                List of qubit ids gate is applied to
        """

    @abc.abstractmethod
    def apply_2q_gates(
        self, circ: stim.Circuit, gate: str, qubit_pairs: list[tuple[int, int]]
    ):
        """
        args:
            circ: stim.Circuit
                The circuit to apply gates and errors to
            gate: str
                The two qubit gate being applied
            qubits: list[tuple[int, int]]
                List of qubit pairs involved in the two qubit gate. For, e.g. CX, the first qubit is the control and the second is the target
        """

    @abc.abstractmethod
    def apply_meas(self, circ: stim.Circuit, basis: PauliXZ, qubits: list[int]):
        """
        args:
            circ: stim.Circuit
                The circuit to apply gates and errors to
            basis: PauliXZ
                The measurement basis, X or Z
            qubits: list[int]
                List of qubit ids being measured
        """

    @abc.abstractmethod
    def apply_reset(self, circ: stim.Circuit, basis: PauliXZ, qubits: list[int]):
        """
        args:
            circ: stim.Circuit
                The circuit to apply gates and errors to
            basis: PauliXZ
                The basis to reset in, X or Z
            qubits: list[int]
                List of qubit ids being reset
        """


class UniformNoiseModel(NoiseModel):
    """
    A uniform noise model where all gate, reset, and measurement errors have equal probability
    """

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

    def apply_meas(self, circ: stim.Circuit, basis: PauliXZ, qubits: list[int]):
        """
        Measure qubits with error probability p
        """
        if basis == Pauli.Z:
            error_gate = "X_ERROR"
            meas_gate = "M"
        elif basis == Pauli.X:
            error_gate = "Z_ERROR"
            meas_gate = "MX"
        else:
            raise ValueError(f"Invalid basis: {basis}")
        circ.append(error_gate, qubits, self.p)
        circ.append("TICK")
        circ.append(meas_gate, qubits)
        circ.append("TICK")

    def apply_reset(self, circ: stim.Circuit, basis: PauliXZ, qubits: list[int]):
        """
        Reset qubits to 0 with error probability p
        """
        if basis == Pauli.Z:
            reset_gate = "R"
            error_gate = "X_ERROR"
        elif basis == Pauli.X:
            reset_gate = "RX"
            error_gate = "Z_ERROR"
        else:
            raise ValueError(f"Invalid basis: {basis}")
        circ.append(reset_gate, qubits)
        circ.append("TICK")
        circ.append(error_gate, qubits, self.p)
        circ.append("TICK")
