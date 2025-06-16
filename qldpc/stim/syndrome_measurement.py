from __future__ import annotations

import abc
from dataclasses import dataclass

import networkx as nx
import stim

from qldpc.codes.common import ClassicalCode, CSSCode
from qldpc.stim.noise_model import NoiseModel


@dataclass
class StimIds:
    data_ids: list[int]
    z_check_ids: list[int]
    x_check_ids: list[int]


class SyndromeMeasurement(abc.ABC):
    """
    Base class for a syndrome measurement scheme
    """

    @abc.abstractmethod
    def compile_sm_circuit(
        self, code: CSSCode, stim_ids: StimIds
    ) -> tuple[stim.Circuit, list[list[int]]]:
        """
        Compiles a syndrome measurement circuit for a given CSSCode and noise model.

        args:
            code: CSSCode
                The quantum code to be compiled into a single round of syndrome measurements
            stim_ids: StimIds
                Stim circuit ids to be used for data qubits, z check qubits, and x check qubits
        returns:
            stim.Circuit:
                Stim circuit containing the compiled syndrome measurement round
            list[list[int]]:
                The history of measurement rounds performed in the circuit.
                Each round is a list of the stim ids measured that round, in the order they were passed to stim.
        """


class BareColorCircuit(SyndromeMeasurement):
    """
    A coloration circuit syndrome measurement scheme as defined in https://arxiv.org/abs/2109.14609 (Algorithm 1).

    NOTE: This is not guaranteed to be distance-preserving.
    """

    def _code_to_subcircuit(
        self,
        code: ClassicalCode,
        check_ids: list[int],
        data_ids: list[int],
        gate: str,
        strategy: str,
    ) -> stim.Circuit:
        coloring = nx.coloring.greedy_color(nx.line_graph(code.graph.to_undirected()), strategy)
        circuit = stim.Circuit()

        schedule: dict[int, list[tuple[int, int]]] = {}
        for edge, color in coloring.items():
            assert edge[0].is_data ^ edge[1].is_data  # Assert valid edge (data <-> check)
            if edge[0].is_data:
                check_op = (check_ids[edge[1].index], data_ids[edge[0].index])
            else:
                check_op = (check_ids[edge[0].index], data_ids[edge[1].index])
            schedule.setdefault(color, []).append(check_op)
        for color, moment in schedule.items():
            for check_qubit, data_qubit in moment:
                circuit.append(gate, [check_qubit, data_qubit])
            if moment:  # Only add TICK if there were operations in this moment
                circuit.append("TICK")
        return circuit

    def compile_sm_circuit(
        self,
        code: CSSCode,
        stim_ids: StimIds,
        coloring_strategy: str = "largest_first",
    ) -> tuple[stim.Circuit, list[list[int]]]:
        """
        Compiles a coloration circuit. Not depth-optimal as no interleaving of opposite type checks is present. Z checks are performed first followed by X checks
        """
        z_subcircuit = self._code_to_subcircuit(
            code.code_z,
            stim_ids.z_check_ids,
            stim_ids.data_ids,
            "CZ",
            coloring_strategy,
        )
        x_subcircuit = self._code_to_subcircuit(
            code.code_x,
            stim_ids.x_check_ids,
            stim_ids.data_ids,
            "CX",
            coloring_strategy,
        )
        circuit = stim.Circuit()

        # Reset check qubits to |+⟩ state
        reset_qubits = stim_ids.z_check_ids + stim_ids.x_check_ids
        if reset_qubits:
            circuit.append("RX", reset_qubits)

        # Append the Z and X subcircuits
        circuit += z_subcircuit
        circuit += x_subcircuit

        # Apply Hadamard gates to return check qubits to Z basis
        if reset_qubits:
            circuit.append("H", reset_qubits)

        # Measure check qubits in Z basis
        if reset_qubits:
            circuit.append("M", reset_qubits)

        measurements = [stim_ids.z_check_ids + stim_ids.x_check_ids]
        return circuit, measurements
