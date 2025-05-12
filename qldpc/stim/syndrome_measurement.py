from __future__ import annotations

import abc
from dataclasses import dataclass

import networkx as nx
import numpy as np
import stim

from qldpc.codes.common import ClassicalCode, CSSCode
from qldpc.objects import Pauli
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
        self, code: CSSCode, noise_model: NoiseModel, stim_ids: StimIds
    ) -> tuple[stim.Circuit, list[list[int]]]:
        """
        Compiles a syndrome measurement circuit for a given CSSCode and noise model.

        args:
            code: CSSCode
                The quantum code to be compiled into a single round of syndrome measurements
            noise_model: NoiseModel
                The noise model to wrap stim operations (1q gate, 2q gate, meas, and reset)
            stim_ids: StimIds
                Stim circuit ids to be used for data qubits, z check qubits, and x check qubits
        returns:
            stim.Circuit:
                Stim circuit containg the compiled syndrome measurement round
            list[list[int]]:
                The history of measurement rounds performed in the circuit.
                Each round is a list of the stim ids measured that round, in the order they were passed to stim.
        """


class BareColorCircuit(SyndromeMeasurement):
    """
    A coloration circuit syndrome measurement scheme as defined in https://arxiv.org/abs/2109.14609 (Algorithm 1).

    NOTE: This is not guaranteed to be distance-preserving.
    """

    def _code_to_edge_coloring(self, code: ClassicalCode) -> dict[tuple[int, int], int]:
        graph = nx.Graph()
        edges: list[tuple[int, int]] = list(
            zip(*np.where(code.matrix))
        )  # Treat edges in tanner graph as nodes in coloring graph

        for edge1 in edges:
            # Connect to all other tanner graph edges that share a check or data qubit
            for edge2 in edges:
                if edge1 != edge2 and (edge1[0] == edge2[0] or edge1[1] == edge2[1]):
                    graph.add_edge(edge1, edge2)
        return nx.coloring.greedy_color(graph, strategy="largest_first")

    def _code_to_subcircuit(
        self,
        code: ClassicalCode,
        noise_model: NoiseModel,
        check_ids: list[int],
        data_ids: list[int],
        gate: str,
    ) -> stim.Circuit:
        coloring = self._code_to_edge_coloring(code)
        circuit = stim.Circuit()

        schedule: dict[int, list[tuple[int, int]]] = {}
        for edge, color in coloring.items():
            check_op = (check_ids[edge[0]], data_ids[edge[1]])
            schedule.setdefault(color, []).append(check_op)
        for color, moment in schedule.items():
            noise_model.apply_2q_gates(circuit, gate, moment)
        return circuit

    def compile_sm_circuit(
        self, code: CSSCode, noise_model: NoiseModel, stim_ids: StimIds
    ) -> tuple[stim.Circuit, list[list[int]]]:
        """
        Compiles a coloration circuit. Not depth-optimal as no interleaving of opposite type checks is present. Z checks are performed first followed by X checks
        """
        z_subcircuit = self._code_to_subcircuit(
            code.code_z,
            noise_model,
            stim_ids.z_check_ids,
            stim_ids.data_ids,
            "CZ",
        )
        x_subcircuit = self._code_to_subcircuit(
            code.code_x,
            noise_model,
            stim_ids.x_check_ids,
            stim_ids.data_ids,
            "CX",
        )
        circuit = stim.Circuit()
        noise_model.apply_reset(circuit, Pauli.Z, stim_ids.z_check_ids + stim_ids.x_check_ids)
        noise_model.apply_1q_gates(circuit, "H", stim_ids.z_check_ids + stim_ids.x_check_ids)
        circuit.append(z_subcircuit)
        circuit.append(x_subcircuit)
        noise_model.apply_1q_gates(circuit, "H", stim_ids.z_check_ids + stim_ids.x_check_ids)
        noise_model.apply_meas(circuit, Pauli.Z, stim_ids.z_check_ids + stim_ids.x_check_ids)
        measurements = [stim_ids.z_check_ids + stim_ids.x_check_ids]
        return circuit, measurements
