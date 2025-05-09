import stim

from qldpc.codes.common import CSSCode
from qldpc.stim.noise_model import NoiseModel
from qldpc.stim.util import Basis


def _update_meas_rec(meas_record: list[dict], qubits: list[int]):
    """
    Measurement results in stim are recorded as a stack and can be tricky to reference.
    The meas_record tracks the indicies in the stack for each 'round' of measurements (call to this function).
    This function is called after every call to NoiseModel.apply_meas with the same list of qubits
    """
    meas_round = {}
    for i in range(len(qubits)):
        q = qubits[-(i + 1)]
        meas_round[q] = -(i + 1)
    for round in meas_record:
        for q, idx in round.items():
            round[q] = idx - len(qubits)
    meas_record.append(meas_round)


def _get_meas_rec(meas_record: list[dict], round_idx: int, qubit: int):
    """
    Get a stim reference to a measurement result. Typically used to compare parity checks/stabilizers between rounds in case there is a change (error).
    """
    return stim.target_rec(meas_record[round_idx][qubit])


def memory_experiment(
    code: CSSCode, noise_model: NoiseModel, num_rounds: int, basis: Basis
) -> stim.Circuit:
    """
    Creates a stim circuit for a memory experiment in the specified basis
    TODO: documentation
    """
