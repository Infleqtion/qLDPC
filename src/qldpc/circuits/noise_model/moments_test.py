# SPDX-License-Identifier: Apache-2.0

"""Unit tests for moments.py."""

import functools
from collections.abc import Iterable, Iterator

import pytest
import stim

from qldpc import circuits
from qldpc.circuits.conftest import circuits_are_equivalent, flatten_repeat_blocks
from qldpc.circuits.noise_model.moments import (
    ScheduleItem,
    _categorize_moment_qubits,
    _close_gate_free_moment,
    _get_immune_qubits,
    _idle_barrier_qubits,
    _MomentScheduler,
    _parse_circuit,
    _ParsedBlock,
    _RepeatedMoments,
    _schedule_barrier_qubits,
    _schedule_moments,
)


def test_classical_controls() -> None:
    """Classically controled gates get special treatment."""
    noise_model: circuits.NoiseModel

    # classically controls are immune to noise, but the qubits still pick up idling errors
    circuit = stim.Circuit("""
        CX 0 1 rec[-1] 2
    """)
    noise_model = circuits.SI1000NoiseModel(0.1)
    noisy_circuit = stim.Circuit("""
        CX 0 1 rec[-1] 2
        DEPOLARIZE2(0.1) 0 1
        DEPOLARIZE1(0.01) 2
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # qubits addressed by classical controls pick up idling errors by default
    circuit = stim.Circuit("""
        H 0
        CX rec[-1] 1
        TICK
        H 0 1 2
    """)
    noise_model = circuits.NoiseModel(idle_error=0.1)
    noisy_circuit = stim.Circuit("""
        H 0
        CX rec[-1] 1
        DEPOLARIZE1(0.1) 1 2
        TICK
        H 0 1 2
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_pauli_product_measurements() -> None:
    """Pauli product measurements get special treatment."""

    circuit = stim.Circuit("""
        MPP X1*Y2*Z3
    """)
    noise_model = circuits.NoiseModel(readout_error=0.1, idle_error=0.2)
    noisy_circuit = stim.Circuit("""
        MPP(0.1) X1*Y2*Z3
        DEPOLARIZE1(0.2) 0
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))

    # override the default MPP rule for specific Pauli products
    circuit = stim.Circuit("""
        MPP Z0*Z1*Z2
        MPP X0*Y1*Z2
    """)
    noise_rule = circuits.NoiseRule(readout_error=0.2)
    noise_model = circuits.NoiseModel(readout_error=0.1, rules={"MXYZ": noise_rule})
    noisy_circuit = stim.Circuit("""
        MPP(0.1) Z0*Z1*Z2
        MPP(0.2) X0*Y1*Z2
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))


def test_repeat_blocks() -> None:
    """Repeat blocks get special treatment."""

    # moments are identified as though repeat blocks were flattened
    circuit = stim.Circuit("""
        H 0
        REPEAT 3 {
            CX 0 1
        }
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    noisy_circuit = stim.Circuit("""
        H 0
        DEPOLARIZE1(0.1) 0
        TICK
        REPEAT 3 {
            CX 0 1
            DEPOLARIZE2(0.1) 0 1
            TICK
        }
    """)
    assert circuits_are_equivalent(noisy_circuit, noise_model.noisy_circuit(circuit))
    with pytest.raises(ValueError, match="multiple uses"):
        noise_model.noisy_circuit(circuit, insert_ticks=False)

    immune_op_tag = "_TEST_"
    circuit = stim.Circuit(f"""
        H 0
        REPEAT['{immune_op_tag}'] 3 {{
            CX 0 1 1 2
        }}
    """)
    noise_model = circuits.DepolarizingNoiseModel(0.1, include_idling_error=False)
    noisy_circuit = stim.Circuit(f"""
        H 0
        DEPOLARIZE1(0.1) 0
        REPEAT['{immune_op_tag}'] 3 {{
            CX 0 1 1 2
        }}
    """)
    assert circuits_are_equivalent(
        noisy_circuit,
        noise_model.noisy_circuit(circuit, immune_op_tag=immune_op_tag),
    )


def test_moment_qubits() -> None:
    """Qubits are categorized by their role in a moment, and qubit reuse is rejected."""
    moment = list(stim.Circuit("MPP X0*Z0\nR 1\nH 2\nI 3\nCX rec[-1] 4\nDETECTOR rec[-1]"))
    assert _categorize_moment_qubits(moment) == ([0, 1], [2])
    with pytest.raises(ValueError, match="multiple uses"):
        _categorize_moment_qubits(list(stim.Circuit("I 0\nH 0")))

    circuit = stim.Circuit("""
        QUBIT_COORDS[immune](0) 0
        REPEAT 2 {
            QUBIT_COORDS[immune](1) 1
            I 2
            OBSERVABLE_INCLUDE(0) X3
            H 4
        }
    """)
    assert _get_immune_qubits(circuit, "immune") == {0, 1}
    assert _idle_barrier_qubits(circuit) == {3, 4}


def _unroll(schedule: Iterable[ScheduleItem]) -> Iterator[stim.Circuit | stim.CircuitRepeatBlock]:
    """Unroll the repeated moments of a moment schedule."""
    for item in schedule:
        if isinstance(item, _RepeatedMoments):
            for _ in range(item.repeat_count):
                yield from _unroll(item.body)
        else:
            yield item


def _expand(schedule: list[ScheduleItem]) -> list[stim.Circuit | stim.CircuitRepeatBlock]:
    """Expand a moment schedule, merging moments without gates into the next moment."""
    expanded: list[stim.Circuit | stim.CircuitRepeatBlock] = []
    gate_free = stim.Circuit()
    for item in _unroll(schedule):
        if isinstance(item, stim.CircuitRepeatBlock):
            expanded.append(item)
        elif _close_gate_free_moment(item)[0]:
            gate_free += flatten_repeat_blocks(item, circuits.DEFAULT_IMMUNE_OP_TAG)
        else:
            expanded.append(gate_free + flatten_repeat_blocks(item, circuits.DEFAULT_IMMUNE_OP_TAG))
            gate_free = stim.Circuit()
    if gate_free:
        expanded.append(gate_free)
    return expanded


SCHEDULE_TEST_CIRCUITS = {
    "ticked": "R 0 1\nTICK\nREPEAT 3 {\n    H 0\n    TICK\n    X 1\n    TICK\n}\nM 0 1",
    "unticked seams": "R 0 1 2\nREPEAT 3 {\n    H 0\n    TICK\n    X 1\n}\nH 2",
    "leading tick": "R 0 1\nTICK\nREPEAT 4 {\n    TICK\n    H 0\n    TICK\n    X 1\n}\nTICK\nM 0",
    "gate seams": "H 1\nREPEAT 5 {\n    CX 0 1 1 2\n    H 3\n}\nX 2",
    "cycle of iterations": "REPEAT 10 {\n    H 0\n    CX 1 2\n    H 1\n}\nH 2",
    "nested": """
        R 0 1 2
        REPEAT 3 {
            H 0
            REPEAT 4 {
                X 1
                TICK
                H 2
            }
            DETECTOR rec[-1]
        }
        M 0 1 2
    """,
    "annotations": """
        M 0
        REPEAT 3 {
            DETECTOR rec[-1]
            SHIFT_COORDS(1)
        }
        H 0
        REPEAT 2 {
            X_ERROR(0.1) 1
        }
        TICK
        REPEAT 2 {
            REPEAT 1 {
                H 1
            }
        }
        TICK
        SHIFT_COORDS(1)
    """,
    "nested large": """
        REPEAT 5 {
            TICK
            REPEAT 6 {
                H 0
                TICK
            }
            X 1
            OBSERVABLE_INCLUDE(0) Z2
        }
    """,
    "immune": """
        H 0
        REPEAT[__IMMUNE_TO_NOISE__] 3 {
            H 0
            H 0
        }
        H 0
    """,
    "surface code": str(
        stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=5)
    ),
}


@pytest.mark.parametrize("circuit_name", sorted(SCHEDULE_TEST_CIRCUITS))
@pytest.mark.parametrize("insert_ticks", [True, False])
def test_schedule_moments(circuit_name: str, insert_ticks: bool) -> None:
    """Moments are identified as though non-immune repeat blocks were flattened."""
    circuit = stim.Circuit(SCHEDULE_TEST_CIRCUITS[circuit_name])
    flattened = flatten_repeat_blocks(circuit, circuits.DEFAULT_IMMUNE_OP_TAG)
    schedule_moments = functools.partial(
        _schedule_moments, insert_ticks=insert_ticks, immune_op_tag=circuits.DEFAULT_IMMUNE_OP_TAG
    )
    expected = schedule_moments(flattened)
    assert not any(isinstance(item, _RepeatedMoments) for item in expected)
    if not _is_valid(expected):
        with pytest.raises(ValueError, match="multiple uses"):
            _is_valid(schedule_moments(circuit), raise_error=True)
        return
    schedule = schedule_moments(circuit)
    assert _expand(schedule) == _expand(expected)


def _is_valid(schedule: list[ScheduleItem], *, raise_error: bool = False) -> bool:
    """Are qubits used at most once per moment of a moment schedule?"""
    try:
        for item in _expand(schedule):
            if isinstance(item, stim.Circuit):
                _categorize_moment_qubits(list(item))
    except ValueError:
        if raise_error:
            raise
        return False
    return True


def test_schedule_moments_size() -> None:
    """Iterations of a repeat block are unrolled only when necessary."""
    schedule_moments = functools.partial(
        _schedule_moments, insert_ticks=True, immune_op_tag=circuits.DEFAULT_IMMUNE_OP_TAG
    )

    # a single gate per iteration
    assert schedule_moments(stim.Circuit("REPEAT 1000 {\n    H 0\n}")) == [
        _RepeatedMoments(1000, (stim.Circuit("H 0"),))
    ]

    # moments without gates do not prevent the identification of repeated moments
    circuit = stim.Circuit("""
        M 0
        TICK
        DETECTOR rec[-1]
        REPEAT 10 {
            H 0
            TICK
            M 0
            TICK
            DETECTOR rec[-1] rec[-2]
        }
        M 0
    """)
    schedule = schedule_moments(circuit)
    assert [type(item) for item in schedule] == [
        stim.Circuit,
        stim.Circuit,
        _RepeatedMoments,
        stim.Circuit,
    ]

    # the first iteration is unrolled when its first moment differs from that of later iterations
    circuit = stim.Circuit("""
        H 0
        REPEAT 10 {
            TICK
            M 0
        }
        X 1
    """)
    assert schedule_moments(circuit) == [
        stim.Circuit("H 0"),
        _RepeatedMoments(9, (stim.Circuit("M 0"),)),
        stim.Circuit("M 0\nX 1"),
    ]


def test_schedule_barrier_qubits() -> None:
    """Idle errors are not deferred past operations on their qubits in any scheduled item."""
    circuit = stim.Circuit("""
        H 0
        TICK
        REPEAT 5 {
            I 1
            TICK
            REPEAT 6 {
                X 2
                TICK
            }
        }
    """)
    schedule = _schedule_moments(circuit, insert_ticks=True, immune_op_tag=None)
    assert isinstance(schedule[1], _RepeatedMoments)
    assert _schedule_barrier_qubits(schedule[:1]) == {0}
    assert _schedule_barrier_qubits(schedule[1:]) == {2}


def test_schedule_cycle_of_iterations() -> None:
    """Repeated moments may consist of several iterations of a repeat block."""

    class AlternatingScheduler(_MomentScheduler):
        """Each iteration closes its open moment and leaves the other one open."""

        def _schedule(
            self,
            body: tuple[stim.CircuitInstruction | _ParsedBlock, ...],
            open_moment: stim.Circuit,
        ) -> tuple[list[ScheduleItem], stim.Circuit]:
            other = (
                stim.Circuit("H 1") if open_moment == stim.Circuit("H 0") else stim.Circuit("H 0")
            )
            return [open_moment], other

    block = _parse_circuit(stim.Circuit("REPEAT 8 {\n    H 2\n    TICK\n}"))[0]
    assert isinstance(block, _ParsedBlock)
    scheduler = AlternatingScheduler(insert_ticks=True, immune_op_tag=None)
    h_0, h_1 = stim.Circuit("H 0"), stim.Circuit("H 1")
    assert scheduler._schedule_repeat(block, stim.Circuit("H 2")) == (
        [stim.Circuit("H 2"), _RepeatedMoments(3, (h_0, h_1)), h_0],
        h_1,
    )


def test_schedule_moments_without_boundaries() -> None:
    """Repeat blocks without moment boundaries stay compact or are rejected."""
    circuit = stim.Circuit("H 0\nREPEAT 1000 {\n    X_ERROR(0.1) 0\n    DETECTOR\n}\nH 1")
    expected = stim.Circuit("H 0")
    expected.append(circuit[1])
    expected.append("H", [1])
    for insert_ticks in [True, False]:
        assert _schedule_moments(circuit, insert_ticks=insert_ticks, immune_op_tag=None) == [
            expected
        ]
    with pytest.raises(ValueError, match="multiple uses"):
        _schedule_moments(
            stim.Circuit("REPEAT 2 {\n    H 0\n}"), insert_ticks=False, immune_op_tag=None
        )
