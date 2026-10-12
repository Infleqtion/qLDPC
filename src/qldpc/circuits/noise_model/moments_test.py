# SPDX-License-Identifier: Apache-2.0

"""Unit tests for moments.py."""

import functools

import pytest
import stim

from qldpc import circuits
from qldpc.circuits.conftest import circuits_are_equivalent
from qldpc.circuits.noise_model.moments import (
    ScheduleItem,
    _categorize_moment_qubits,
    _GateFreeOps,
    _get_immune_qubits,
    _idle_barrier_qubits,
    _MomentScheduler,
    _needs_initial_tick,
    _parse_circuit,
    _ParsedBlock,
    _RepeatedMoments,
    _schedule_barrier_qubits,
    _schedule_moments,
    _Tick,
    _TickState,
    _without_trailing_block_tick,
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


def test_schedule_moments_size() -> None:
    """Iterations of a repeat block are unrolled only when necessary."""
    schedule_moments = functools.partial(
        _schedule_moments, insert_ticks=True, immune_op_tag=circuits.DEFAULT_IMMUNE_OP_TAG
    )

    # a single gate per iteration, where the last moment is left open for subsequent operations
    assert schedule_moments(stim.Circuit("REPEAT 1000 {\n    H 0\n}")) == [
        _RepeatedMoments(999, (stim.Circuit("H 0"),)),
        stim.Circuit("H 0"),
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
        _GateFreeOps,
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

    # all iterations are unrolled if they end before their moments repeat
    circuit = stim.Circuit("H 0\nREPEAT 2 {\n    TICK\n    M 0\n}")
    assert schedule_moments(circuit) == [
        stim.Circuit("H 0"),
        stim.Circuit("M 0"),
        stim.Circuit("M 0"),
    ]

    # an inner repeat block that is reached twice with the same open moment is scheduled once
    circuit = stim.Circuit("""
        H 1
        REPEAT 3 {
            REPEAT 2 {
                H 0
                TICK
            }
            X 1
        }
    """)
    assert schedule_moments(circuit) == [
        stim.Circuit("H 1\nH 0"),
        stim.Circuit("H 0"),
        _RepeatedMoments(2, (stim.Circuit("X 1\nH 0"), stim.Circuit("H 0"))),
        stim.Circuit("X 1"),
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

    # operations without gates may still depend on the state of qubits
    gate_free_ops = _GateFreeOps(stim.Circuit("DETECTOR(0)\nOBSERVABLE_INCLUDE(0) Z1"))
    assert _schedule_barrier_qubits([gate_free_ops]) == {1}


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
    """Repeat blocks without gates are kept as-is, and those without TICKs may be rejected."""
    for repeat_count in [1, 1000]:
        block = stim.CircuitRepeatBlock(repeat_count, stim.Circuit("X_ERROR(0.1) 0\nDETECTOR"))
        circuit = stim.Circuit("H 0")
        circuit.append(block)
        circuit.append("H", [1])
        for insert_ticks in [True, False]:
            schedule = _schedule_moments(circuit, insert_ticks=insert_ticks, immune_op_tag=None)
            assert schedule == [circuit]

    circuit = stim.Circuit("H 1\nREPEAT 1000 {\n    H 0\n}")
    assert _schedule_moments(circuit, insert_ticks=True, immune_op_tag=None) == [
        stim.Circuit("H 1\nH 0"),
        _RepeatedMoments(998, (stim.Circuit("H 0"),)),
        stim.Circuit("H 0"),
    ]
    with pytest.raises(ValueError, match="multiple uses"):
        _schedule_moments(circuit, insert_ticks=False, immune_op_tag=None)

    # repeat blocks that are empty when flattened are dropped, even if tagged as immune to noise
    circuit = stim.Circuit("""
        H 0
        REPEAT 2 {
            REPEAT[__IMMUNE_TO_NOISE__] 3 {
            }
            REPEAT 1 {
                REPEAT 4 {
                }
            }
        }
        H 1
    """)
    assert _schedule_moments(
        circuit, insert_ticks=True, immune_op_tag=circuits.DEFAULT_IMMUNE_OP_TAG
    ) == [stim.Circuit("H 0\nH 1")]


def test_schedule_ticks() -> None:
    """TICKs that end empty moments are kept only to the extent that they matter."""
    schedule_moments = functools.partial(_schedule_moments, insert_ticks=True, immune_op_tag=None)
    assert schedule_moments(stim.Circuit("H 0\nTICK\nTICK\nH 0")) == [
        stim.Circuit("H 0"),
        _Tick(),
        stim.Circuit("H 0"),
    ]
    # a repeat block of TICKs is equivalent to a single TICK
    circuit = stim.Circuit("H 0\nTICK\nDETECTOR\nREPEAT 5 {\n    TICK\n}\nH 0")
    assert schedule_moments(circuit) == [
        stim.Circuit("H 0"),
        _GateFreeOps(stim.Circuit("DETECTOR")),
        _Tick(),
        stim.Circuit("H 0"),
    ]

    # a TICK ends preceding gate-free operations, but is itself not preceded by a TICK
    h_0 = stim.Circuit("H 0")
    assert not _needs_initial_tick([_Tick()], _TickState.GATE_FREE_OPS)
    assert _needs_initial_tick([_Tick(), h_0], _TickState.GATE_FREE_OPS)
    assert not _needs_initial_tick([h_0], _TickState.GATE_FREE_OPS)
    assert _needs_initial_tick([_RepeatedMoments(2, (h_0,))], _TickState.MOMENT)


def test_without_trailing_block_tick() -> None:
    """The TICK at the end of the last repeat block is removed, unrolling its last iteration."""
    for repeat_count, expected in [
        (3, "REPEAT[tag] 2 {\n    H 0\n    TICK\n}\nH 0\nX_ERROR(0.1) 0"),
        (2, "H 0\nTICK\nH 0\nX_ERROR(0.1) 0"),
        (1, "H 0\nX_ERROR(0.1) 0"),
    ]:
        circuit = stim.Circuit(f"REPEAT[tag] {repeat_count} {{\n    H 0\n    TICK\n}}")
        circuit.append("X_ERROR", [0], 0.1)
        assert _without_trailing_block_tick(circuit) == stim.Circuit(expected)

    # nested repeat blocks
    circuit = stim.Circuit("""
        REPEAT 2 {
            X 1
            TICK
            REPEAT 3 {
                H 0
                TICK
            }
        }
        X_ERROR(0.1) 0
    """)
    trimmed = _without_trailing_block_tick(circuit)
    assert trimmed.num_ticks == circuit.num_ticks - 1
    flattened = circuit.flattened()
    assert trimmed.flattened() == flattened[:-2] + flattened[-1:]


def test_schedule_preserves_tags() -> None:
    """Repeated moments keep the tag of their repeat block."""
    circuit = stim.Circuit("""
        H 0
        TICK
        REPEAT[my_tag] 3 {
            H 0
            TICK
        }
        H 1
    """)
    assert _schedule_moments(circuit, insert_ticks=True, immune_op_tag=None) == [
        stim.Circuit("H 0"),
        _RepeatedMoments(3, (stim.Circuit("H 0"),), "my_tag"),
        stim.Circuit("H 1"),
    ]
