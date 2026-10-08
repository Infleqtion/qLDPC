# SPDX-License-Identifier: Apache-2.0

"""Decomposition of circuits into moments for noise application.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import collections
from collections.abc import Collection, Iterable, Iterator

import stim

from .operations import (
    ANNOTATION,
    CLIFFORD_2Q,
    CLIFFORD_PP,
    COLLAPSING_OPS,
    IDLE_OPS,
    JUST_MEASURE_2Q,
    JUST_MEASURE_PP,
    NOISE,
    _involves_classical_bits,
    op_type,
)


def _idle_barrier_qubits(
    instructions: Iterable[stim.CircuitInstruction | stim.CircuitRepeatBlock],
) -> set[int]:
    """Qubits that pending idle errors may not be deferred past, at any depth of repeat blocks.

    These are the qubits targeted by any instruction other than an explicit idle (``I``/``II``) or
    a ``QUBIT_COORDS`` annotation.  Annotations such as ``OBSERVABLE_INCLUDE`` with Pauli targets
    are included, since they depend on the state of their target qubits.
    """
    qubits: set[int] = set()
    for instruction in instructions:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            qubits.update(_idle_barrier_qubits(instruction.body_copy()))
        elif instruction.name not in IDLE_OPS and instruction.name != "QUBIT_COORDS":
            qubits.update(
                target.qubit_value
                for target in instruction.targets_copy()
                if target.qubit_value is not None
            )
    return qubits


def _get_immune_qubits(circuit: stim.Circuit, immune_qubit_tag: str) -> frozenset[int]:
    """Identify qubits declared immune to noise by tagged coordinate annotations."""
    immune_qubits: set[int] = set()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            immune_qubits.update(_get_immune_qubits(instruction.body_copy(), immune_qubit_tag))
        elif instruction.name == "QUBIT_COORDS" and immune_qubit_tag in instruction.tag:
            immune_qubits.update(
                target.qubit_value
                for target in instruction.targets_copy()
                if target.qubit_value is not None
            )
    return frozenset(immune_qubits)


def _categorize_moment_qubits(
    moment: Collection[stim.CircuitInstruction],
) -> tuple[list[int], list[int]]:
    """Categorize a moment's qubit targets and check for reuse.

    Iterates every non-annotation instruction and buckets its qubit targets into four lists —
    collapsed (measurement / reset), classically-controlled, explicit-idle, and everything else
    ("operation").  Raises if any qubit is used more than once across the four buckets, since noise
    application relies on the "each qubit at most once per moment" invariant that
    ``_split_moments_with_ticks`` enforces when ``insert_ticks=True``.  The classically-controlled
    and explicit-idle buckets contribute only to the reuse check.

    Args:
        moment: The moment's operations.

    Returns:
        A tuple ``(collapsed_qubits, operation_qubits)``, each in the order the moment referenced
        them.  Consumers use these for idle-error emission (see
        ``NoiseModel._inplace_append_idle_errors``).

    Raises:
        ValueError: If any qubit is operated on multiple times within the moment without a TICK in
            between.
    """
    collapsed_qubits: list[int] = []
    operation_qubits: list[int] = []
    classically_controlled_qubits: list[int] = []
    explicit_idle_qubits: list[int] = []
    for op in moment:
        if op_type(op.name) in (ANNOTATION, NOISE):
            continue
        target_qubits = [
            target.qubit_value for target in op.targets_copy() if target.qubit_value is not None
        ]
        if op_type(op.name) in (CLIFFORD_PP, JUST_MEASURE_PP):
            # A Pauli product may intentionally address one qubit more than once, e.g.
            # ``MPP X0*Z0``.  It is one atomic operation, not two operations in the moment.
            target_qubits = list(dict.fromkeys(target_qubits))
        if op.name in COLLAPSING_OPS:
            qubits = collapsed_qubits
        elif _involves_classical_bits(op):
            qubits = classically_controlled_qubits
        elif op.name in IDLE_OPS:
            qubits = explicit_idle_qubits
        else:
            qubits = operation_qubits
        qubits.extend(target_qubits)

    usage_counts = collections.Counter(
        collapsed_qubits + operation_qubits + classically_controlled_qubits + explicit_idle_qubits
    )
    qubits_used_multiple_times = {qubit for qubit, count in usage_counts.items() if count != 1}
    if qubits_used_multiple_times:
        raise ValueError(
            f"Qubits were operated on multiple times without a TICK in between:\n"
            f"multiple uses: {sorted(qubits_used_multiple_times)}\n"
            f"moment:\n{moment}"
        )
    return collapsed_qubits, operation_qubits


def _split_moments_with_ticks(circuit: stim.Circuit, immune_op_tag: str | None) -> stim.Circuit:
    """Insert TICKs into a circuit to split stim.CircuitInstruction that reuse qubits.

    This preprocessing ensures that errors are applied correctly to a stim.CircuitInstruction that
    reuses qubits.

    Args:
        circuit: The input circuit to preprocess.
        immune_op_tag: Don't split operations with this tag.

    Returns:
        stim.Circuit: A circuit with TICKs added to prevent instructions from reusing qubits.
    """
    result = stim.Circuit()
    used_qubits: set[int] = set()

    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if immune_op_tag is not None and immune_op_tag in op.tag:
                result.append(op)
                continue

            # Process repeat blocks recursively
            if used_qubits:
                result.append("TICK")
                used_qubits.clear()
            processed_body = _split_moments_with_ticks(op.body_copy(), immune_op_tag)
            result.append(
                stim.CircuitRepeatBlock(
                    repeat_count=op.repeat_count, body=processed_body, tag=op.tag
                )
            )
            continue

        if op.name == "TICK":
            # Explicit TICK - clear used qubits
            result.append("TICK")
            used_qubits.clear()
            continue

        """
        For preprocessing, we need to force splitting of multi-target operations to detect qubit
        reuse properly.  Use a dummy immune_qubits set with -1 to force splitting of 2-qubit
        operations.
        """
        split_ops = list(_split_targets_if_needed(op, frozenset({-1}), immune_op_tag))

        for split_op in split_ops:
            # Check if this split operation would reuse any qubits
            op_qubits = set()
            if op_type(split_op.name) not in (ANNOTATION, NOISE):
                for target in split_op.targets_copy():
                    if not target.is_combiner and target.qubit_value is not None:
                        op_qubits.add(target.qubit_value)

            # If there's qubit reuse, insert a TICK first
            if op_qubits & used_qubits:
                result.append("TICK")
                used_qubits.clear()

            # Add the operation and update used qubits
            result.append(split_op)
            used_qubits.update(op_qubits)

    return result


def _iter_moments_and_repeat_blocks(
    circuit: stim.Circuit,
    immune_qubits: frozenset[int],
    immune_op_tag: str | None,
    *,
    force_split: bool = False,
) -> Iterator[stim.CircuitRepeatBlock | list[stim.CircuitInstruction]]:
    """Splits a circuit into moments and some operations into pieces.

    Classical control system operations like ``CX rec[-1] 0`` are split from quantum operations like
    ``CX 1 0``.  SPP and MPP operations are split into one operation per Pauli product.

    Args:
        circuit: The circuit to split into moments.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.
        force_split: If True, decompose every broadcast gate into its individual applications (one
            target for one-qubit gates, one pair for two-qubit gates) even when no qubit is immune.
            Used when a ``rule_func`` is present so the callback is consulted once per gate
            application.

    Yields:
        Lists of operations corresponding to one moment in the circuit, with any problematic
        operations like MPPs split into pieces, or CircuitRepeatBlock instances for repeat blocks.

    Note:
        A moment is the time between two TICKs.
    """
    current_moment: list[stim.CircuitInstruction] = []

    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if current_moment:
                yield current_moment
                current_moment = []
            yield op
        elif op.name == "TICK":
            if current_moment:
                yield current_moment
                current_moment = []
        else:
            current_moment.extend(
                _split_targets_if_needed(op, immune_qubits, immune_op_tag, force_split=force_split)
            )
    if current_moment:
        yield current_moment


def _split_targets_if_needed(
    op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    immune_op_tag: str | None,
    *,
    force_split: bool = False,
) -> Iterator[stim.CircuitInstruction]:
    """Splits operations into pieces as needed.

    This function splits operations like SPP and MPP into each Pauli product, and separates
    classical control operations from quantum operations.

    Args:
        op: The circuit instruction to potentially split.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.
        force_split: If True, always decompose broadcast one- and two-qubit gates into individual
            applications, regardless of immunity.

    Yields:
        Circuit instructions, potentially split into smaller pieces.
    """
    this_op_type = op_type(op.name)
    # Two-qubit ops (both Clifford CX/CZ/... and joint measurements MXX/MYY/MZZ) split per-pair via
    # _split_targets_clifford_2q so partial-immunity decisions happen at the pair level.
    if this_op_type in (CLIFFORD_2Q, JUST_MEASURE_2Q):
        yield from _split_targets_clifford_2q(
            op, immune_qubits, immune_op_tag, force_split=force_split
        )
    elif this_op_type == CLIFFORD_PP or this_op_type == JUST_MEASURE_PP:
        yield from _split_targets_pp(op)
    elif this_op_type in (NOISE, ANNOTATION):
        yield op
    else:
        yield from _split_targets_clifford_1q(
            op, immune_qubits, immune_op_tag, force_split=force_split
        )


def _split_targets_clifford_1q(
    op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    immune_op_tag: str | None,
    *,
    force_split: bool = False,
) -> Iterator[stim.CircuitInstruction]:
    """Splits single-qubit Clifford operations when immune qubits are present.

    Args:
        op: The single-qubit Clifford operation to split.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.
        force_split: If True, split into individual single-target operations unconditionally.

    Yields:
        Circuit instructions split into individual single-target operations.
    """
    if force_split or immune_qubits or (immune_op_tag is not None and immune_op_tag in op.tag):
        args = op.gate_args_copy()
        for target in op.targets_copy():
            yield stim.CircuitInstruction(op.name, [target], args, tag=op.tag)
    else:
        yield op


def _split_targets_clifford_2q(
    op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    immune_op_tag: str | None,
    *,
    force_split: bool = False,
) -> Iterator[stim.CircuitInstruction]:
    """Splits two-qubit operations into individual gate pairs.

    Handles both two-qubit Cliffords (CX/CZ/...) and joint two-qubit measurements (MXX/MYY/MZZ),
    separating classical-control from quantum-only gates.

    Args:
        op: The two-qubit operation to split.
        immune_qubits: Qubits that are declared to be immune to noise.
        immune_op_tag: Don't split operations with this tag.
        force_split: If True, split into individual two-qubit gate pairs unconditionally.

    Yields:
        Circuit instructions split into individual two-qubit gate operations.
    """
    assert op_type(op.name) in (CLIFFORD_2Q, JUST_MEASURE_2Q)
    targets = op.targets_copy()
    if (
        force_split
        or immune_qubits
        or (immune_op_tag is not None and immune_op_tag in op.tag)
        or any(target.is_measurement_record_target for target in targets)
    ):
        args = op.gate_args_copy()
        for k in range(0, len(targets), 2):
            yield stim.CircuitInstruction(op.name, targets[k : k + 2], args, tag=op.tag)
    else:
        yield op


def _split_targets_pp(op: stim.CircuitInstruction) -> Iterator[stim.CircuitInstruction]:
    """Splits a Pauli product operation into one operation for each Pauli product.

    Args:
        op: The Pauli product operation to split.

    Yields:
        Circuit instructions, one for each Pauli product.
    """
    assert op_type(op.name) in (CLIFFORD_PP, JUST_MEASURE_PP)
    targets = op.targets_copy()
    args = op.gate_args_copy()
    # group combiner-linked targets so each instruction contains one Pauli product
    start = end = 0
    while end < len(targets):
        if end + 1 == len(targets) or not targets[end + 1].is_combiner:
            yield stim.CircuitInstruction(op.name, targets[start : end + 1], args, tag=op.tag)
            end += 1
            start = end
        else:
            end += 2
    assert end == len(targets)
