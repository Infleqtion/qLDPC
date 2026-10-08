# SPDX-License-Identifier: Apache-2.0

"""Decomposition of circuits into moments for noise application.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import collections
import dataclasses
from collections.abc import Collection, Iterable, Iterator
from typing import TypeAlias

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
    moment: Collection[stim.CircuitInstruction | stim.CircuitRepeatBlock],
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
        if isinstance(op, stim.CircuitRepeatBlock) or op_type(op.name) in (ANNOTATION, NOISE):
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


@dataclasses.dataclass(frozen=True)
class _RepeatedMoments:
    """A repeat block whose body is a sequence of whole moments.

    Moments do not straddle the boundaries of the body, so repeating the body is equivalent to
    flattening it.
    """

    repeat_count: int
    body: tuple[ScheduleItem, ...]
    tag: str = ""


# An item in a moment schedule: one moment (a stim.Circuit without TICKs, which may contain repeat
# blocks of annotations and noise), repeated moments, or a repeat block that is immune to noise.
ScheduleItem: TypeAlias = stim.Circuit | _RepeatedMoments | stim.CircuitRepeatBlock


@dataclasses.dataclass(eq=False)
class _ParsedBlock:
    """A repeat block of an input circuit, with its body parsed once for reuse."""

    block: stim.CircuitRepeatBlock
    body: tuple[stim.CircuitInstruction | _ParsedBlock, ...]


def _parse_circuit(circuit: stim.Circuit) -> tuple[stim.CircuitInstruction | _ParsedBlock, ...]:
    """Convert a circuit into a tuple of instructions and parsed repeat blocks."""
    return tuple(
        _ParsedBlock(instruction, _parse_circuit(instruction.body_copy()))
        if isinstance(instruction, stim.CircuitRepeatBlock)
        else instruction
        for instruction in circuit
    )


def _gate_qubits(op: stim.CircuitInstruction | stim.CircuitRepeatBlock) -> set[int]:
    """The qubits that an operation acts on, for the purpose of identifying qubit reuse."""
    if isinstance(op, stim.CircuitRepeatBlock) or op_type(op.name) in (ANNOTATION, NOISE):
        return set()
    return {
        target.qubit_value
        for target in op.targets_copy()
        if not target.is_combiner and target.qubit_value is not None
    }


def _schedule_moments(
    circuit: stim.Circuit, *, insert_ticks: bool, immune_op_tag: str | None
) -> list[ScheduleItem]:
    """Split a circuit into the moments of its flattened form, keeping repeat blocks compact.

    Moments are delimited by TICKs, and empty moments are dropped.  If ``insert_ticks`` is True,
    broadcast operations are split into individual gates, and moments are additionally split
    wherever a gate would reuse a qubit.  Moments are identified as though all repeat blocks were
    flattened, except that a repeat block tagged with ``immune_op_tag`` is an opaque item.

    Repeated moments are identified by tracking the moment that is open at the end of each iteration
    of a repeat block.  Once it recurs, all further iterations repeat a cycle of iterations.
    Iterations before the cycle, and at most one partial cycle, are unrolled.

    Args:
        circuit: The circuit to split into moments.
        insert_ticks: Whether to split moments to prevent qubit reuse.
        immune_op_tag: Repeat blocks with this tag are not split into moments.  Operations with
            this tag are not split into individual gates.

    Returns:
        A list of moments, repeated moments, and immune repeat blocks.

    Raises:
        ValueError: If ``insert_ticks`` is False and a repeat block with no TICK reuses a qubit
            across iterations.
    """
    return _MomentScheduler(insert_ticks, immune_op_tag).schedule(circuit)


class _MomentScheduler:
    """Implementation of ``_schedule_moments``."""

    def __init__(self, insert_ticks: bool, immune_op_tag: str | None) -> None:
        self.insert_ticks = insert_ticks
        self.immune_op_tag = immune_op_tag
        # cache of results of `_schedule`, keyed by (id(body), str(open_moment))
        self._cache: dict[tuple[int, str], tuple[list[ScheduleItem], stim.Circuit]] = {}
        self._has_boundary: dict[int, bool] = {}

    def schedule(self, circuit: stim.Circuit) -> list[ScheduleItem]:
        """Split a circuit into moments."""
        items, open_moment = self._schedule(_parse_circuit(circuit), stim.Circuit())
        if open_moment:
            items = [*items, open_moment]
        return _fold_repeated_moments(items)

    def _is_immune(self, block: stim.CircuitRepeatBlock) -> bool:
        return self.immune_op_tag is not None and self.immune_op_tag in block.tag

    def _schedule(
        self, body: tuple[stim.CircuitInstruction | _ParsedBlock, ...], open_moment: stim.Circuit
    ) -> tuple[list[ScheduleItem], stim.Circuit]:
        """Split a sequence of operations into moments, starting with a partially filled moment.

        Returns the completed items, and the moment that is still open at the end of the sequence.
        Neither the input nor the outputs of this method may be modified.
        """
        key = (id(body), str(open_moment))
        if key in self._cache:
            return self._cache[key]

        items: list[ScheduleItem] = []
        moment = open_moment.copy()
        used_qubits = set().union(*map(_gate_qubits, moment))

        def close_moment() -> None:
            nonlocal moment
            if moment:
                items.append(moment)
                moment = stim.Circuit()
                used_qubits.clear()

        for node in body:
            if isinstance(node, _ParsedBlock):
                if self._is_immune(node.block):
                    close_moment()
                    items.append(node.block)
                else:
                    new_items, moment = self._schedule_repeat(node, moment)
                    items.extend(new_items)
                    moment = moment.copy()
                    used_qubits = set().union(*map(_gate_qubits, moment))
            elif node.name == "TICK":
                close_moment()
            elif not self.insert_ticks:
                moment.append(node)
            else:
                for op in _split_targets_if_needed(node, frozenset({-1}), self.immune_op_tag):
                    qubits = _gate_qubits(op)
                    if qubits & used_qubits:
                        close_moment()
                    moment.append(op)
                    used_qubits |= qubits

        self._cache[key] = items, moment
        return items, moment

    def _schedule_repeat(
        self, node: _ParsedBlock, open_moment: stim.Circuit
    ) -> tuple[list[ScheduleItem], stim.Circuit]:
        """Split a repeat block into moments, starting with a partially filled moment."""
        repeat_count = node.block.repeat_count
        if repeat_count > 1 and not self._body_has_boundary(node.body):
            if qubits := _flattened_gate_qubits(node.body):
                raise ValueError(
                    "Qubits were operated on multiple times without a TICK in between:\n"
                    f"multiple uses: {sorted(qubits)}\n"
                    f"repeat block:\n{node.block}"
                )
            # the repeat block contains only annotations and noise, so it fits in one moment
            moment = open_moment.copy()
            moment.append(node.block)
            return [], moment

        # Simulate iterations until the open moment at the end of an iteration recurs.  Moments
        # without gates are closed at iteration boundaries, which does not change their noise.
        prefix, open_moment = _close_gate_free_moment(open_moment)
        outputs: list[list[ScheduleItem]] = []
        open_moments = [open_moment]
        iterations_by_open_moment = {str(open_moment): 0}
        start: int | None = None
        while start is None:
            items, open_moment = self._schedule(node.body, open_moment)
            closed, open_moment = _close_gate_free_moment(open_moment)
            outputs.append(items + closed)
            open_moments.append(open_moment)
            iteration = len(outputs)
            if iteration == repeat_count:
                return prefix + _concatenate(outputs), open_moment
            start = iterations_by_open_moment.setdefault(str(open_moment), iteration)
            if start == iteration:
                start = None

        # iterations after `start` repeat a cycle of `period` iterations
        period = iteration - start
        num_cycles, remainder = divmod(repeat_count - start, period)
        cycle = _concatenate(outputs[start:iteration])
        items = prefix + _concatenate(outputs[:start])
        if cycle:
            items.append(_RepeatedMoments(num_cycles, tuple(cycle), node.block.tag))
        items.extend(_concatenate(outputs[start : start + remainder]))
        return items, open_moments[start + remainder]

    def _body_has_boundary(self, body: tuple[stim.CircuitInstruction | _ParsedBlock, ...]) -> bool:
        """Does the given repeat block body necessarily end a moment?"""
        if id(body) not in self._has_boundary:
            self._has_boundary[id(body)] = any(
                (self._is_immune(node.block) or self._body_has_boundary(node.body))
                if isinstance(node, _ParsedBlock)
                else node.name == "TICK" or (self.insert_ticks and bool(_gate_qubits(node)))
                for node in body
            )
        return self._has_boundary[id(body)]


def _flattened_gate_qubits(body: tuple[stim.CircuitInstruction | _ParsedBlock, ...]) -> set[int]:
    """The qubits that operations in a (parsed) circuit act on."""
    return set().union(
        *(
            _flattened_gate_qubits(node.body)
            if isinstance(node, _ParsedBlock)
            else _gate_qubits(node)
            for node in body
        )
    )


def _concatenate(outputs: Iterable[list[ScheduleItem]]) -> list[ScheduleItem]:
    return [item for items in outputs for item in items]


def _close_gate_free_moment(moment: stim.Circuit) -> tuple[list[ScheduleItem], stim.Circuit]:
    """Close a nonempty moment that contains only annotations and noise.

    A moment without gates receives no noise, so closing it is equivalent to merging it into the
    next moment: either way, its operations precede those of the next moment and its noise.

    Returns:
        The closed moment (if any), and the moment that remains open.
    """
    if moment and all(
        isinstance(op, stim.CircuitRepeatBlock) or op_type(op.name) in (ANNOTATION, NOISE)
        for op in moment
    ):
        return [moment], stim.Circuit()
    return [], moment


def _fold_repeated_moments(items: Iterable[ScheduleItem]) -> list[ScheduleItem]:
    """Absorb unrolled iterations of repeated moments into the repeated moments.

    If the items before repeated moments end with a suffix of their body, and the items after them
    begin with the rest of the body, then these items are one more (rotated) iteration of the body.
    """
    result = [
        dataclasses.replace(item, body=tuple(_fold_repeated_moments(item.body)))
        if isinstance(item, _RepeatedMoments)
        else item
        for item in items
    ]
    index = 0
    while index < len(result):
        item = result[index]
        if isinstance(item, _RepeatedMoments):
            size = len(item.body)
            # prefer splits that do not rotate the body
            for split in sorted(range(size + 1), key=lambda split: 0 < split < size):
                start, stop = index - (size - split), index + 1 + split
                if (
                    start >= 0
                    and result[start:index] == list(item.body[split:])
                    and result[index + 1 : stop] == list(item.body[:split])
                ):
                    body = item.body[split:] + item.body[:split]
                    result[start:stop] = [_RepeatedMoments(item.repeat_count + 1, body, item.tag)]
                    index = start
                    break
            else:
                index += 1
        else:
            index += 1
    return result


def _schedule_barrier_qubits(items: Iterable[ScheduleItem]) -> set[int]:
    """Qubits that pending idle errors may not be deferred past in a moment schedule."""
    qubits: set[int] = set()
    for item in items:
        if isinstance(item, _RepeatedMoments):
            qubits |= _schedule_barrier_qubits(item.body)
        else:
            qubits |= _idle_barrier_qubits(
                [item] if isinstance(item, stim.CircuitRepeatBlock) else item
            )
    return qubits


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
