# SPDX-License-Identifier: Apache-2.0

"""Normalization and validation of the noise rules that a noise model applies to operations.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

from collections.abc import Mapping

import stim

from ._operations import JUST_MEASURE_OPS, JUST_RESET_OPS, MEASURE_AND_RESET_OPS
from .channels import AbstractPauliChannel, PauliChannel
from .rules import ErrorSpec, NoiseRule

####################################################################################################
# normalization of error specifications


def as_noise_rule(error: ErrorSpec | None, default_arity: int) -> NoiseRule | None:
    """Normalize a noise-error argument to a NoiseRule (or None if the input is ``None``).

    A float scalar is interpreted as a uniform depolarizing channel of the given ``default_arity``.
    A ``PauliChannel`` / ``PauliChannelSequence`` — or a raw ``Mapping[str, float]`` of Pauli-string
    probabilities, which is auto-wrapped as a ``PauliChannel`` — is used as the rule's ``after``.
    Storing the depolarizing shape as a ``PauliChannel`` (rather than a Mapping-form
    ``DEPOLARIZE{n}`` fragment) preserves the Pauli-channel structure so downstream
    ``PauliChannel.conditioned_on`` can project it under partial immunity.

    Does NOT trivialize an empty NoiseRule to ``None`` — callers must run their guard checks on the
    declared shape (e.g. arity) BEFORE trivializing, so that user typos like
    ``PauliChannel({}, num_qubits=3)`` in a 1-qubit-only slot are surfaced.  Trivialize via
    ``rule or None`` after the guard checks.
    """
    if isinstance(error, NoiseRule):
        return error
    if error is False:
        return None
    if isinstance(error, AbstractPauliChannel):
        return NoiseRule(after=error)
    if isinstance(error, Mapping):
        return NoiseRule(after=PauliChannel(error))
    if error is None:
        return None
    if isinstance(error, bool) or not isinstance(error, (int, float)):
        raise TypeError(
            f"expected a float, PauliChannel, PauliChannelSequence, Mapping[str, float], or "
            f"NoiseRule; got {type(error).__name__!r}"
        )
    return NoiseRule(after=PauliChannel.depolarizing(default_arity, float(error)))


def normalize_clifford_nq_error(
    error: Mapping[int, ErrorSpec] | None,
) -> dict[int, NoiseRule]:
    """Normalize the ``clifford_nq_error`` argument to a ``dict[int, NoiseRule]``.

    - Floats become uniform ``k``-qubit depolarizing noise: ``DEPOLARIZE1`` for ``k == 1``,
      ``DEPOLARIZE2`` for ``k == 2``, and ``PauliChannel.depolarizing`` otherwise.
    - ``PauliChannel`` / ``PauliChannelSequence`` values (or raw Pauli-string
      ``Mapping[str, float]`` dicts, auto-wrapped as a ``PauliChannel``) become a
      ``NoiseRule(after=<channel>)``.
    - ``NoiseRule`` values are used directly.
    - Falsy entries (0.0, empty NoiseRule, empty channel) are dropped.
    - Entries are rejected if their joint Pauli channel's ``num_qubits`` disagrees with the key, if
      their ``after`` broadcast channels are incompatible with ``k`` qubits, or if they set
      ``readout_error`` / ``reset_error`` (Pauli-product Cliffords are neither).
    """
    if not error:
        return {}
    result: dict[int, NoiseRule] = {}
    for weight, entry in error.items():
        if weight < 1:
            raise ValueError(f"clifford_nq_error key {weight} must be >= 1")
        if isinstance(entry, NoiseRule):
            rule = entry
        elif isinstance(entry, AbstractPauliChannel):
            rule = NoiseRule(after=entry)
        elif isinstance(entry, Mapping):
            rule = NoiseRule(after=PauliChannel(entry))
        elif isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError(
                f"clifford_nq_error[{weight}] has unsupported type {type(entry).__name__}; "
                "expected NoiseRule, PauliChannel, PauliChannelSequence, Mapping[str, float], int, "
                "or float"
            )
        else:
            if not (0 <= entry <= 1):
                raise ValueError(f"clifford_nq_error[{weight}]={entry} is not in [0, 1]")
            if entry == 0:
                continue
            rule = NoiseRule(after=PauliChannel.depolarizing(weight, entry))
        validate_rule_for_arity(rule, weight, f"clifford_nq_error[{weight}]")
        if rule:
            result[weight] = rule
    return result


####################################################################################################
# validation of noise rules


def validate_rule_for_arity(
    rule: NoiseRule,
    num_qubits: int,
    context: str,
    *,
    can_measure: bool = False,
    can_reset: bool = False,
) -> None:
    """Reject a NoiseRule whose channels are ambiguous / incompatible on ``num_qubits`` qubits.

    - The rule's ``after`` arity must equal ``num_qubits`` unless ``after`` is ``None`` (the trivial
      default).  An explicitly-declared arity — even on an empty PauliChannel like
      ``PauliChannel({}, num_qubits=3)`` — is honored so callers can flag user typos.
    - ``readout_error`` is only meaningful if ``can_measure`` is True.
    - ``reset_error`` is only meaningful if ``can_reset`` is True.
    """
    if rule.after is not None and rule.after.num_qubits != num_qubits:
        raise ValueError(
            f"{context}: `after` has arity {rule.after.num_qubits}; expected {num_qubits}"
        )
    if rule.readout_error is not None and not can_measure:
        raise ValueError(f"{context}: `readout_error` is only valid on measurement gates")
    if rule.reset_error is not None and not can_reset:
        raise ValueError(f"{context}: `reset_error` is only valid on reset gates")


def validate_custom_rule(rule: NoiseRule, op: stim.CircuitInstruction) -> None:
    """Reject a ``rule_func`` result that is incompatible with the gate application ``op``."""
    num_qubits = sum(1 for target in op.targets_copy() if not target.is_combiner)
    validate_rule_for_arity(
        rule,
        num_qubits,
        context=f"rule_func returned a rule for {op.name!r}",
        can_measure=op.name in JUST_MEASURE_OPS or op.name in MEASURE_AND_RESET_OPS,
        can_reset=op.name in JUST_RESET_OPS or op.name in MEASURE_AND_RESET_OPS,
    )


####################################################################################################
# noise rules on immune qubits


def rule_with_immunity(
    rule: NoiseRule,
    op: stim.CircuitInstruction,
    immune_qubits: frozenset[int],
    *,
    immunize_gates: bool,
) -> NoiseRule | None:
    """Return the ``NoiseRule`` to apply to ``op`` given ``immune_qubits``, or ``None`` to skip.

    Policy — ``op``'s qubits fall into three immunity states:

    - **None immune**: apply ``rule`` unchanged.
    - **All immune**: skip noise entirely (return ``None``); the op still appears in the output
      circuit, but no ``after``, ``readout_error``, or ``reset_error`` is applied.
    - **Mixed** (only possible for atomic ``k>=2`` ops the splitter cannot decompose further —
      2-qubit Clifford pairs, ``MXX``/``MYY``/``MZZ``, and MPP/SPP Pauli products):
        - ``immunize_gates=True`` (default): dead simple — drop everything (``after``,
          ``readout_error``, ``reset_error``).  The gate touches an immune qubit, so no noise is
          applied.
        - ``immunize_gates=False``: only ``PauliChannel`` ``after``-noise supports proper
          conditioning via ``PauliChannel.conditioned_on``.  Anything else (a ``stim.Circuit``
          ``after``, ``readout_error``, or ``reset_error``) has no defined projection on the
          identity-on-immune subspace and raises.
    """
    if not immune_qubits:
        return rule
    qubits = [t.qubit_value for t in op.targets_copy() if t.qubit_value is not None]
    immune_flags = [q in immune_qubits for q in qubits]
    if not any(immune_flags):
        return rule
    if all(immune_flags):
        return None
    # Partial immunity: the atom touches both immune and non-immune qubits.
    if immunize_gates:
        return None  # dead simple: no noise on gates that touch immune qubits.
    # immunize_gates=False: only PauliChannel `after` supports proper conditioning.
    if isinstance(rule.after, PauliChannel) and rule.after:
        immune_positions = [i for i, immune in enumerate(immune_flags) if immune]
        return NoiseRule(after=rule.after.conditioned_on(immune_positions))
    raise ValueError(
        f"Cannot apply a rule to {op.name!r} with partial immunity (immune qubits: "
        f"{sorted(q for q in qubits if q in immune_qubits)}) under immunize_gates=False.  Only "
        "PauliChannel `after`-noise supports conditioning; set immunize_gates=True to drop noise "
        "on gates that touch immune qubits.  If this edge case matters for your use case, please "
        "open an issue at https://github.com/qLDPCOrg/qLDPC/issues."
    )
