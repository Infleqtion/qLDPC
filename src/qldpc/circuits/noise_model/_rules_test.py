# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _rules.py."""

import pytest
import stim

from qldpc import circuits
from qldpc.circuits.noise_model import _rules


def test_rule_with_immunity() -> None:
    """Noise rules are dropped or conditioned for operations that touch immune qubits."""
    rule = circuits.NoiseRule(after={"XX": 0.1, "ZZ": 0.2, "XZ": 0.3})
    op = stim.CircuitInstruction("CX", [0, 1])
    assert _rules.rule_with_immunity(rule, op, frozenset(), immunize_gates=True) is rule
    assert _rules.rule_with_immunity(rule, op, frozenset({2}), immunize_gates=True) is rule
    assert _rules.rule_with_immunity(rule, op, frozenset({0, 1}), immunize_gates=False) is None
    assert _rules.rule_with_immunity(rule, op, frozenset({1}), immunize_gates=True) is None
    rule = circuits.NoiseRule(after={"XI": 0.1, "XX": 0.2})
    conditioned = _rules.rule_with_immunity(rule, op, frozenset({1}), immunize_gates=False)
    assert conditioned is not None and conditioned.after == circuits.PauliChannel({"XI": 0.1})
    with pytest.raises(ValueError, match="partial immunity"):
        _rules.rule_with_immunity(
            circuits.NoiseRule(readout_error=0.1),
            stim.CircuitInstruction("MXX", [0, 1]),
            frozenset({1}),
            immunize_gates=False,
        )


def test_rule_validation() -> None:
    """Rules are validated against the shape of the operations they apply to."""
    rule = circuits.NoiseRule(after={"X": 0.1})
    _rules.validate_rule_for_arity(rule, 1, "context")
    with pytest.raises(ValueError, match="arity 1; expected 2"):
        _rules.validate_rule_for_arity(rule, 2, "context")
    with pytest.raises(ValueError, match="readout_error"):
        _rules.validate_rule_for_arity(circuits.NoiseRule(readout_error=0.1), 1, "context")
    with pytest.raises(ValueError, match="reset_error"):
        _rules.validate_rule_for_arity(circuits.NoiseRule(reset_error=0.1), 1, "context")

    _rules.validate_custom_rule(
        circuits.NoiseRule(readout_error=0.1), stim.CircuitInstruction("MR", [0])
    )
    with pytest.raises(ValueError, match="rule_func returned a rule for 'CX'"):
        _rules.validate_custom_rule(rule, stim.CircuitInstruction("CX", [0, 1]))


def test_as_noise_rule() -> None:
    """Error specifications are normalized to noise rules."""
    rule = circuits.NoiseRule(after={"X": 0.1})
    assert _rules.as_noise_rule(rule, 1) is rule
    assert _rules.as_noise_rule(None, 1) is None
    assert _rules.as_noise_rule(False, 1) is None
    assert repr(_rules.as_noise_rule(circuits.PauliChannel({"X": 0.1}), 1)) == repr(rule)
    assert repr(_rules.as_noise_rule({"X": 0.1}, 1)) == repr(rule)
    assert repr(_rules.as_noise_rule(0.1, 2)) == repr(
        circuits.NoiseRule(after=circuits.PauliChannel.depolarizing(2, 0.1))
    )
    with pytest.raises(TypeError, match="expected a float"):
        _rules.as_noise_rule(True, 1)


def test_normalize_clifford_nq_error() -> None:
    """The clifford_nq_error argument is normalized to a dictionary of noise rules."""
    rule = circuits.NoiseRule(after={"XXX": 0.1})
    assert _rules.normalize_clifford_nq_error(None) == {}
    normalized = _rules.normalize_clifford_nq_error(
        {1: 0.1, 2: circuits.PauliChannel({"XX": 0.1}), 3: rule, 4: {"XXXX": 0.1}, 5: 0}
    )
    assert {weight: rule.after for weight, rule in normalized.items()} == {
        1: circuits.PauliChannel.depolarizing(1, 0.1),
        2: circuits.PauliChannel({"XX": 0.1}),
        3: rule.after,
        4: circuits.PauliChannel({"XXXX": 0.1}),
    }
    with pytest.raises(ValueError, match="must be >= 1"):
        _rules.normalize_clifford_nq_error({0: 0.1})
    with pytest.raises(TypeError, match="unsupported type"):
        _rules.normalize_clifford_nq_error({1: "X"})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match=r"is not in \[0, 1\]"):
        _rules.normalize_clifford_nq_error({1: 1.5})
