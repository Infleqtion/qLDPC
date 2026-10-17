# SPDX-License-Identifier: Apache-2.0

"""Compatibility tests for the renamed state-preparation spread module."""

from __future__ import annotations

import pytest

from qldpc.circuits import fault_tolerance
from qldpc.circuits.benchmarking import state_prep_spread


@pytest.mark.parametrize("name", fault_tolerance.__all__)
def test_deprecated_imports(name: str) -> None:
    """Old qualified imports warn and retain the canonical functions and result types."""
    with pytest.warns(
        DeprecationWarning,
        match=rf"use qldpc\.circuits\.benchmarking\.state_prep_spread\.{name}",
    ):
        assert getattr(fault_tolerance, name) is getattr(state_prep_spread, name)


def test_deprecated_from_import() -> None:
    """A caller can still use the old module in an explicit from-import."""
    with pytest.warns(DeprecationWarning, match="certify_state_prep_spread is deprecated"):
        from qldpc.circuits.fault_tolerance import certify_state_prep_spread

    assert certify_state_prep_spread is state_prep_spread.certify_state_prep_spread


def test_unknown_name() -> None:
    """The compatibility module does not silently accept unknown public names."""
    name = "does_not_exist"
    with pytest.raises(AttributeError, match=name):
        getattr(fault_tolerance, name)
