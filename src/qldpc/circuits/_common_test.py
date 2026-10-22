# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _common.py."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import circuits, codes
from qldpc.circuits import _common


def test_restriction() -> None:
    """restrict_to_qubits passes through for qubit codes and raises for qudit codes."""

    @_common.restrict_to_qubits
    def get_empty_circuit(code: codes.QuditCode) -> stim.Circuit:
        return stim.Circuit()

    get_empty_circuit(codes.SurfaceCode(2, field=2))  # this does not raise an error

    with pytest.raises(ValueError, match="only supported for qubit codes"):
        get_empty_circuit(codes.SurfaceCode(2, field=3))


def test_postselection_masks() -> None:
    """Automatic selection chooses flags, while explicit out-of-range indices are rejected."""
    record = circuits.DetectorRecord({"prep": [0], "flags": [1]})
    detector_mask, observable_mask = _common.get_postselection_masks(True, (), record, 0)
    assert detector_mask is not None
    assert np.array_equal(detector_mask, [2])
    assert observable_mask is None

    detector_mask, observable_mask = _common.get_postselection_masks(False, [-1], record, 3)
    assert detector_mask is None
    assert observable_mask is not None
    assert np.array_equal(observable_mask, [4])

    with pytest.raises(ValueError, match="2 selectable prep/flag detectors"):
        _common.get_postselection_masks([2], (), record, 0)
    with pytest.raises(ValueError, match="Observable post-selection"):
        _common.get_postselection_masks((), [1], record, 1)
