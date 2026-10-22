# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _common.py."""

from __future__ import annotations

import pytest
import stim

from qldpc import codes
from qldpc.circuits import _common


def test_restriction() -> None:
    """restrict_to_qubits passes through for qubit codes and raises for qudit codes."""

    @_common.restrict_to_qubits
    def get_empty_circuit(code: codes.QuditCode) -> stim.Circuit:
        return stim.Circuit()

    get_empty_circuit(codes.SurfaceCode(2, field=2))  # this does not raise an error

    with pytest.raises(ValueError, match="only supported for qubit codes"):
        get_empty_circuit(codes.SurfaceCode(2, field=3))
