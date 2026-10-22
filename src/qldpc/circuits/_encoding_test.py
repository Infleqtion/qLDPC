# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _encoding.py."""

import pytest
import stim

from qldpc import circuits, codes, math
from qldpc.circuits import _encoding
from qldpc.objects import Pauli


def test_get_logical_tableau_from_code_data() -> None:
    """Identify a logical tableau, and reject invalid encoders and non-logical circuits."""
    code = codes.FiveQubitCode()
    encoder, decoder = circuits.get_encoder_and_decoder(code)
    logical_x = math.op_to_string(code.get_logical_ops(Pauli.X, symplectic=True)[0])
    logical_tableau = _encoding.get_logical_tableau_from_code_data(
        1, 0, encoder, decoder, logical_x.to_tableau().to_circuit()
    )
    assert logical_tableau == stim.Tableau.from_named_gate("X")

    with pytest.raises(ValueError, match="does not implement a logical operation"):
        _encoding.get_logical_tableau_from_code_data(1, 0, encoder, decoder, stim.Circuit("X 0"))
    with pytest.raises(ValueError, match="incompatible dimensions"):
        _encoding.get_logical_tableau_from_code_data(
            1, 0, stim.Tableau(1), stim.Tableau(2), stim.Circuit()
        )
