# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _common.py."""

import pytest

from qldpc import circuits, codes
from qldpc.circuits.memory import _common


def test_validate_syndrome_qubit_ids() -> None:
    """Validate default, explicit, and unsupported syndrome-measurement layouts."""
    code = codes.SurfaceCode(2)
    qubit_ids = circuits.QubitIDs.from_code(code)
    assert _common.validate_syndrome_qubit_ids(code) == qubit_ids
    assert _common.validate_syndrome_qubit_ids(code, qubit_ids) is not qubit_ids

    with pytest.raises(ValueError, match="subsystem"):
        _common.validate_syndrome_qubit_ids(codes.BaconShorCode(2))
