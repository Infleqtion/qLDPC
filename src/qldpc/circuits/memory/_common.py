# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by syndrome measurement strategies."""

from __future__ import annotations

from qldpc import codes

from ..bookkeeping import QubitIDs


def validate_syndrome_qubit_ids(
    code: codes.QuditCode, qubit_ids: QubitIDs | None = None
) -> QubitIDs:
    """Validate the qubit layout for a syndrome-measurement strategy.

    Args:
        code: The code whose syndromes will be measured.
        qubit_ids: Integer indices for the data and check qubits.  Defaults to
            ``QubitIDs.from_code(code)``.

    Returns:
        A validated qubit layout.

    Raises:
        ValueError: If the code is a subsystem code or the qubit layout is invalid.
    """
    if code.is_subsystem_code:
        raise ValueError("Syndrome measurement strategies require a non-subsystem code")
    return QubitIDs.validated(qubit_ids or QubitIDs.from_code(code), code)
