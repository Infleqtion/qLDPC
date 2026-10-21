# SPDX-License-Identifier: Apache-2.0

"""Functional counts and coverage of the MacWilliams method."""

from __future__ import annotations

import numpy as np
import pytest

from . import _macwilliams as mw


@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize("num_bits, count_two", [(8, 12), (70, 1903)])
def test_logical_counts(
    monkeypatch: pytest.MonkeyPatch, num_bits: int, count_two: int, compiled: bool
) -> None:
    """Count two disjoint four-qubit blocks, with any remaining qubits free."""
    if not compiled:
        monkeypatch.setattr(mw, "_add_span_weights", mw._add_span_weights.py_func)
        monkeypatch.setattr(mw, "_transform_weights", mw._transform_weights.py_func)
    monkeypatch.setattr(mw, "_CHUNK_SIZE", 1)
    checks = np.zeros((2, num_bits), dtype=np.uint8)
    checks[0, :4] = checks[1, -4:] = 1
    enumerators = mw._CheckEnumerators(checks, checks if num_bits == 8 else checks.copy())
    assert enumerators.count_logicals(2) == (count_two, count_two)
    if num_bits > 64:
        assert enumerators.count_logicals(num_bits // 2)[0] > 2**64


def test_check_rank_limit() -> None:
    checks = np.eye(mw._MAX_RANK + 1, mw._MAX_RANK + 2, dtype=np.uint8)
    with pytest.raises(ValueError, match="check ranks"):
        mw._CheckEnumerators(checks, checks[:0])
