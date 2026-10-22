# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _fields.py."""

from __future__ import annotations

import galois

from qldpc.abstract._fields import resolve_field


def test_resolve_field() -> None:
    """resolve_field accepts None (GF2 default), a field order, or a galois field type."""
    assert resolve_field(None) is galois.GF2
    assert resolve_field(3).order == 3
    field = galois.GF(4)
    assert resolve_field(field) is field
