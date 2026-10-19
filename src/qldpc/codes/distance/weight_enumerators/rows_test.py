# SPDX-License-Identifier: Apache-2.0

"""Tests for packed physical rows, local Clifford transport and storage budgets."""

from __future__ import annotations

import numpy as np
import pytest

from . import rows
from .conftest import _native


@pytest.mark.parametrize("width", [1, 63, 64, 65, 129])
def test_pack_layout_and_round_trip(width: int) -> None:
    """Coordinate zero is the low bit of word zero, and padding never leaks into unpacking."""
    matrix = np.random.default_rng(width).integers(0, 2, size=(3, width), dtype=np.uint8)
    packed = rows.pack(matrix)
    assert packed.dtype == np.uint64 and packed.shape == (3, (width + 63) // 64)
    for row, words in zip(matrix, packed):
        value = sum(int(word) << (64 * jj) for jj, word in enumerate(words))
        assert value == sum(int(bit) << jj for jj, bit in enumerate(row))
    assert np.array_equal(rows.unpack(packed, width), matrix)
    assert rows.unpack(rows.pack(np.empty((0, width), dtype=np.uint8)), width).shape == (0, width)


def test_support_keys_compare_whole_rows() -> None:
    """Rows that agree in a single word are still distinct keys."""
    words = np.array([[1, 2], [1, 3], [1, 2]], dtype=np.uint64)
    keys = rows.get_support_keys(words)
    assert len(keys) == 3 and keys[0] == keys[2] and keys[0] != keys[1]
    assert len(np.unique(keys)) == 2


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("width", [5, 130])
def test_transform_rows_against_bit_oracle(interpreted: bool, width: int) -> None:
    """Targets may straddle word boundaries; untouched coordinates must be preserved."""
    transform = _native(rows.transform_rows, interpreted)
    rng = np.random.default_rng(width)
    matrix = rng.integers(0, 2, size=(4, width), dtype=np.uint8)
    targets = [0, width - 1, width // 2] if width > 64 else [0, 4]
    local = rng.integers(0, 2, size=(len(targets), len(targets)), dtype=np.uint8)
    expected = matrix.copy()
    expected[:, targets] = (matrix[:, targets] @ local.T) % 2
    packed = rows.pack(matrix)
    before = packed.copy()
    result = transform(
        packed, np.asarray(targets, dtype=np.int64), rows.pack(local)[:, 0].astype(np.uint64)
    )
    assert np.array_equal(rows.unpack(result, width), expected)
    assert np.array_equal(packed, before)


def test_storage_budget_reserves_bases_and_allowance() -> None:
    budget = rows.StorageBudget(limit_bytes=5000, basis_bytes=1000, batch_size=7)
    assert budget.available() == 5000 - 1000 - 1024
    assert budget.available(300) == 5000 - 1000 - 300 - 1024
    assert budget.batch_size == 7
