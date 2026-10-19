# SPDX-License-Identifier: Apache-2.0

"""Packed physical rows shared by the weight-enumeration backends.

A physical coordinate j of a packed row is bit j % 64 of word j // 64, so coordinate zero is the
low bit.  This differs from the layout used by the distance searches in ``_binary``; do not mix
the two.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from numba import njit

from .._binary_numba import popcount_word

BinaryMatrix = npt.NDArray[np.uint8]
PackedMatrix = npt.NDArray[np.uint64]

####################################################################################################
# storage budget


class StorageBudget(NamedTuple):
    """Conservative working-storage limits for one counting call.

    ``basis_bytes`` is the storage already committed to logical bases.  The limit covers working
    arrays only, not the interpreter's or the JIT's resident memory.
    """

    limit_bytes: int
    basis_bytes: int
    batch_size: int

    def available(self, reserved_bytes: int = 0) -> int:
        """Bytes left after the bases, ``reserved_bytes`` and a small fixed allowance."""
        return self.limit_bytes - self.basis_bytes - reserved_bytes - 1024


####################################################################################################
# packing


def pack(matrix: BinaryMatrix) -> PackedMatrix:
    """Pack binary rows into little-endian uint64 words."""
    padded = np.pad(matrix, ((0, 0), (0, -matrix.shape[1] % 64)))
    return np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little").view("<u8"))


def unpack(matrix: PackedMatrix, width: int) -> BinaryMatrix:
    """Inverse of ``pack`` for rows with ``width`` physical coordinates."""
    return np.unpackbits(matrix.view(np.uint8), axis=1, bitorder="little")[:, :width].copy()


def get_support_keys(words: PackedMatrix) -> npt.NDArray[np.void]:
    """Compare complete packed rows, not individual words or collision-prone hashes."""
    return words.view(np.dtype((np.void, 8 * words.shape[1]))).reshape(-1)


####################################################################################################
# local Clifford transport


@njit(cache=True)
def transform_rows(
    rows: PackedMatrix, targets: npt.NDArray[np.int64], matrix: npt.NDArray[np.uint64]
) -> PackedMatrix:
    """Propagate all packed rows through a small local binary matrix on ordered targets."""
    result = rows.copy()
    for index, row in enumerate(rows):
        # Gather the target bits into one local word, then rewrite each target from its matrix row.
        local = np.uint64(0)
        for jj, target in enumerate(targets):
            local |= ((row[target // 64] >> np.uint64(target % 64)) & np.uint64(1)) << np.uint64(jj)
        for jj, target in enumerate(targets):
            mask = np.uint64(1) << np.uint64(target % 64)
            result[index, target // 64] &= ~mask
            if popcount_word(local & matrix[jj]) & 1:
                result[index, target // 64] |= mask
    return result
