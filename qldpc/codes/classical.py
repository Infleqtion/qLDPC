"""Concrete classical error correction codes

   Copyright 2023 The qLDPC Authors and Infleqtion Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import annotations

import itertools

import galois
import numpy as np
import numpy.typing as npt

from qldpc import codes
from qldpc.abstract import DEFAULT_FIELD_ORDER


class RepetitionCode(codes.ClassicalCode):
    """Classical repetition code."""

    def __init__(self, bits: int, field: int | None = None) -> None:
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        self._matrix = self.field.Zeros((bits - 1, bits))
        for row in range(bits - 1):
            self._matrix[row, row] = 1
            self._matrix[row, row + 1] = -self.field(1)


class RingCode(codes.ClassicalCode):
    """Classical ring code: repetition code with periodic boundary conditions."""

    def __init__(self, bits: int, field: int | None = None) -> None:
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        self._matrix = self.field.Zeros((bits, bits))
        for row in range(bits):
            self._matrix[row, row] = 1
            self._matrix[row, (row + 1) % bits] = -self.field(1)


class HammingCode(codes.ClassicalCode):
    """Classical Hamming code."""

    def __init__(self, rank: int, field: int | None = None) -> None:
        """Construct a Hamming code of a given rank."""
        self._exact_distance = 3
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        if self.field.order == 2:
            # parity check matrix: columns = all nonzero bitstrings
            bitstrings = list(itertools.product([0, 1], repeat=rank))
            self._matrix = self.field(bitstrings[1:]).T

        else:
            # More generally, columns = [maximal set of linearly independent strings], so collect
            # together all strings whose first nonzero element is a 1.
            strings = [
                (0,) * top_row + (1,) + rest
                for top_row in range(rank - 1, -1, -1)
                for rest in itertools.product(range(self.field.order), repeat=rank - top_row - 1)
            ]
            self._matrix = self.field(strings).T


class ReedSolomonCode(codes.ClassicalCode):
    """Classical Reed-Solomon code.

    Source: https://galois.readthedocs.io/en/v0.3.8/api/galois.ReedSolomon/
    References:
    - https://errorcorrectionzoo.org/c/reed_solomon
    - https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes6.pdf
    """

    def __init__(self, bits: int, dimension: int) -> None:
        codes.ClassicalCode.__init__(self, galois.ReedSolomon(bits, dimension).H)


class BCHCode(codes.ClassicalCode):
    """Classical binary BCH code.

    Source: https://galois.readthedocs.io/en/v0.3.8/api/galois.BCH/
    References:
    - https://errorcorrectionzoo.org/c/bch
    - https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes6.pdf
    """

    def __init__(self, bits: int, dimension: int) -> None:
        if "0" in format(bits, "b"):
            raise ValueError("BCH codes only defined for 2^m - 1 bits with integer m.")
        codes.ClassicalCode.__init__(self, galois.BCH(bits, dimension).H)


class ReedMullerCode(codes.ClassicalCode):
    """Classical Reed-Muller code.

    References:
    - https://errorcorrectionzoo.org/c/reed_muller
    - https://feog.github.io/10-coding.pdf
    """

    def __init__(self, order: int, size: int, field: int | None = None) -> None:
        self._assert_valid_params(order, size)
        self._exact_distance = 2 ** (size - order)
        self._order = order
        self._size = size

        generator = ReedMullerCode.get_generator(order, size)
        self._matrix = codes.ClassicalCode(generator, field).generator
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)

    @classmethod
    def get_generator(cls, order: int, size: int) -> npt.NDArray[np.int_]:
        """Get the generator matrix for the specified Reed-Muller code."""
        cls._assert_valid_params(order, size)

        if order == 0:
            return np.ones(2**size, dtype=int)
        if order == size:
            return np.identity(2**size, dtype=int)

        mat_a = cls.get_generator(order, size - 1)
        mat_b = cls.get_generator(order - 1, size - 1)
        mat_z = np.zeros_like(mat_b)
        return np.block([[mat_a, mat_a], [mat_z, mat_b]]).astype(int)

    @classmethod
    def _assert_valid_params(self, order: int, size: int) -> None:
        if not (size >= 0 and 0 <= order <= size):
            raise ValueError(
                "Reed-Muller code R(r,m) must have m >= 0 and 0 <= r <= m\n"
                + f"Provided: (r,m) = ({order},{size})"
            )
