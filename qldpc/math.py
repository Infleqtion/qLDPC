"""Miscellaneous mathematical and linear algebra methods

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

import functools
from typing import TypeVar

import galois
import numpy as np
import numpy.typing as npt
import scipy.special

IntegerArray = TypeVar("IntegerArray", npt.NDArray[np.int_], galois.FieldArray)


def symplectic_conjugate(vectors: IntegerArray) -> IntegerArray:
    """Take symplectic vectors to their duals.

    The symplectic conjugate of a Pauli string swaps its X and Z support, and multiplies its X
    sector by -1, taking P = [P_x|P_z] -> [-P_z|P_x], such that the symplectic inner product between
    Pauli strings P and Q is ⟨P,Q⟩_s = P_x @ Q_z - P_z @ Q_x = symplectic_conjugate(P) @ Q.
    """
    assert vectors.shape[-1] % 2 == 0
    conjugated_string = vectors.copy().reshape(-1, 2, vectors.shape[-1] // 2)[:, ::-1, :]
    conjugated_string[:, 0, :] *= -1
    return conjugated_string.reshape(vectors.shape)  # type:ignore[return-value]


def first_nonzero_cols(matrix: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Get the first nonzero column for every row in a matrix."""
    if matrix.size == 0:
        return np.array([], dtype=int)
    boolean_matrix = matrix.reshape(matrix.shape[0], -1).view(np.ndarray).astype(bool)
    return np.argmax(boolean_matrix, axis=1)


@functools.cache
def log_choose(n: int, k: int) -> float:
    """Natural logarithm of (n choose k) = n! / ( k! * (n-k)! )."""
    return (
        scipy.special.gammaln(n + 1)
        - scipy.special.gammaln(k + 1)
        - scipy.special.gammaln(n - k + 1)
    )
