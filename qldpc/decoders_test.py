"""Unit tests for decoder.py

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

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import decoders


def test_custom_decoder() -> None:
    """Inject custom decoders."""

    class CustomDecoder(decoders.Decoder):
        def __init__(self, matrix: npt.NDArray[np.int_]) -> None: ...
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return syndrome

    matrix = np.eye(2, dtype=int)
    syndrome = np.zeros(2, dtype=int)
    result = decoders.decode(matrix, syndrome, decoder_constructor=CustomDecoder)
    assert result is syndrome

    decoder = CustomDecoder(matrix)
    result = decoders.decode(matrix, syndrome, static_decoder=decoder)
    assert result is syndrome


def test_decoding() -> None:
    """Decode a simple problem."""
    matrix = np.eye(3, 2, dtype=int)
    syndrome = np.array([1, 1, 0])
    error = np.array([1, 1], dtype=int)
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_ILP=False))
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_ILP=True))
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_MWPM=True))
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_BF=True))
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_BP_OSD=True))
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_BP_LSD=True))

    # decode over trinary field
    modulus = 3
    answer = -error % modulus
    result = decoders.decode(-matrix, syndrome, with_ILP=True, modulus=modulus, lower_bound_row=-1)
    assert np.allclose(answer, result)


def test_decoding_errors() -> None:
    """Fail to solve an invalid optimization problem."""
    matrix = np.ones((2, 2), dtype=int)
    syndrome = np.array([0, 1], dtype=int)

    with pytest.raises(ValueError, match="must have modulus >= 2"):
        decoders.get_decoder_ILP(matrix, with_ILP=True, modulus=1)

    with pytest.raises(ValueError, match="row index must be an integer"):
        decoders.get_decoder_ILP(matrix, with_ILP=True, lower_bound_row="x")

    with pytest.raises(ValueError, match="could not be found"):
        with pytest.warns(UserWarning, match="infeasible or unbounded"):
            decoders.decode(matrix, syndrome, with_ILP=True)
