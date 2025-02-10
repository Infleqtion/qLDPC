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

import galois
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
    error = np.array([1, 1], dtype=int)
    syndrome = np.array([1, 1, 0], dtype=int)
    default_decoder = decoders.get_decoder(matrix)

    assert np.allclose(error, default_decoder.decode(syndrome))
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_ILP=True))
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_MWPM=True))
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_BF=True))
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_BP_LSD=True))

    # the distance of the given code is undefined, so lookup decoding to half the distance fails
    assert np.allclose([0, 0], decoders.decode(matrix, syndrome, with_lookup=True))

    # but it works if we manually tell it to try and decode errors of weight <= 2
    assert np.allclose(error, decoders.decode(matrix, syndrome, with_lookup=True, max_weight=2))

    # decode two copies of the problem
    block_error = np.concatenate([error, error])
    block_syndrome = np.concatenate([syndrome, syndrome])
    block_decoder = decoders.BlockDecoder(syndrome.size, default_decoder)
    assert np.allclose(block_error, block_decoder.decode(block_syndrome))

    # decode directly from a corrupted code word
    code_word = np.zeros_like(error)
    corrupted_code_word = (code_word + error) % 2
    direct_decoder = decoders.DirectDecoder.from_indirect(default_decoder, matrix)
    assert np.allclose(code_word, direct_decoder.decode(corrupted_code_word))

    # decode over trinary field
    field = galois.GF(3)
    matrix = -field(matrix)
    error = -field(error)
    assert np.allclose(error.view(np.ndarray), decoders.decode(matrix, syndrome))

    # decode directly from a corrupted code word
    code_word = field.Zeros(error.size)
    corrupted_code_word = code_word + field(error)
    ilp_decoder = decoders.ILPDecoder(matrix)
    direct_decoder = decoders.DirectDecoder.from_indirect(ilp_decoder, field(matrix))
    assert np.allclose(code_word.view(np.ndarray), direct_decoder.decode(corrupted_code_word))


def test_decoding_errors() -> None:
    """Fail to solve an invalid optimization problem."""
    matrix = np.ones((2, 2), dtype=int)
    syndrome = np.array([0, 1], dtype=int)

    with pytest.raises(ValueError, match="ILP decoding only supports prime number fields"):
        decoders.decode(galois.GF(4)(matrix), syndrome, with_ILP=True)

    with pytest.raises(ValueError, match="could not be found"):
        with pytest.warns(UserWarning, match="infeasible or unbounded"):
            decoders.decode(matrix, syndrome, with_ILP=True)
