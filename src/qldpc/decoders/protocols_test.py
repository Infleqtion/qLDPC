# SPDX-License-Identifier: Apache-2.0

"""Unit tests for protocols.py."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import decoders


def test_error_decoder_protocols() -> None:
    """Error decoder protocols provide decode as an alias for decode_errors, and vice versa."""
    syndromes = np.eye(2, dtype=int)

    # a subclass implements either name of a decoding method, and inherits the other
    class NewDecoder(decoders.BatchErrorDecoder):
        def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return 2 * syndrome

        def decode_errors_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return 3 * syndromes

    class OldDecoder(decoders.BatchErrorDecoder):
        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return 2 * syndrome

        def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return 3 * syndromes

    for decoder in [NewDecoder(), OldDecoder()]:
        assert np.array_equal(decoder.decode_errors(syndromes[0]), [2, 0])
        assert np.array_equal(decoder.decode(syndromes[0]), [2, 0])
        assert np.array_equal(decoder.decode_errors_batch(syndromes), 3 * syndromes)
        assert np.array_equal(decoder.decode_batch(syndromes), 3 * syndromes)

    # a subclass must implement one of them
    class IncompleteDecoder(decoders.BatchErrorDecoder): ...

    with pytest.raises(NotImplementedError, match="must implement decode_errors"):
        IncompleteDecoder().decode(syndromes[0])
    with pytest.raises(NotImplementedError, match="must implement decode_errors_batch"):
        IncompleteDecoder().decode_batch(syndromes)
