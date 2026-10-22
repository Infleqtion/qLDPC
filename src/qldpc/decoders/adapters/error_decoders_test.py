# SPDX-License-Identifier: Apache-2.0

"""Tests for error-decoder adapters."""

from __future__ import annotations

import pickle

import numpy as np
import numpy.typing as npt
import stim

from qldpc import decoders
from qldpc.decoders.adapters import error_decoders


class _FixedDecoder(decoders.ErrorDecoder):
    def __init__(self, output: npt.ArrayLike, *, has_erasure_bit: bool = False) -> None:
        self.output = np.asarray(output)
        self.has_erasure_bit = has_erasure_bit

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return self.output.copy()


class _DetailedFixedDecoder(_FixedDecoder):
    def decode_errors_detailed(self, syndrome: npt.NDArray[np.int_]) -> decoders.ErrorDecodeResult:
        return decoders.ErrorDecodeResult(self.output, bool(self.output[-1]), {"fixed": 1})


def test_errors_to_observables_decoder() -> None:
    """Inferred errors are aligned with a DEM and projected onto its observables."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0 L0
        error(0.1) D1 L1
    """)
    decoder = error_decoders.ErrorsToObservablesDecoder(_FixedDecoder([1, 0]), dem)
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)
    assert np.array_equal(decoder.decode_observables(syndromes[0]), [1, 0])
    assert np.array_equal(decoder.decode_observables_batch(syndromes), [[1, 0], [1, 0]])

    erasing = error_decoders.ErrorsToObservablesDecoder(
        _FixedDecoder([0, 1, 1], has_erasure_bit=True), dem
    )
    assert erasing.has_erasure_bit
    assert np.array_equal(erasing.decode_observables(np.array([0, 1])), [0, 1, 1])
    assert not isinstance(erasing, decoders.DetailedObservableDecoder)

    # the diagnostics of a detailed error decoder are forwarded through merged-error expansion
    detailed = error_decoders.ErrorsToObservablesDecoder(
        _DetailedFixedDecoder([1, 0, 1], has_erasure_bit=True), dem
    )
    assert isinstance(detailed, decoders.DetailedObservableDecoder)
    result = detailed.decode_observables_detailed(syndromes[0])
    assert result.observable_flips.tolist() == [1, 0, 1]
    assert result.erasure
    assert result.diagnostics == {"fixed": 1}
    assert type(pickle.loads(pickle.dumps(detailed))) is type(detailed)  # noqa: S301
