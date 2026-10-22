# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _dem_matching.py."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc.decoders.adapters import _dem_matching


class _FixedDecoder(decoders.ErrorDecoder):
    def __init__(self, output: npt.ArrayLike, *, has_erasure_bit: bool = False) -> None:
        self.output = np.asarray(output)
        self.has_erasure_bit = has_erasure_bit

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return self.output.copy()


class _DetailedFixedDecoder(_FixedDecoder):
    def decode_errors_detailed(self, syndrome: npt.NDArray[np.int_]) -> decoders.ErrorDecodeResult:
        return decoders.ErrorDecodeResult(self.output, bool(self.output[-1]), {"fixed": 1})


def test_expanded_error_decoder() -> None:
    """Merged errors are expanded to the original DEM width, preserving erasure bits."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0 L0
        error(0.1) D1
    """)
    syndromes = np.array([[1, 0], [0, 1]], dtype=int)

    for add_erasure_bit in [False, True]:
        merging_decoder = decoders.custom.GUFDecoder(
            decoders.DetectorErrorModelArrays(dem).detector_flip_matrix.toarray(),
            add_erasure_bit=add_erasure_bit,
        )
        decoder = _dem_matching.ExpandedErrorDecoder(merging_decoder, dem)
        errors = decoder.decode_errors_batch(syndromes)
        assert errors.shape == (2, 3 + add_erasure_bit)
        assert np.array_equal(errors, [decoder.decode_errors(syndrome) for syndrome in syndromes])
        assert np.array_equal(errors[:, [0, 1]].sum(axis=1), [1, 0])
        assert decoder.decode_errors_batch(syndromes[:0]).shape == (0, 3 + add_erasure_bit)

    # a matching decoder merges equivalent mechanisms, and decodes in batches
    matching_decoder = _dem_matching.match_error_decoder_to_dem(decoders.mwpm().build(dem), dem)
    assert isinstance(matching_decoder, _dem_matching.ExpandedErrorDecoder)
    assert matching_decoder.decode_errors_batch(syndromes).shape == (2, 3)

    # the diagnostics of a detailed decoder are kept, and its error is expanded
    detailed = _dem_matching.ExpandedErrorDecoder(_DetailedFixedDecoder([1, 0, 1]), dem)
    assert isinstance(detailed, decoders.DetailedErrorDecoder)
    result = detailed.decode_errors_detailed(syndromes[0])
    assert result.error.tolist() == [0, 1, 0, 1]
    assert result.erasure
    assert result.diagnostics == {"fixed": 1}


def test_match_error_decoder_to_dem() -> None:
    """Output validation returns, expands, or rejects a decoder as appropriate."""
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0 L0
        error(0.1) D1
    """)
    full_width = _FixedDecoder([0, 0, 0])
    assert _dem_matching.match_error_decoder_to_dem(full_width, dem) is full_width

    merged = _dem_matching.match_error_decoder_to_dem(_FixedDecoder([0, 0]), dem)
    assert isinstance(merged, _dem_matching.ExpandedErrorDecoder)

    with pytest.raises(ValueError, match="inferred an error of length 1"):
        _dem_matching.match_error_decoder_to_dem(_FixedDecoder([0]), dem)

    decomposed = _FixedDecoder([0, 0])
    decomposed._infers_decomposed_errors = True  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="components of decomposed error mechanisms"):
        _dem_matching.match_error_decoder_to_dem(decomposed, dem)
