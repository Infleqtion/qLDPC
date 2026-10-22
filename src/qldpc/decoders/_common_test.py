# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _common.py."""

from __future__ import annotations

import copy
import re
import types
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import decoders
from qldpc._deprecations import decoders as deprecated_decoders
from qldpc.decoders import _common


def test_error_decoder_coercion() -> None:
    """Objects with a decode method are wrapped to provide decode_errors."""
    syndromes = np.eye(2, dtype=int)

    class BareDecoder:
        scale = 2

        def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return self.scale * syndrome

    class BareBatchDecoder(BareDecoder):
        def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return 3 * syndromes

    # a wrapped decoder decodes with, and reads attributes of, the object that it wraps
    bare_decoder = BareDecoder()
    decoder: Any = _common.as_error_decoder(bare_decoder)
    assert decoder.decoder is bare_decoder
    assert repr(decoder) == f"WrappedErrorDecoder({bare_decoder!r})"
    assert np.array_equal(decoder.decode_errors(syndromes[0]), [2, 0])
    assert np.array_equal(decoder.decode(syndromes[0]), [2, 0])
    bare_decoder.scale = 4
    assert decoder.scale == 4
    assert not hasattr(decoder, "decode_errors_batch")
    assert not hasattr(decoder, "__missing_attribute__")
    assert copy.copy(decoder).decoder is bare_decoder

    # batch decoding methods are provided if the wrapped object has them
    decoder = _common.as_error_decoder(BareBatchDecoder())
    assert np.array_equal(decoder.decode_errors_batch(syndromes), 3 * syndromes)
    assert _common.supports_batch_decoding(decoder)
    assert np.array_equal(_common.batch_decode_errors(decoder, syndromes), 3 * syndromes)

    # an error decoder is returned as is, and its native batch method is used
    lookup_decoder = decoders.custom.LookupDecoder(np.eye(2, dtype=int), max_weight=1)
    assert _common.as_error_decoder(lookup_decoder) is lookup_decoder
    assert _common.supports_batch_decoding(lookup_decoder)
    assert np.array_equal(_common.batch_decode_errors(lookup_decoder, syndromes), syndromes)

    # an error decoder without a batch method decodes batches one syndrome at a time
    class SingleSyndromeDecoder(decoders.ErrorDecoder):
        def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return np.concatenate([syndrome, syndrome])

    error_decoder = SingleSyndromeDecoder()
    assert not _common.supports_batch_decoding(error_decoder)
    batch = _common.batch_decode_errors(error_decoder, syndromes)
    assert np.array_equal(batch, np.hstack([syndromes, syndromes]))
    assert _common.batch_decode_errors(error_decoder, syndromes[:0]).shape == (0, 4)

    # objects that predict observable flips, or that do not decode, are rejected
    observable_decoders = [
        types.SimpleNamespace(decode_observables=lambda syndrome: syndrome),
        types.SimpleNamespace(decode=lambda syndrome: syndrome, decode_returns_observables=True),
    ]
    for observable_decoder in observable_decoders:
        with pytest.raises(TypeError, match="observable flips rather than errors"):
            _common.as_error_decoder(observable_decoder)
    with pytest.raises(TypeError, match="cannot decode until it is compiled"):
        _common.as_error_decoder(decoders.SinterDecoder())
    with pytest.raises(TypeError, match="must be an ErrorDecoder, or have a decode method"):
        _common.as_error_decoder(object())


def test_erasure_bit_support_decorator() -> None:
    """Erasure support declarations enforce capabilities and use an explicit display name."""

    class NoErasureDecoder(decoders.ErrorDecoder):
        def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
            return syndrome

    @_common.erasure_bit_support("Friendly Name", supported=True)
    def unusually_named_builder(
        matrix: npt.NDArray[np.int_], *, add_erasure_bit: bool = False
    ) -> decoders.ErrorDecoder:
        del matrix, add_erasure_bit
        return NoErasureDecoder()

    with pytest.raises(ValueError, match="The Friendly Name decoder cannot signal erasure"):
        unusually_named_builder(np.eye(1, dtype=int), add_erasure_bit=True)

    @_common.erasure_bit_support("unsupported", supported=False)
    def unsupported_builder(
        matrix: npt.NDArray[np.int_], *, add_erasure_bit: bool = False
    ) -> decoders.ErrorDecoder:
        del matrix, add_erasure_bit
        return NoErasureDecoder()

    decoder = unsupported_builder(np.eye(1, dtype=int), add_erasure_bit=False)
    assert np.array_equal(decoder.decode_errors(np.array([1])), [1])
    with pytest.raises(ValueError, match="The unsupported decoder cannot signal erasure"):
        unsupported_builder(np.eye(1, dtype=int), add_erasure_bit=True)


def test_to_pcm() -> None:
    """Matrices pass through unchanged and DEM detector matrices are densified."""
    matrix = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int32)
    assert _common.to_pcm(matrix) is matrix

    dem = decoders.DetectorErrorModelArrays.from_arrays(matrix, None, 0.1).to_dem()
    assert np.array_equal(_common.to_pcm(dem), matrix)


def test_get_matrix_error_channel() -> None:
    """Matrix probabilities are normalized and DEM-owned probabilities are protected."""
    matrix = np.eye(2, dtype=int)
    default_channel = _common.get_matrix_error_channel(matrix, None, None)
    assert default_channel is not None
    assert np.array_equal(
        default_channel,
        [_common.PLACEHOLDER_ERROR_RATE] * 2,
    )
    channel = np.array([0.1, 0.2])
    normalized_channel = _common.get_matrix_error_channel(matrix, channel, None)
    assert normalized_channel is not None
    assert np.array_equal(normalized_channel, channel)

    with pytest.warns(DeprecationWarning, match="error_rate=0.3.*error_channel=0.3") as caught:
        deprecated_channel = _common.get_matrix_error_channel(matrix, None, 0.3)
    assert deprecated_channel is not None
    assert np.array_equal(deprecated_channel, [0.3, 0.3])
    assert caught[0].filename == __file__

    with pytest.raises(ValueError, match="cannot both be specified"):
        _common.get_matrix_error_channel(matrix, 0.2, 0.3)
    with pytest.raises(ValueError, match="error probabilities of shape"):
        _common.get_matrix_error_channel(matrix, [0.1], None)
    for invalid_channel in ([np.nan, 0.2], [-0.1, 0.2], [0.1, 1.1]):
        with pytest.raises(ValueError, match="finite and between 0 and 1"):
            _common.get_matrix_error_channel(matrix, invalid_channel, None)

    dem = stim.DetectorErrorModel("error(0.1) D0")
    assert _common.get_matrix_error_channel(dem, None, None) is None
    for error_channel, error_rate, specified in (
        (None, 0.2, "error_rate=0.2"),
        (0.2, None, "error_channel=0.2"),
        (0.2, 0.3, "error_channel=0.2 and error_rate=0.3"),
    ):
        with pytest.raises(ValueError, match=f"so {re.escape(specified)} cannot"):
            _common.get_matrix_error_channel(dem, error_channel, error_rate)


def test_dem_error_probabilities_through_public_paths() -> None:
    """Explicit probabilities for a DEM fail with migration advice through every entry point."""
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 D1\nerror(0.1) D1")
    match = r"supplies its own error probabilities.*detector_flip_matrix"
    with pytest.raises(ValueError, match=match):
        decoders.bp_osd(error_channel=0.1).build(dem)
    with pytest.raises(ValueError, match=match):
        decoders.SinterDecoder(decoder=decoders.bf(error_channel=0.1)).compile_decoder_for_dem(dem)
    with pytest.warns(DeprecationWarning, match="free-form decoder options"):
        legacy_sinter_decoder = decoders.SinterDecoder(error_rate=0.1)
    with pytest.raises(ValueError, match=match):
        legacy_sinter_decoder.compile_decoder_for_dem(dem)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match=match):
        deprecated_decoders.get_decoder(dem, with_BP_LSD=True, error_rate=0.1)

    # the suggested alternative decodes the detector-flip matrix with the given probabilities
    matrix = decoders.DetectorErrorModelArrays(dem).detector_flip_matrix
    decoder: Any = decoders.bp_osd(error_channel=0.3).build(matrix)
    assert np.array_equal(decoder.error_channel, [0.3, 0.3, 0.3])


def test_is_default_value() -> None:
    """Only the default itself, or a plain copy of it of the same type, is a default value."""
    default = object()
    assert _common.is_default_value(default, default)
    assert _common.is_default_value(float("1"), 1.0)
    assert not _common.is_default_value(1, 1.0)
    assert not _common.is_default_value(object(), default)
