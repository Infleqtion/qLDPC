# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _capabilities.py."""

from __future__ import annotations

import types

import galois
import numpy as np
import stim

from qldpc import decoders
from qldpc.decoders import _capabilities

_OBSERVABLE_DECODER = types.SimpleNamespace(decode_observables=lambda syndrome: np.zeros(1))
_ERROR_DECODER = types.SimpleNamespace(
    decode_errors=lambda syndrome: np.zeros(2), decode=lambda syndrome: np.zeros(2)
)
_BIT_PACKED_COMPILED_DECODER = types.SimpleNamespace(
    decode_shots_bit_packed=lambda **kwargs: np.zeros((1, 1), dtype=np.uint8)
)


def test_is_prebuilt_decoder() -> None:
    """Prebuilt decoders decode, and are not specifications, constructors, or compilers."""
    matrix = np.eye(2, dtype=int)
    prebuilt_decoders: list[object] = [
        _ERROR_DECODER,
        _OBSERVABLE_DECODER,
        _BIT_PACKED_COMPILED_DECODER,
        decoders.TrivialDecoder().compile_decoder_for_dem(stim.DetectorErrorModel()),
        decoders.bp_osd().build(matrix),
    ]
    for decoder in prebuilt_decoders:
        assert _capabilities.is_prebuilt_decoder(decoder)
    deferred_decoders: list[object] = [
        None,
        decoders.bp_osd(),
        decoders.custom.LookupDecoder,
        lambda matrix: _ERROR_DECODER,
        decoders.from_matrix(lambda matrix: _ERROR_DECODER),
        decoders.from_dem(lambda dem: _OBSERVABLE_DECODER),
        decoders.SinterDecoder(),
        decoders.TrivialDecoder(),
    ]
    for decoder in deferred_decoders:
        assert not _capabilities.is_prebuilt_decoder(decoder)


def test_is_prebuilt_observable_decoder() -> None:
    """Classify decoder inputs by explicit capabilities."""
    matrix = galois.GF2([[1, 1]])
    assert _capabilities.is_prebuilt_observable_decoder(_OBSERVABLE_DECODER)
    assert _capabilities.is_prebuilt_observable_decoder(_BIT_PACKED_COMPILED_DECODER)
    assert not _capabilities.is_prebuilt_observable_decoder(_ERROR_DECODER)
    assert not _capabilities.is_prebuilt_observable_decoder(decoders.lookup(max_weight=1))
    assert not _capabilities.is_prebuilt_observable_decoder(decoders.custom.ObservableLookupDecoder)
    sinter_decoder = decoders.TrivialDecoder()
    assert not isinstance(sinter_decoder, decoders.ObservableDecoder)
    assert not _capabilities.is_prebuilt_observable_decoder(sinter_decoder)
    compiled_sinter_decoder = sinter_decoder.compile_decoder_for_dem(stim.DetectorErrorModel())
    assert _capabilities.is_prebuilt_observable_decoder(compiled_sinter_decoder)
    assert not _capabilities.is_prebuilt_observable_decoder(decoders.relay_bp().build(matrix))
    assert _capabilities.compiles_for_dem(sinter_decoder)
    assert not _capabilities.compiles_for_dem(decoders.TrivialDecoder)
