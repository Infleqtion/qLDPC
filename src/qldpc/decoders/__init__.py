# SPDX-License-Identifier: Apache-2.0

"""Decoder specifications, protocols, and workflow entry points."""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from . import custom as custom
from . import dems as dems
from . import external as external
from . import sinter as sinter
from .adapters.error_decoders import (
    ErrorsToObservablesDecoder,
    ExpandedErrorDecoder,
    match_error_decoder_to_dem,
)
from .capabilities import (
    compiles_for_dem,
    is_prebuilt_decoder,
    is_prebuilt_observable_decoder,
)
from .common import (
    ObservableDecodingFallbackWarning,
    get_error_and_erasure,
    with_erasure_bits,
)
from .construction.factories import (
    DEMDecoderFactory,
    MatrixDecoderFactory,
    from_dem,
    from_matrix,
)
from .construction.resolution import (
    reject_prebuilt_decoder,
    resolve_decoder,
    resolve_observable_decoder,
)
from .construction.specs import (
    DecoderInput,
    DecoderSpec,
    DeferredDecoderInput,
    DeferredErrorDecoderInput,
    ErrorDecoderConstructor,
    ErrorDecoderInput,
    ObservableDecoderCompiler,
    ObservableDecoderConstructor,
    PcmOrDem,
)
from .custom.guf import guf
from .custom.ilp import ilp
from .custom.lookup import lookup
from .dems import DetectorErrorModelArrays
from .external.frontier import frontier
from .external.ldpc import bf, bp_lsd, bp_osd
from .external.pymatching import mwpm
from .external.relay_bp import min_sum_bp, relay_bp
from .external.tesseract import tesseract, tesseract_preset
from .protocols import (
    BatchDetailedErrorDecoder,
    BatchDetailedObservableDecoder,
    BatchErrorDecoder,
    BatchObservableDecoder,
    DetailedErrorDecoder,
    DetailedObservableDecoder,
    ErrorDecoder,
    ErrorDecodeResult,
    ObservableDecoder,
    ObservableDecodeResult,
    SupportsDecode,
    WrappedErrorDecoder,
    as_error_decoder,
    batch_decode_errors,
    supports_batch_decoding,
)
from .sinter import (
    SequentialWindowDecoder,
    SinterDecoder,
    SlidingWindowDecoder,
    SubgraphDecoder,
    TrivialDecoder,
)

__all__ = [
    "BatchDetailedErrorDecoder",
    "BatchDetailedObservableDecoder",
    "BatchErrorDecoder",
    "BatchObservableDecoder",
    "DEMDecoderFactory",
    "DecoderInput",
    "DecoderSpec",
    "DeferredDecoderInput",
    "DeferredErrorDecoderInput",
    "DetailedErrorDecoder",
    "DetailedObservableDecoder",
    "DetectorErrorModelArrays",
    "ErrorDecodeResult",
    "ErrorDecoder",
    "ErrorDecoderConstructor",
    "ErrorDecoderInput",
    "ErrorsToObservablesDecoder",
    "ExpandedErrorDecoder",
    "MatrixDecoderFactory",
    "ObservableDecodeResult",
    "ObservableDecoder",
    "ObservableDecoderCompiler",
    "ObservableDecoderConstructor",
    "ObservableDecodingFallbackWarning",
    "PcmOrDem",
    "SequentialWindowDecoder",
    "SinterDecoder",
    "SlidingWindowDecoder",
    "SubgraphDecoder",
    "SupportsDecode",
    "TrivialDecoder",
    "WrappedErrorDecoder",
    "as_error_decoder",
    "batch_decode_errors",
    "bf",
    "bp_lsd",
    "bp_osd",
    "compiles_for_dem",
    "from_dem",
    "from_matrix",
    "frontier",
    "get_error_and_erasure",
    "guf",
    "ilp",
    "is_prebuilt_decoder",
    "is_prebuilt_observable_decoder",
    "lookup",
    "match_error_decoder_to_dem",
    "min_sum_bp",
    "mwpm",
    "reject_prebuilt_decoder",
    "relay_bp",
    "resolve_decoder",
    "resolve_observable_decoder",
    "supports_batch_decoding",
    "tesseract",
    "tesseract_preset",
    "with_erasure_bits",
]
__all__.extend(_deprecations.deprecated_names(__name__))

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
