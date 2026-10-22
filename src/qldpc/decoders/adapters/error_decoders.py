# SPDX-License-Identifier: Apache-2.0

"""Adapters that convert errors inferred by error decoders into observable flips."""

from __future__ import annotations

from typing import Self, cast

import numpy as np
import numpy.typing as npt
import stim

from .._common import batch_decode_errors
from ..dems import DetectorErrorModelArrays
from ..protocols import (
    DetailedErrorDecoder,
    ErrorDecoder,
    ObservableDecoder,
    ObservableDecodeResult,
)
from ._dem_matching import match_error_decoder_to_dem

####################################################################################################
# observable prediction from inferred errors


class ErrorsToObservablesDecoder(ObservableDecoder):
    """Convert errors inferred by an error decoder into binary DEM observable flips.

    If the error decoder is a DetailedErrorDecoder, this decoder is a DetailedObservableDecoder that
    forwards the erasure flag and diagnostics of each detailed error result.
    """

    def __new__(cls, error_decoder: object = None, *args: object, **kwargs: object) -> Self:
        if cls is ErrorsToObservablesDecoder and isinstance(error_decoder, DetailedErrorDecoder):
            return super().__new__(cast(type[Self], _DetailedErrorsToObservablesDecoder))
        return super().__new__(cls)

    def __init__(self, error_decoder: ErrorDecoder, dem: stim.DetectorErrorModel) -> None:
        self.error_decoder = error_decoder
        self._aligned_error_decoder = match_error_decoder_to_dem(error_decoder, dem)
        self.has_erasure_bit = bool(getattr(error_decoder, "has_erasure_bit", False))
        self.observable_flip_matrix = DetectorErrorModelArrays(
            dem, simplify=False
        ).observable_flip_matrix

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode one syndrome to predicted observable flips."""
        return self.decode_observables_batch(np.asarray(syndrome)[None, :])[0]

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes to predicted observable flips."""
        errors = batch_decode_errors(self._aligned_error_decoder, syndromes)
        erasure_bits = errors[:, -1:] if self.has_erasure_bit else errors[:, :0]
        errors = errors[:, : errors.shape[1] - erasure_bits.shape[1]]
        flips = np.asarray(errors @ self.observable_flip_matrix.T) % 2
        return np.hstack([flips, erasure_bits]).astype(np.uint8)


class _DetailedErrorsToObservablesDecoder(ErrorsToObservablesDecoder):
    """An ErrorsToObservablesDecoder that forwards the diagnostics of its error decoder."""

    def decode_observables_detailed(self, syndrome: npt.NDArray[np.int_]) -> ObservableDecodeResult:
        """Decode one syndrome to observable flips, with the error decoder's diagnostics."""
        error_decoder = cast(DetailedErrorDecoder, self._aligned_error_decoder)
        result = error_decoder.decode_errors_detailed(syndrome)
        num_errors = self.observable_flip_matrix.shape[1]
        flips = np.asarray(result.error[:num_errors] @ self.observable_flip_matrix.T) % 2
        prediction = np.hstack([flips, result.error[num_errors:]])
        return ObservableDecodeResult(prediction, result.erasure, result.diagnostics)
