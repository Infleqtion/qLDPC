# SPDX-License-Identifier: Apache-2.0

"""Alignment of errors inferred by error decoders with the error mechanisms of a DEM."""

from __future__ import annotations

from typing import Self, cast

import numpy as np
import numpy.typing as npt
import stim

from .._common import batch_decode_errors
from ..dems import DetectorErrorModelArrays
from ..protocols import BatchErrorDecoder, DetailedErrorDecoder, ErrorDecoder, ErrorDecodeResult

####################################################################################################
# error-mechanism alignment


class ExpandedErrorDecoder(BatchErrorDecoder):
    """Map errors inferred for a simplified DEM back to the full DEM.

    If the wrapped decoder is a DetailedErrorDecoder, this decoder is also a DetailedErrorDecoder.
    """

    def __new__(cls, decoder: object = None, *args: object, **kwargs: object) -> Self:
        if cls is ExpandedErrorDecoder and isinstance(decoder, DetailedErrorDecoder):
            return super().__new__(cast(type[Self], _DetailedExpandedErrorDecoder))
        return super().__new__(cls)

    def __init__(self, decoder: ErrorDecoder, dem: stim.DetectorErrorModel) -> None:
        self._decoder = decoder
        self.has_erasure_bit = bool(getattr(decoder, "has_erasure_bit", False))

        original_errors = DetectorErrorModelArrays.get_circuit_errors(dem)
        simplified_errors = DetectorErrorModelArrays.get_merged_circuit_errors(original_errors)
        self._num_original_errors = len(original_errors)
        signature_to_original_error_index = {
            signature: original_error_index
            for original_error_index, (_, signature) in enumerate(original_errors)
        }
        self._simplified_to_original_index = np.array(
            [signature_to_original_error_index[signature] for _, signature in simplified_errors],
            dtype=np.intp,
        )

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode one syndrome and expand the inferred error."""
        simplified_error = self._decoder.decode_errors(syndrome)
        original_error = np.zeros(
            self._num_original_errors + self.has_erasure_bit, dtype=syndrome.dtype
        )
        if self.has_erasure_bit:
            original_error[-1] = simplified_error[-1]
            simplified_error = simplified_error[:-1]
        original_error[self._simplified_to_original_index] = simplified_error
        return np.asarray(original_error, dtype=syndrome.dtype)

    def decode_errors_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of syndromes and expand the inferred errors."""
        simplified_errors = batch_decode_errors(self._decoder, syndromes)
        original_errors = np.zeros(
            (len(syndromes), self._num_original_errors + self.has_erasure_bit),
            dtype=syndromes.dtype,
        )
        if self.has_erasure_bit:
            original_errors[:, -1] = simplified_errors[:, -1]
            simplified_errors = simplified_errors[:, :-1]
        original_errors[:, self._simplified_to_original_index] = simplified_errors
        return original_errors


def match_error_decoder_to_dem(decoder: ErrorDecoder, dem: stim.DetectorErrorModel) -> ErrorDecoder:
    """Validate an error decoder's output width and align merged errors with a DEM."""
    if getattr(decoder, "_infers_decomposed_errors", False):
        raise ValueError(
            "The error decoder infers errors in the components of decomposed error mechanisms,"
            " which cannot be read as errors of the detector error model.  To predict observable"
            " flips with decomposed errors, pass a decoder specification such as"
            " decoders.mwpm(decompose_errors=True), which build a matching decoder that predicts"
            " observable flips natively"
        )
    num_erasure_bits = int(getattr(decoder, "has_erasure_bit", False))
    test_error = decoder.decode_errors(np.zeros(dem.num_detectors, dtype=int))
    num_inferred_errors = len(test_error) - num_erasure_bits
    circuit_errors = DetectorErrorModelArrays.get_circuit_errors(dem)
    num_merged_errors = len(DetectorErrorModelArrays.get_merged_circuit_errors(circuit_errors))
    if num_inferred_errors == len(circuit_errors):
        return decoder
    if num_inferred_errors == num_merged_errors:
        return ExpandedErrorDecoder(decoder, dem)
    raise ValueError(
        f"An error decoder inferred an error of length {num_inferred_errors} for a detector error"
        f" model with {len(circuit_errors)} error mechanisms ({num_merged_errors} after merging"
        " equivalent mechanisms).  If the decoder predicts observable flips rather than errors,"
        " give it a decode_observables method, and pass it where an observable decoder is accepted"
    )


class _DetailedExpandedErrorDecoder(ExpandedErrorDecoder):
    """An ExpandedErrorDecoder that forwards the diagnostics of its wrapped decoder."""

    def decode_errors_detailed(self, syndrome: npt.NDArray[np.int_]) -> ErrorDecodeResult:
        """Decode one syndrome, expand the inferred error, and keep the decoder's diagnostics."""
        result = cast(DetailedErrorDecoder, self._decoder).decode_errors_detailed(syndrome)
        num_errors = len(self._simplified_to_original_index)
        error = np.zeros(self._num_original_errors, dtype=result.error.dtype)
        error[self._simplified_to_original_index] = result.error[:num_errors]
        error = np.concatenate([error, result.error[num_errors:]])
        return ErrorDecodeResult(error, result.erasure, result.diagnostics)
