# SPDX-License-Identifier: Apache-2.0

"""Protocols for error and observable decoders, and their detailed decoding results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

####################################################################################################
# detailed decode results


@dataclass(frozen=True, eq=False)
class ErrorDecodeResult:
    """Detailed result of decoding a syndrome to an inferred error.

    ``error`` is the array that ``decode_errors`` returns, which ends with an erasure flag only if
    erasure signaling is enabled.  ``erasure`` reports whether the decoder signals erasure, whether
    or not erasure signaling is enabled.  ``diagnostics`` contains explicitly named decoder-specific
    results, not a cross-decoder confidence score.
    """

    error: npt.NDArray[np.int_]
    erasure: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class ObservableDecodeResult:
    """Detailed result of decoding a syndrome to predicted observable flips.

    ``observable_flips`` is the array that ``decode_observables`` returns, which ends with an
    erasure flag only if erasure signaling is enabled.  ``erasure`` reports whether the decoder
    signals erasure, whether or not erasure signaling is enabled.  ``diagnostics`` contains
    explicitly named decoder-specific results, not a cross-decoder confidence score.
    """

    observable_flips: npt.NDArray[np.int_]
    erasure: bool = False
    diagnostics: dict[str, object] = field(default_factory=dict)


####################################################################################################
# decoder protocols


@runtime_checkable
class ErrorDecoder(Protocol):
    """Protocol for a decoder that maps a syndrome to an inferred error.

    An error decoder has a ``decode_errors`` method, and a ``decode`` method that is an alias for
    it.  A subclass of ErrorDecoder may implement either one, and inherits the other.

    Methods of qLDPC that accept an error decoder also accept any object whose ``decode`` method
    returns an inferred error, such as a decoder from the ldpc package, and wrap it to provide
    ``decode_errors``.
    """

    def decode_errors(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error."""
        if type(self).decode is not ErrorDecoder.decode:
            return self.decode(syndrome)
        raise NotImplementedError(f"{type(self).__name__} must implement decode_errors or decode")

    def decode(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return an inferred error (alias for decode_errors)."""
        return self.decode_errors(syndrome)


@runtime_checkable
class BatchErrorDecoder(ErrorDecoder, Protocol):
    """Protocol for an error decoder that can decode in batches.

    A batch error decoder has a ``decode_errors_batch`` method, and a ``decode_batch`` method that
    is an alias for it.  A subclass of BatchErrorDecoder may implement either one, and inherits the
    other.
    """

    def decode_errors_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes, one per row, and return inferred errors."""
        if type(self).decode_batch is not BatchErrorDecoder.decode_batch:
            return self.decode_batch(syndromes)
        raise NotImplementedError(
            f"{type(self).__name__} must implement decode_errors_batch or decode_batch"
        )

    def decode_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes, one per row (alias for decode_errors_batch)."""
        return self.decode_errors_batch(syndromes)


@runtime_checkable
class ObservableDecoder(Protocol):
    """Protocol for a decoder that maps a syndrome to predicted observable flips.

    An observable decoder is built for a specific set of observables, such as those of a detector
    error model, and predicts which of them an error with the given syndrome flips.  If an
    observable decoder signals erasure (``has_erasure_bit = True``), it appends an erasure flag to
    each prediction.
    """

    def decode_observables(self, syndrome: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode an error syndrome and return predicted observable flips."""


@runtime_checkable
class BatchObservableDecoder(ObservableDecoder, Protocol):
    """Protocol for an observable decoder that can decode in batches."""

    def decode_observables_batch(self, syndromes: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        """Decode a batch of error syndromes, one per row, and return predicted observable flips."""


####################################################################################################
# detailed decoder protocols


@runtime_checkable
class DetailedErrorDecoder(ErrorDecoder, Protocol):
    """Optional protocol for an error decoder that returns per-shot diagnostics."""

    def decode_errors_detailed(self, syndrome: npt.NDArray[np.int_]) -> ErrorDecodeResult:
        """Decode one syndrome and return the error, erasure flag, and diagnostics."""


@runtime_checkable
class BatchDetailedErrorDecoder(DetailedErrorDecoder, Protocol):
    """Optional protocol for an error decoder that returns per-shot diagnostics in batches."""

    def decode_errors_detailed_batch(
        self, syndromes: npt.NDArray[np.int_]
    ) -> tuple[ErrorDecodeResult, ...]:
        """Decode a batch and return one detailed result per syndrome."""


@runtime_checkable
class DetailedObservableDecoder(ObservableDecoder, Protocol):
    """Optional protocol for an observable decoder that returns per-shot diagnostics."""

    def decode_observables_detailed(self, syndrome: npt.NDArray[np.int_]) -> ObservableDecodeResult:
        """Decode one syndrome and return observable flips, erasure, and diagnostics."""


@runtime_checkable
class BatchDetailedObservableDecoder(DetailedObservableDecoder, Protocol):
    """Optional protocol for an observable decoder that returns per-shot diagnostics in batches."""

    def decode_observables_detailed_batch(
        self, syndromes: npt.NDArray[np.int_]
    ) -> tuple[ObservableDecodeResult, ...]:
        """Decode a batch and return one detailed result per syndrome."""


@runtime_checkable
class SupportsDecode(Protocol):
    """Protocol for an object whose decode method returns an inferred error.

    Such an object, for example a decoder from the ldpc package, is accepted wherever an error
    decoder is, and is wrapped to provide the decode_errors method of an ErrorDecoder.
    """

    def decode(self, syndrome: Any, /) -> Any:
        """Decode an error syndrome and return an inferred error."""
