# SPDX-License-Identifier: Apache-2.0

"""Coercion into error decoders, batch decoding, and helpers for decoder builders."""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any, ParamSpec, TypeAlias, TypeVar

import numpy as np
import numpy.typing as npt
import stim

from qldpc.math import IntegerArray

from .dems import DetectorErrorModelArrays
from .protocols import ErrorDecoder, ObservableDecoder, SupportsDecode

PLACEHOLDER_ERROR_RATE = 1e-3  # required for some decoding methods

_PcmOrDem: TypeAlias = IntegerArray | stim.DetectorErrorModel
_Parameters = ParamSpec("_Parameters")
_Decoder = TypeVar("_Decoder", bound=ErrorDecoder)
_ErrorChannel: TypeAlias = float | npt.NDArray[np.floating] | Sequence[float] | None


####################################################################################################
# coercion into error decoders


def as_error_decoder(decoder: object, source: str = "A decoder") -> ErrorDecoder:
    """Coerce an object into an error decoder, or raise an error if it is not one.

    An ErrorDecoder is returned as is.  Another object with a decode method is wrapped in a
    WrappedErrorDecoder.  An object that predicts observable flips is rejected; this includes an
    object whose decode_returns_observables attribute is True, which declares that its decode method
    returns observable flips.

    Args:
        decoder: The object to coerce.
        source: A description of the object, which begins any error message.
    """
    predicts_observables = TypeError(
        f"{source} predicts observable flips rather than errors.  " + _OBSERVABLE_DECODER_ADVICE
    )
    if getattr(decoder, "decode_returns_observables", False):
        raise predicts_observables
    if isinstance(decoder, ErrorDecoder):
        return decoder
    if not isinstance(decoder, SupportsDecode) and callable(
        getattr(decoder, "compile_decoder_for_dem", None)
    ):
        raise TypeError(f"{source} cannot decode until it is compiled for a detector error model")
    if isinstance(decoder, SupportsDecode):
        return WrappedErrorDecoder(decoder)
    if isinstance(decoder, ObservableDecoder):
        raise predicts_observables
    raise TypeError(f"{source} must be an ErrorDecoder, or have a decode method")


class WrappedErrorDecoder(ErrorDecoder):
    """Error decoder that wraps an object whose decode method returns an inferred error.

    The wrapped object is the .decoder attribute.  Its decode method provides decode_errors, its
    decode_batch method (if any) provides decode_errors_batch, and its other attributes are readable
    from the wrapper.
    """

    def __init__(self, decoder: SupportsDecode) -> None:
        self.decoder = decoder

    def __getattr__(self, name: str) -> object:
        """Read an attribute of the wrapped object, reading decode_errors_batch as decode_batch.

        Special (dunder) attributes are not read from the wrapped object, which keeps copying and
        unpickling from recursing before the wrapped object is set.
        """
        if name == "decoder" or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self.decoder, "decode_batch" if name == "decode_errors_batch" else name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.decoder!r})"

    def decode_errors(self, syndrome: npt.NDArray[np.int_], *args: Any, **kwargs: Any) -> Any:
        """Decode an error syndrome and return an inferred error."""
        return self.decoder.decode(syndrome, *args, **kwargs)

    decode = decode_errors


_OBSERVABLE_DECODER_ADVICE = (
    "Pass an error-decoder specification such as decoders.bp_osd(...), or pass the observable decoder where"
    " one is accepted, such as to decoders.SinterDecoder"
)


####################################################################################################
# batch decoding


def batch_decode_errors(
    decoder: ErrorDecoder, syndromes: npt.NDArray[np.int_]
) -> npt.NDArray[np.int_]:
    """Decode a batch of syndromes, one per row, and return inferred errors, one per row.

    The inferred errors form a two-dimensional array even if the batch is empty.
    """
    syndromes = np.asarray(syndromes)
    if (decode_errors_batch := _get_batch_decoding_method(decoder)) is not None:
        return np.asarray(decode_errors_batch(syndromes))
    if len(syndromes) == 0:
        # decode a trivial syndrome to identify the length of an inferred error
        test_error = decoder.decode_errors(np.zeros(syndromes.shape[1], dtype=syndromes.dtype))
        return np.zeros((0, len(test_error)), dtype=np.asarray(test_error).dtype)
    return np.array([decoder.decode_errors(syndrome) for syndrome in syndromes])


def supports_batch_decoding(decoder: ErrorDecoder) -> bool:
    """Whether an error decoder has a decode_errors_batch method, or its alias decode_batch."""
    return _get_batch_decoding_method(decoder) is not None


def _get_batch_decoding_method(decoder: ErrorDecoder) -> Any:
    """The decode_errors_batch method of an error decoder, its alias decode_batch, or None."""
    return getattr(decoder, "decode_errors_batch", None) or getattr(decoder, "decode_batch", None)


####################################################################################################
# decoder builders and their options


def erasure_bit_support(
    display_name: str,
    *,
    supported: bool,
) -> Callable[[Callable[_Parameters, _Decoder]], Callable[_Parameters, _Decoder]]:
    """Declare and enforce whether a decoder builder supports an erasure bit."""

    def decorator(
        decoder_builder: Callable[_Parameters, _Decoder],
    ) -> Callable[_Parameters, _Decoder]:
        message = (
            f"The {display_name} decoder cannot signal erasure, so it does not accept the"
            " add_erasure_bit argument"
        )

        @functools.wraps(decoder_builder)
        def checked_builder(*args: _Parameters.args, **kwargs: _Parameters.kwargs) -> _Decoder:
            add_erasure_bit = bool(kwargs.get("add_erasure_bit"))
            if not supported:
                if add_erasure_bit:
                    raise ValueError(message)
                kwargs.pop("add_erasure_bit", None)

            decoder = decoder_builder(*args, **kwargs)
            if add_erasure_bit and not getattr(decoder, "has_erasure_bit", False):
                raise ValueError(message)
            return decoder

        return checked_builder

    return decorator


def to_pcm(pcm_or_dem: _PcmOrDem) -> IntegerArray:
    """Return a parity-check matrix, densifying a detector error model."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        return DetectorErrorModelArrays(pcm_or_dem).detector_flip_matrix.toarray()
    return pcm_or_dem


def get_matrix_error_channel(
    pcm_or_dem: _PcmOrDem,
    error_channel: _ErrorChannel,
    error_rate: float | None,
) -> npt.NDArray[np.floating] | None:
    """Normalize matrix error probabilities and reject explicit probabilities for a DEM."""
    if isinstance(pcm_or_dem, stim.DetectorErrorModel):
        reject_dem_error_probabilities(error_channel, error_rate)
        return None

    if error_rate is not None:
        # imported here because qldpc._deprecations.decoders imports modules that import this one
        from qldpc._deprecations.decoders import deprecate_error_rate_option

        deprecate_error_rate_option(
            {"error_channel": error_channel, "error_rate": error_rate},
            frozenset(
                {"error_rate", "error_channel"} if error_channel is not None else {"error_rate"}
            ),
        )
        error_channel = error_rate
    if error_channel is None:
        error_channel = PLACEHOLDER_ERROR_RATE

    if np.isscalar(error_channel):
        probabilities = np.full(pcm_or_dem.shape[1], error_channel, dtype=float)
    else:
        probabilities = np.asarray(error_channel, dtype=float)
    expected_shape = (pcm_or_dem.shape[1],)
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"error probabilities of shape {expected_shape} are required in error_channel, but got"
            f" {probabilities.shape}"
        )
    if np.any(~np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("error_channel probabilities must be finite and between 0 and 1")
    return probabilities


def reject_dem_error_probabilities(error_channel: object, error_rate: object) -> None:
    """Reject explicit error probabilities for a detector error model, which supplies its own."""
    options = {"error_channel": error_channel, "error_rate": error_rate}
    if specified := [f"{name}={value!r}" for name, value in options.items() if value is not None]:
        raise ValueError(
            "A detector error model supplies its own error probabilities, so"
            f" {' and '.join(specified)} cannot be specified with one.  Remove the option, as for a"
            " SinterDecoder, which always decodes detector error models.  Alternatively, pass"
            " error_channel with the detector-flip matrix of the model,"
            " decoders.DetectorErrorModelArrays(dem).detector_flip_matrix"
        )


def is_default_value(value: object, default: object) -> bool:
    """Whether an option is a plain copy of its default value."""
    if value is default:
        return True
    plain_types = (bool, int, float, str)
    return type(value) is type(default) and isinstance(value, plain_types) and value == default
