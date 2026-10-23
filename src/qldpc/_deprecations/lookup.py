# SPDX-License-Identifier: Apache-2.0

"""Deprecated lookup decoders whose decode methods predict observable flips.

LookupDecoder(..., predict_observable_flips=True) and WeightedLookupDecoder(...,
predict_observable_flips=True) build the observable lookup decoders of this module, which add the
interface of qLDPC 0.4.1 to the observable lookup decoders that replace them: their decode methods
are aliases of the decode_observables methods.  This module imports qldpc.decoders when it is
imported, so qLDPC modules import it only within the functions that use it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import stim

from qldpc import _deprecations
from qldpc._utils import get_external_caller_stacklevel
from qldpc.decoders.custom.lookup import (
    ObservableLookupDecoder,
    WeightedLookupDecoder,
    WeightedObservableLookupDecoder,
)

from .decoders import _MATRIX_OBSERVABLE_LOOKUP_ADVICE

if TYPE_CHECKING:
    from qldpc.math import IntegerArray

####################################################################################################
# deprecated observable lookup decoders


class LegacyObservableLookupDecoder(ObservableLookupDecoder):
    """An ObservableLookupDecoder with the decode methods of qLDPC 0.4.1.

    LookupDecoder(..., predict_observable_flips=True) builds this decoder.  Its decode,
    decode_errors, decode_batch, and decode_errors_batch methods predict observable flips.
    """

    predict_observable_flips = True
    decode_returns_observables = True

    decode = decode_errors = ObservableLookupDecoder.decode_observables
    decode_batch = decode_errors_batch = ObservableLookupDecoder.decode_observables_batch


class LegacyWeightedObservableLookupDecoder(WeightedObservableLookupDecoder):
    """A WeightedObservableLookupDecoder with the decode methods of qLDPC 0.4.1.

    WeightedLookupDecoder(..., predict_observable_flips=True) builds this decoder.  Its decode,
    decode_errors, decode_batch, and decode_errors_batch methods predict observable flips, and take
    the penalty_func argument of decode_observables.
    """

    predict_observable_flips = True
    decode_returns_observables = True

    decode = decode_errors = WeightedObservableLookupDecoder.decode_observables
    decode_batch = decode_errors_batch = WeightedObservableLookupDecoder.decode_observables_batch


def become_legacy_observable_decoder(
    decoder: object,
    pcm_or_dem: IntegerArray | stim.DetectorErrorModel,
    max_weight: int,
    **options: object,
) -> None:
    """Warn about predict_observable_flips=True, and initialize a decoder as one that uses it.

    The decoder, a LookupDecoder that is being initialized, becomes a
    LegacyWeightedObservableLookupDecoder if it is a WeightedLookupDecoder, and a
    LegacyObservableLookupDecoder otherwise, which is initialized with the given arguments.
    """
    legacy_class: type[ObservableLookupDecoder | WeightedObservableLookupDecoder]
    if isinstance(decoder, WeightedLookupDecoder):
        legacy_class = LegacyWeightedObservableLookupDecoder
        advice = (
            "use qldpc.decoders.custom.lookup.WeightedObservableLookupDecoder(...)"
            ".decode_observables(syndrome, penalty_func) instead"
        )
    else:
        legacy_class = LegacyObservableLookupDecoder
        advice = (
            "use decoders.lookup(max_weight=...).build_observable_decoder(dem) instead"
            if isinstance(pcm_or_dem, stim.DetectorErrorModel)
            else _MATRIX_OBSERVABLE_LOOKUP_ADVICE
        )
    _deprecations.warn_deprecated(
        f"predict_observable_flips=True is deprecated; {advice}",
        stacklevel=get_external_caller_stacklevel(),
    )
    decoder.__class__ = legacy_class
    # the arguments are those of a LookupDecoder, which an observable lookup decoder also accepts
    initialize = cast("Callable[..., None]", legacy_class.__init__)
    initialize(decoder, pcm_or_dem, max_weight, **options)
