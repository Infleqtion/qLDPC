# SPDX-License-Identifier: Apache-2.0

"""Sampled logical error and discard rates for annotated circuits."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import sinter
import stim

from qldpc import decoders

from ..bookkeeping import DetectorRecord
from ._postselection import _get_postselection_masks


def get_logical_error_and_discard_rate(
    circuit_or_dem: stim.Circuit | stim.DetectorErrorModel,
    sinter_decoder: sinter.Decoder,
    num_samples: int,
    *,
    post_select: Collection[int] = (),
    post_select_observables: Collection[int] = (),
    dem_to_decode: stim.DetectorErrorModel | None = None,
) -> tuple[float, float]:
    """Compute a logical error rate and discard rate from samples of the provided circuit.

    Each logical error rate is a fraction of the (possibly post-selected) shots in which observable
    flips are predicted incorrectly by the provided decoder.

    This method is provided for convenience, but if you are doing heavy numerics you should probably
    build a sinter.Task and call sinter.collect.  In this case, circuit_or_dem should just be a
    circuit, and the sinter.Task would be built as follows::

        postselection_mask = np.zeros(circuit.num_detectors, dtype=int)
        postselection_mask[post_select] = 1
        task = sinter.Task(
            circuit=circuit,
            detector_error_model=dem_to_decode,
            postselection_mask=np.packbits(postselection_mask, bitorder="little"),
        )

    Sampling data would then be collected with::

        stats = sinter.collect(
            tasks=[task],  # or more maybe more tasks
            decoders=["custom"],
            custom_decoders={"custom": sinter_decoder},
            num_shots=num_samples,
            # other options such as num_workers=os.cpu_count() or max_errors=100
        )

    Args:
        circuit_or_dem: The circuit or detector error model we wish to sample.
        sinter_decoder: The circuit-level decoder used to predict observable flips.
        num_samples: The number of times to sample the circuit_or_dem.

    Keyword args:
        post_select: The detectors in circuit_or_dem to post-select on.
        post_select_observables: The observables to post-select on.  Any time the decoder's
            prediction for these observable flips do not agree with the actual observable flips, the
            shot is discarded rather than counting as an error.
        dem_to_decode: The detector error model to decode.  This DEM should include _all_ detectors,
            including any that are post-selected on: the decoder receives every detector bit, for
            consistency with how Sinter works.  If ``dem_to_decode is None``, this method decodes
            with the (simplified) DEM sampled from ``circuit_or_dem``.

    Returns:
        A fraction of the retained samples in which at least one observable was decoded incorrectly.
        A fraction of samples that were discarded, either by post-selection or at the request of the
        decoder, which signals that a shot should be discarded by predicting observable flips in one
        more byte than the observables of the sampled circuit require.  These are point estimates.
        Use ``sinter.collect`` when counts or confidence intervals are needed.
    """
    # identify and simplify the DEM to sample
    dem_arrays = decoders.DetectorErrorModelArrays(circuit_or_dem, simplify=True)
    dem = dem_arrays.to_dem()

    if dem_to_decode is not None:
        same_num_observables = dem_to_decode.num_observables == dem.num_observables
        same_num_detectors = dem_to_decode.num_detectors == dem.num_detectors
        if not same_num_observables or not same_num_detectors:
            raise ValueError(
                f"Incompatible detector error models."
                "\n(num_detectors, num_observables) in the DEM to sample:"
                f" {(dem.num_detectors, dem.num_observables)}"
                "\n(num_detectors, num_observables) in the DEM to decode:"
                f" {(dem_to_decode.num_detectors, dem_to_decode.num_observables)}"
            )

    # sample detector and observable flips in the circuit
    sampler = dem.compile_sampler()
    det_data, obs_data, _ = sampler.sample(shots=num_samples, bit_packed=True)

    # identify post-selection masks
    detector_record = DetectorRecord({"prep": range(circuit_or_dem.num_detectors)})
    postselection_mask, postselected_observables_mask = _get_postselection_masks(
        post_select, post_select_observables, detector_record, dem.num_observables
    )
    num_discards = 0

    # if applicable, post-select on flag detectors
    if postselection_mask is not None:
        shot_mask = ~np.any(det_data & postselection_mask, axis=1)
        det_data = det_data[shot_mask]
        obs_data = obs_data[shot_mask]
        num_discards += int(np.sum(~shot_mask))

    # decode and identify incorrectly predicted observable flips
    compiled_sinter_decoder = sinter_decoder.compile_decoder_for_dem(dem_to_decode or dem)
    predicted_flips = compiled_sinter_decoder.decode_shots_bit_packed(det_data)

    # a prediction one byte wider than the sampled observables need signals discards in that byte
    if predicted_flips.shape[1] == obs_data.shape[1] + 1:
        discarded = predicted_flips[:, -1] != 0
        obs_data = obs_data[~discarded]
        predicted_flips = predicted_flips[~discarded, :-1]
        num_discards += int(np.sum(discarded))
    elif predicted_flips.shape[1] != obs_data.shape[1]:
        raise ValueError(
            f"The decoder predicted {predicted_flips.shape[1]} bytes of observable flips per shot,"
            f" but the {dem.num_observables} observables of the sampled detector error model take"
            f" {obs_data.shape[1]} bytes, or {obs_data.shape[1] + 1} bytes with a byte added to"
            " signal discards"
        )

    incorrectly_predicted_flips = obs_data ^ predicted_flips

    # if applicable, post-select on observables
    if postselected_observables_mask is not None:
        shot_mask = ~np.any(incorrectly_predicted_flips & postselected_observables_mask, axis=1)
        incorrectly_predicted_flips = incorrectly_predicted_flips[shot_mask]
        num_discards += int(np.sum(~shot_mask))

    # compute logical error rate: fraction of shots with incorrectly predicted observable flips
    failures = np.any(incorrectly_predicted_flips, axis=1)
    logical_error_rate = np.sum(failures) / len(failures) if len(failures) else np.nan
    discard_rate = num_discards / num_samples if num_samples else 0.0
    return logical_error_rate, discard_rate
