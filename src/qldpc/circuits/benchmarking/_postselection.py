# SPDX-License-Identifier: Apache-2.0

"""Post-selection masks for sampling and decoding circuits with sinter."""

from __future__ import annotations

from collections.abc import Collection

import numpy as np
import numpy.typing as npt

from ..bookkeeping import DetectorRecord

####################################################################################################
# post-selection masks


def get_postselection_masks(
    post_select: bool | Collection[int],
    post_select_observables: Collection[int],
    detector_record: DetectorRecord,
    num_observables: int,
) -> tuple[npt.NDArray[np.uint8] | None, npt.NDArray[np.uint8] | None]:
    """Build the bit-packed detector and observable post-selection masks for sinter."""
    if isinstance(post_select, bool):
        post_select = detector_record.get_events("flags") if post_select else ()
    post_select = list(post_select)
    post_select_observables = list(post_select_observables)

    if not post_select:
        postselection_mask = None
    else:
        num_prep = len(detector_record.get_events("prep"))
        num_flags = len(detector_record.get_events("flags"))
        num_detectors = num_prep + num_flags
        if not all(-num_detectors <= dd < num_detectors for dd in post_select):
            raise ValueError(
                f"The detector record exposes {num_detectors} selectable prep/flag detectors "
                f"(out of {detector_record.num_events} total), so post-selection accepts indices "
                f"in the range [-{num_detectors}, {num_detectors});"
                f" requested: {post_select}"
            )
        post_select = [dd % num_detectors for dd in post_select]
        postselection_array = np.zeros(detector_record.num_events, dtype=int)
        postselection_array[post_select] = 1
        postselection_mask = np.packbits(postselection_array, bitorder="little")

    if not post_select_observables:
        postselected_observables_mask = None
    else:
        if not all(-num_observables <= oo < num_observables for oo in post_select_observables):
            raise ValueError(
                f"Observable post-selection accepts indices in the range [-{num_observables},"
                f" {num_observables}); requested: {post_select_observables}"
            )
        postselection_array = np.zeros(num_observables, dtype=int)
        postselection_array[[oo % num_observables for oo in post_select_observables]] = 1
        postselected_observables_mask = np.packbits(postselection_array, bitorder="little")

    return postselection_mask, postselected_observables_mask
