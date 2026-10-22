# SPDX-License-Identifier: Apache-2.0

"""Logical tableaus of physical circuits, given the encoder and decoder of a code."""

from __future__ import annotations

import numpy as np
import stim


def get_logical_tableau_from_code_data(
    dimension: int,  # number of logical qubits of a QuditCode
    gauge_dimension: int,  # number of gauge qubits of a QuditCode
    encoder: stim.Tableau,
    decoder: stim.Tableau,
    physical_circuit: stim.Circuit,
    skip_validation: bool = False,
) -> stim.Tableau:
    """Identify the logical tableau implemented by the physical circuit."""
    from .encoding import restrict_tableau  # avoid circular import

    if len(encoder) != len(decoder) or len(encoder) < dimension + gauge_dimension:
        raise ValueError("Encoder and decoder have incompatible dimensions")
    identity_phys = stim.Circuit(f"I {len(encoder) - 1}")
    physical_tableau = (physical_circuit + identity_phys).to_tableau()

    # Validate the decoded physical action before restricting it.  Restriction can itself fail for
    # an invalid action, obscuring the useful public error below.
    decoded_tableau = encoder.then(physical_tableau).then(decoder)
    if not skip_validation:
        # identify sectors that address logical, gauge, and stabilizer qubits
        sector_l = slice(dimension)
        sector_g = slice(dimension, dimension + gauge_dimension)
        sector_s = slice(dimension + gauge_dimension, len(encoder))
        x2x, x2z, z2x, z2z, _, z_signs = decoded_tableau.to_numpy()

        # sanity check: stabilizers, logicals, and gauge operators should not pick up destabilizers
        ops_acquired_destabilizers = (
            np.any(z2x[:, sector_s])
            or np.any(x2x[sector_l, sector_s])
            or np.any(x2x[sector_g, sector_s])
        )

        # sanity check: gauge operators should not pick up logical factors
        gauges_acquired_logicals = (
            np.any(x2x[sector_g, sector_l])
            or np.any(x2z[sector_g, sector_l])
            or np.any(z2x[sector_g, sector_l])
            or np.any(z2z[sector_g, sector_l])
        )

        # sanity check: no stabilizers get flipped
        stabilizers_flipped = np.any(z_signs[sector_s])

        if ops_acquired_destabilizers or gauges_acquired_logicals or stabilizers_flipped:
            raise ValueError("The provided physical circuit does not implement a logical operation")

    # compute the "upper left" block of the decoded tableau that acts on all logical qubits
    logical_tableau = restrict_tableau(decoded_tableau, range(dimension))
    return logical_tableau
