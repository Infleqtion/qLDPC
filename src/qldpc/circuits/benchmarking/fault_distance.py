# SPDX-License-Identifier: Apache-2.0

"""Exact observable-changing fault distance of an annotated noisy circuit."""

from __future__ import annotations

import galois
import numpy as np
import stim

from qldpc import codes, decoders
from qldpc.codes.distance import DistanceMethod
from qldpc.objects import Pauli


def get_fault_distance(
    circuit: stim.Circuit, *, method: DistanceMethod = "brouwer_zimmermann"
) -> int | float:
    r"""Compute the exact fault distance of a noisy circuit.

    Consider a noisy Stim circuit annotated with detectors and logical observables.  Calling
    ``circuit.detector_error_model()`` converts it into a detector error model (DEM).  The DEM lists
    fault mechanisms, their probabilities, and the detectors and observables flipped by each
    mechanism.

    ``decoders.DetectorErrorModelArrays`` collects this data into two binary flip matrices and a
    vector of fault probabilities.  Specifically, its ``detector_flip_matrix`` has one column per
    fault mechanism and one row per detector, with each row specifying which fault mechanisms flip
    that detector.  Its ``observable_flip_matrix`` similarly specifies how fault mechanisms flip
    observables.

    Denote these matrices by :math:`D` and :math:`L`, respectively.  Let :math:`e` be a binary
    vector indicating which faults have occurred.  The detector and observable flips caused by
    :math:`e` are :math:`D \cdot e` and :math:`L \cdot e`.  The circuit fault distance is the
    minimum-weight undetectable logical fault:

    .. math::

        d_{\mathrm{fault}} = \min_{D \cdot e=0,\;L \cdot e\ne0}|e|,

    where :math:`|e|` is the Hamming weight of :math:`e`, or the number of faults.

    Imagine one artificial qubit per fault mechanism, and define an auxiliary CSS code :math:`C`
    with X-check matrix :math:`H_X=D`.  Thinking of :math:`e` as the support of a Pauli-Z string,
    :math:`D \cdot e=0` means that this string commutes with all X-type stabilizers, so it is a
    logical operator of :math:`C`.  This logical operator may be trivial (a stabilizer), acting as
    the identity on the code space.

    We seek a minimum-weight logical operator :math:`e` for which additionally
    :math:`L \cdot e\ne0`.  To do so, we mod out the search space by trivial logicals for which
    :math:`L \cdot e=0`, which we can do by adding Z-type stabilizers to :math:`C` that commute with
    both its X-type stabilizers and each X string supported on a row of :math:`L`.  These conditions
    are satisfied by ``H_Z = np.vstack([D, L]).null_space()`` over GF(2).  The rows of :math:`H_Z`
    are in the null spaces of both :math:`D` and :math:`L`: precisely the undetected fault patterns
    with no observable effect, which we declare trivial errors.  The remaining nontrivial Z-logical
    operators have zero syndrome but flip an observable.

    Thus the Z-distance of :math:`C` is exactly the circuit fault distance.  The artificial Z label
    does not restrict the underlying circuit faults to physical Z errors.  The exact code distance
    calculation can be expensive for larger circuits.

    Args:
        circuit: A noisy Stim circuit annotated with detectors and logical observables.
        method: The method used to compute the Z-distance of an auxiliary fault code.

    Returns:
        The minimum number of DEM fault mechanisms causing an undetected observable flip, or np.nan
        if no such fault pattern exists.
    """
    dem_arrays = decoders.DetectorErrorModelArrays(circuit)
    detectors = galois.GF2(dem_arrays.detector_flip_matrix.toarray())
    observables = galois.GF2(dem_arrays.observable_flip_matrix.toarray())
    harmless_faults = galois.GF2(np.vstack([detectors, observables])).null_space()
    fault_code = codes.CSSCode(detectors, harmless_faults)
    return fault_code.get_distance_exact(Pauli.Z, method=method)
