# SPDX-License-Identifier: Apache-2.0

"""Tests for exact fault distance and sampled logical error rates."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import circuits, decoders
from qldpc.circuits.benchmarking import common
from qldpc.codes.distance import DistanceMethod


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "brute_force"])
@pytest.mark.parametrize("measurement, expected_distance", [(-7, 3), (-4, 4)])
def test_fault_distance(measurement: int, expected_distance: int, method: DistanceMethod) -> None:
    """Undetected faults flip whole blocks of three or four qubits; only one block is logical."""
    circuit = stim.Circuit("""
        R 0 1 2 3 4 5 6
        X_ERROR(0.01) 0 1 2 3 4 5 6
        M 0 1 2 3 4 5 6
        DETECTOR rec[-7] rec[-6]
        DETECTOR rec[-6] rec[-5]
        DETECTOR rec[-4] rec[-3]
        DETECTOR rec[-3] rec[-2]
        DETECTOR rec[-2] rec[-1]
        DETECTOR rec[-4] rec[-2]
    """)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(measurement)], 0)
    assert circuits.get_fault_distance(circuit, method=method) == expected_distance


def test_fault_distance_without_logical_faults() -> None:
    """No undetected observable flip has the code distance convention np.nan."""
    circuit = stim.Circuit("R 0\nX_ERROR(0.01) 0\nM 0\nDETECTOR rec[-1]")
    assert np.isnan(circuits.get_fault_distance(circuit))


def test_logical_error_and_discard_rate() -> None:
    """Sample a noiseless circuit and post-select detectors or incorrect observables."""
    circuit = stim.Circuit("R 0\nM 0\nDETECTOR rec[-1]")

    # bypass sinter to compute logical error rates
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        circuit,
        sinter_decoder=decoders.TrivialDecoder(),
        num_samples=1,
        post_select=range(circuit.num_detectors),
    )
    assert logical_error_rate == 0
    assert discard_rate == 0

    # post-selecting on a detector that always fires discards every shot, also for arrays
    dem = stim.DetectorErrorModel("error(1) D0 L0")
    for post_select in ([0], np.array([0])):
        logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
            dem, sinter_decoder=decoders.TrivialDecoder(), num_samples=2, post_select=post_select
        )
        assert np.isnan(logical_error_rate)
        assert discard_rate == 1

    # post-select on observables
    dem = circuit.detector_error_model()
    dem.append("error", [1], [stim.DemTarget.logical_observable_id(circuit.num_observables)])
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        dem,
        sinter_decoder=decoders.TrivialDecoder(),
        num_samples=1,
        post_select_observables=[circuit.num_observables],
    )
    assert np.isnan(logical_error_rate)
    assert discard_rate == 1

    # sampling nothing reports no rates rather than dividing by zero
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        circuit, sinter_decoder=decoders.TrivialDecoder(), num_samples=0
    )
    assert np.isnan(logical_error_rate)
    assert discard_rate == 0

    # incompatible DEMs for sampling and decoding
    with pytest.raises(ValueError, match="Incompatible detector error models"):
        circuits.get_logical_error_and_discard_rate(
            circuit,
            sinter_decoder=decoders.TrivialDecoder(),
            num_samples=1,
            dem_to_decode=stim.DetectorErrorModel(),
        )


def test_erasure_counts_as_a_discard() -> None:
    """A decoder that signals erasure has its erased shots discarded, not counted as errors."""
    # seven independent qubits, so an erasure bit does not fit in the last byte of the observables
    num_qubits = 7
    circuit = stim.Circuit()
    for qubit in range(num_qubits):
        circuit.append("X_ERROR", [qubit], 0.2)
        circuit.append("M", [qubit])
    for index in range(num_qubits):
        circuit.append("DETECTOR", [stim.target_rec(-num_qubits + index)])
        circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-num_qubits + index)], [index])

    # every error of weight one is decoded exactly, and every heavier error is erased, so the
    # logical error rate is exactly zero and the discard rate is 1 - 0.8**7 - 7 * 0.2 * 0.8**6
    decoder = decoders.SinterDecoder(decoder=decoders.lookup(max_weight=1, add_erasure_bit=True))
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        circuit, sinter_decoder=decoder, num_samples=1000
    )
    assert logical_error_rate == 0
    assert 0.30 < discard_rate < 0.55

    # a prediction that is neither as wide as the observables nor one byte wider cannot be compared
    class WideDecoder(decoders.TrivialDecoder):
        """A decoder whose predictions are two bytes too wide."""

        def compile_decoder_for_dem(
            self, dem: stim.DetectorErrorModel
        ) -> decoders.sinter.CompiledSinterDecoder:
            compiled = super().compile_decoder_for_dem(dem)
            assert isinstance(compiled, decoders.sinter.CompiledTrivialDecoder)
            compiled.packed_observable_size += 2
            return compiled

    with pytest.raises(ValueError, match="bytes of observable flips per shot"):
        circuits.get_logical_error_and_discard_rate(
            circuit, sinter_decoder=WideDecoder(), num_samples=2
        )


def test_postselection_masks() -> None:
    """Automatic selection chooses flags, while explicit out-of-range indices are rejected."""
    record = circuits.DetectorRecord({"prep": [0], "flags": [1]})
    detector_mask, observable_mask = common._get_postselection_masks(True, (), record, 0)
    assert detector_mask is not None
    assert np.array_equal(detector_mask, [2])
    assert observable_mask is None

    detector_mask, observable_mask = common._get_postselection_masks(False, [-1], record, 3)
    assert detector_mask is None
    assert observable_mask is not None
    assert np.array_equal(observable_mask, [4])

    with pytest.raises(ValueError, match="2 selectable prep/flag detectors"):
        common._get_postselection_masks([2], (), record, 0)
    with pytest.raises(ValueError, match="Observable post-selection"):
        common._get_postselection_masks((), [1], record, 1)
