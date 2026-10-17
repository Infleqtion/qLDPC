# SPDX-License-Identifier: Apache-2.0

"""Tests for sampled logical error and discard rates."""

import numpy as np
import pytest
import stim

from qldpc import circuits, decoders


def test_logical_error_and_discard_rate() -> None:
    """Sample deterministic circuits and post-select detectors or incorrect observables."""
    circuit = stim.Circuit("M 0\nDETECTOR rec[-1]")
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        circuit,
        sinter_decoder=decoders.TrivialDecoder(),
        num_samples=1,
        post_select=[0],
    )
    assert logical_error_rate == 0
    assert discard_rate == 0

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

    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        circuit, sinter_decoder=decoders.TrivialDecoder(), num_samples=0
    )
    assert np.isnan(logical_error_rate)
    assert discard_rate == 0

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
