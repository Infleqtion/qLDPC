# SPDX-License-Identifier: Apache-2.0

"""Tests for the benchmarking package's public exports."""

from typing import get_type_hints

from qldpc import circuits
from qldpc.circuits import benchmarking
from qldpc.circuits.benchmarking import (
    fault_distance,
    logical_error_rates,
    state_prep,
    state_prep_spread,
)
from qldpc.codes.distance import DistanceMethod


def test_benchmarking_exports() -> None:
    """The package and circuit-root paths resolve to the same defining objects."""
    exports = {
        "get_fault_distance": fault_distance.get_fault_distance,
        "get_logical_error_and_discard_rate": logical_error_rates.get_logical_error_and_discard_rate,
        "get_state_prep_diagnostic_circuit": state_prep.get_state_prep_diagnostic_circuit,
        "get_state_prep_diagnostic_tasks": state_prep.get_state_prep_diagnostic_tasks,
        "certify_state_prep_spread": state_prep_spread.certify_state_prep_spread,
        "StatePrepSpreadCertificate": state_prep_spread.StatePrepSpreadCertificate,
        "StatePrepSpreadCounterexample": state_prep_spread.StatePrepSpreadCounterexample,
        "StatePrepSpreadStatistics": state_prep_spread.StatePrepSpreadStatistics,
    }
    assert set(benchmarking.__all__) == set(exports)
    for name, obj in exports.items():
        assert getattr(benchmarking, name) is obj
        assert getattr(circuits, name) is obj
    assert get_type_hints(benchmarking.get_fault_distance)["method"] == DistanceMethod
