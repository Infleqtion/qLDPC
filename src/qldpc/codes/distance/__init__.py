# SPDX-License-Identifier: Apache-2.0

"""Exact binary distance calculations, bounded enumeration, and affine Pauli searches."""

from .bounded import BoundedKernelSearch, PauliCosetSearch, PauliWeightDecision
from .core import (
    DistanceBackend,
    DistanceMethod,
    get_distance_classical,
    get_distance_css_brouwer_zimmermann,
    get_distance_quantum,
    validate_distance_backend,
    validate_distance_method,
)
from .logical_counts import CSSLogicalCounter, CSSLogicalCounts, count_logicals_css

__all__ = [
    "BoundedKernelSearch",
    "CSSLogicalCounter",
    "CSSLogicalCounts",
    "DistanceBackend",
    "DistanceMethod",
    "PauliCosetSearch",
    "PauliWeightDecision",
    "count_logicals_css",
    "get_distance_classical",
    "get_distance_css_brouwer_zimmermann",
    "get_distance_quantum",
    "validate_distance_backend",
    "validate_distance_method",
]
