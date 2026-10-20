# SPDX-License-Identifier: Apache-2.0

"""Exact binary distance calculations and logical-operator counts."""

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
    "CSSLogicalCounter",
    "CSSLogicalCounts",
    "DistanceBackend",
    "DistanceMethod",
    "count_logicals_css",
    "get_distance_classical",
    "get_distance_css_brouwer_zimmermann",
    "get_distance_quantum",
    "validate_distance_backend",
    "validate_distance_method",
]
