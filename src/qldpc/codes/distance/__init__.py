# SPDX-License-Identifier: Apache-2.0

"""Exact binary distance calculations, logical-operator counts, and distance method selectors."""

from .core import (
    DistanceBackend,
    DistanceMethod,
    get_distance_classical,
    get_distance_css_brouwer_zimmermann,
    get_distance_quantum,
    validate_distance_backend,
    validate_distance_method,
)
from .logical_counts import (
    CSSLogicalCounter,
    CSSLogicalCounts,
    LogicalCountMethod,
    count_logical_ops_css,
    validate_logical_count_method,
)

__all__ = [
    "CSSLogicalCounter",
    "CSSLogicalCounts",
    "DistanceBackend",
    "DistanceMethod",
    "LogicalCountMethod",
    "count_logical_ops_css",
    "get_distance_classical",
    "get_distance_css_brouwer_zimmermann",
    "get_distance_quantum",
    "validate_distance_backend",
    "validate_distance_method",
    "validate_logical_count_method",
]
