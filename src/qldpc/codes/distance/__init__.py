# SPDX-License-Identifier: Apache-2.0

"""Exact binary distance calculations, logical-operator counts, and distance method selectors."""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .core import (
    DistanceBackend,
    DistanceMethod,
    get_distance_classical,
    get_distance_quantum,
)
from .logical_counts import (
    CSSLogicalCounter,
    CSSLogicalCounts,
    LogicalCountMethod,
    count_logical_ops_css,
)

__all__ = [
    "CSSLogicalCounter",
    "CSSLogicalCounts",
    "DistanceBackend",
    "DistanceMethod",
    "LogicalCountMethod",
    "count_logical_ops_css",
    "get_distance_classical",
    "get_distance_quantum",
]

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
