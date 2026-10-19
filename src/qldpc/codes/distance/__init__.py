# SPDX-License-Identifier: Apache-2.0

"""Exact binary distance calculations with unchanged public entry points."""

from .core import (
    DistanceBackend,
    DistanceMethod,
    get_distance_classical,
    get_distance_css_brouwer_zimmermann,
    get_distance_quantum,
    validate_distance_backend,
    validate_distance_method,
)

__all__ = [
    "DistanceBackend",
    "DistanceMethod",
    "get_distance_classical",
    "get_distance_css_brouwer_zimmermann",
    "get_distance_quantum",
    "validate_distance_backend",
    "validate_distance_method",
]
