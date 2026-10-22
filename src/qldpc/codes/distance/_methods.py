# SPDX-License-Identifier: Apache-2.0

"""Validation of exact-distance method and distance-bound backend selectors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .core import DistanceBackend, DistanceMethod

# The canonical name of every accepted exact-distance method name, including aliases.  Logical
# counts accept the same names.
DISTANCE_METHODS: dict[str, Literal["brouwer_zimmermann", "brute_force"]] = {
    "brouwer_zimmermann": "brouwer_zimmermann",
    "brute_force": "brute_force",
    "information_sets": "brouwer_zimmermann",
}


####################################################################################################
# method and backend selectors


def validate_distance_backend(backend: DistanceBackend) -> None:
    """Validate a distance-bound backend selector."""
    if backend not in ("auto", "decoder", "gap", "sqetch"):
        raise ValueError(
            f"Unknown distance backend {backend!r}; choose from 'auto', 'decoder', 'gap', or "
            "'sqetch'."
        )


def validate_distance_method(
    method: DistanceMethod,
) -> Literal["brouwer_zimmermann", "brute_force"]:
    """Validate an exact-distance method selector, and return its canonical name.

    The alias ``"information_sets"`` resolves to ``"brouwer_zimmermann"``.
    """
    if method not in DISTANCE_METHODS:
        raise ValueError(
            f"Unknown distance method {method!r}; choose from 'brouwer_zimmermann' or "
            "'brute_force'."
        )
    return DISTANCE_METHODS[method]
