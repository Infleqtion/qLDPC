# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _methods.py."""

from __future__ import annotations

import pytest

from . import _methods


def test_distance_backend_validation() -> None:
    """Distance backend and exact-method selectors are restricted to documented values."""
    for backend in ["auto", "decoder", "gap", "sqetch"]:
        _methods.validate_distance_backend(backend)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown distance backend"):
        _methods.validate_distance_backend("other")  # type: ignore[arg-type]

    for method in ["brouwer_zimmermann", "brute_force"]:
        assert _methods.validate_distance_method(method) == method  # type: ignore[arg-type]
    assert _methods.validate_distance_method("information_sets") == "brouwer_zimmermann"
    with pytest.raises(ValueError, match="Unknown distance method"):
        _methods.validate_distance_method("other")  # type: ignore[arg-type]
