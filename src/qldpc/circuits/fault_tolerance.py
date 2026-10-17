# SPDX-License-Identifier: Apache-2.0

"""Deprecated imports for state-preparation spread certification.

Use ``qldpc.circuits.benchmarking.state_prep_spread`` or the ``qldpc.circuits`` public exports.
"""

from typing import TYPE_CHECKING, Any

from qldpc._util import get_deprecated_alias

from .benchmarking import state_prep_spread as _state_prep_spread

__all__ = [
    "StatePrepSpreadCertificate",
    "StatePrepSpreadCounterexample",
    "StatePrepSpreadStatistics",
    "certify_state_prep_spread",
]

_DEPRECATED_ALIASES = {name: getattr(_state_prep_spread, name) for name in __all__}

if TYPE_CHECKING:
    from .benchmarking.state_prep_spread import (
        StatePrepSpreadCertificate,
        StatePrepSpreadCounterexample,
        StatePrepSpreadStatistics,
        certify_state_prep_spread,
    )
else:

    def __getattr__(name: str) -> Any:
        """Preserve old public imports while directing callers to the specific module."""
        return get_deprecated_alias(
            __name__,
            name,
            _DEPRECATED_ALIASES,
            {name: f"{_state_prep_spread.__name__}.{name}" for name in __all__},
        )
