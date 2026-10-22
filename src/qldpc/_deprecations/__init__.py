# SPDX-License-Identifier: Apache-2.0

"""Deprecated names, methods, arguments, and decoder construction of qLDPC.

All deprecation code lives in this private package.  The core module registers deprecated names,
defines deprecated methods, and provides the helpers that inline deprecated-argument blocks call.
The decoders module holds deprecated decoder construction, which qldpc.decoders serves through the
registry; it is not imported here, because decoder modules import it.
"""

from .core import (
    ALIASES,
    SILENT_ALIASES,
    defunct,
    deprecated_names,
    get_deprecated_attribute,
    install_deprecated_methods,
    warn_deprecated,
)

__all__ = [
    "ALIASES",
    "SILENT_ALIASES",
    "defunct",
    "deprecated_names",
    "get_deprecated_attribute",
    "install_deprecated_methods",
    "warn_deprecated",
]
