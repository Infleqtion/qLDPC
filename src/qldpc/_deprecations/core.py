# SPDX-License-Identifier: Apache-2.0

"""Registry of deprecated names, deprecated methods, and helpers that warn about deprecated usage.

A module with deprecated names ends with this marked stub, which serves the names that ALIASES and
SILENT_ALIASES list for that module:

    if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

        def __getattr__(name: str) -> object:
            return _deprecations.get_deprecated_attribute(__name__, name)

Python calls a module-level __getattr__ (PEP 562) only for names that the module does not define.
A deprecated name thereby refers to the same object as its replacement, which preserves isinstance
checks, subclassing, and unpickling.  Type checkers do not see the stub, so they reject deprecated
names.  A package that must keep deprecated names in its __all__ extends it with deprecated_names.

Registry targets are lazy "module:attribute" strings, and this module imports no qLDPC subpackage
at import time, so that every module can import it without an import cycle.  Deprecated methods are
defined at the bottom of this module, and attached to their classes by install_deprecated_methods,
which qldpc/__init__.py calls once all subpackages are imported.  Deprecated arguments and modes
are handled where they are accepted, by calling warn_deprecated or raising defunct(...).
"""

from __future__ import annotations

import importlib
import sys
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from qldpc._util import get_external_caller_stacklevel

if TYPE_CHECKING:
    from collections.abc import Collection
    from types import FrameType

    import numpy as np
    import numpy.typing as npt

    from qldpc.abstract.groups import NestedSequence

_FunctionT = TypeVar("_FunctionT", bound=Callable[..., object])


####################################################################################################
# registry and attribute lookup


# Deprecated names that warn when accessed, by module, mapped to "module:attribute" targets.
ALIASES: dict[str, dict[str, str]] = {
    "qldpc.abstract": {
        "Element": "qldpc.abstract.rings:RingMember",
        "Protograph": "qldpc.abstract.ring_array:RingArray",
    },
    "qldpc.abstract.ring_array": {
        "Protograph": "qldpc.abstract.ring_array:RingArray",
    },
    "qldpc.abstract.rings": {
        "Element": "qldpc.abstract.rings:RingMember",
        "Protograph": "qldpc.abstract.ring_array:RingArray",
    },
    "qldpc.circuits": {
        "validate_syndrome_qubit_ids": "qldpc.circuits.memory._common:validate_syndrome_qubit_ids",
    },
    "qldpc.circuits.memory": {
        "validate_syndrome_qubit_ids": "qldpc.circuits.memory._common:validate_syndrome_qubit_ids",
    },
    "qldpc.codes": {
        "get_distance_css_brouwer_zimmermann": (
            "qldpc.codes.distance._brouwer_zimmermann:get_distance_css_brouwer_zimmermann"
        ),
        "validate_distance_backend": "qldpc.codes.distance._methods:validate_distance_backend",
        "validate_distance_method": "qldpc.codes.distance._methods:validate_distance_method",
    },
    "qldpc.codes.distance": {
        "get_distance_css_brouwer_zimmermann": (
            "qldpc.codes.distance._brouwer_zimmermann:get_distance_css_brouwer_zimmermann"
        ),
        "validate_distance_backend": "qldpc.codes.distance._methods:validate_distance_backend",
        "validate_distance_method": "qldpc.codes.distance._methods:validate_distance_method",
    },
    "qldpc.decoders": {
        "BatchDecoder": "qldpc.decoders.protocols:BatchErrorDecoder",
        "CompiledSequentialWindowDecoder": (
            "qldpc.decoders.sinter.window:CompiledSequentialWindowDecoder"
        ),
        "CompiledSinterDecoder": "qldpc.decoders.sinter.core:CompiledSinterDecoder",
        "CompiledSubgraphDecoder": "qldpc.decoders.sinter.subgraph:CompiledSubgraphDecoder",
        "CompiledTrivialDecoder": "qldpc.decoders.sinter.core:CompiledTrivialDecoder",
        "CompositeDecoder": "qldpc.decoders.custom.composition:CompositeDecoder",
        "Decoder": "qldpc.decoders.protocols:ErrorDecoder",
        "DirectDecoder": "qldpc.decoders.custom.composition:DirectDecoder",
        "ExpandedErrorDecoder": "qldpc.decoders.adapters._dem_matching:ExpandedErrorDecoder",
        "FlipPattern": "qldpc.decoders.dems:FlipPattern",
        "FrontierObservableDecoder": "qldpc.decoders.external.frontier:FrontierObservableDecoder",
        "GUFDecoder": "qldpc.decoders.custom.guf:GUFDecoder",
        "ILPDecoder": "qldpc.decoders.custom.ilp:ILPDecoder",
        "LookupDecoder": "qldpc.decoders.custom.lookup:LookupDecoder",
        "ObservableLookupDecoder": "qldpc.decoders.custom.lookup:ObservableLookupDecoder",
        "RelayBPDecoder": "qldpc.decoders.external.relay_bp:RelayBPDecoder",
        "SequentialSinterDecoder": "qldpc.decoders.sinter.window:SequentialWindowDecoder",
        "SubgraphSinterDecoder": "qldpc.decoders.sinter.subgraph:SubgraphDecoder",
        "TesseractDecoder": "qldpc.decoders.external.tesseract:TesseractDecoder",
        "WeightedLookupDecoder": "qldpc.decoders.custom.lookup:WeightedLookupDecoder",
        "WeightedObservableLookupDecoder": (
            "qldpc.decoders.custom.lookup:WeightedObservableLookupDecoder"
        ),
        "WrappedErrorDecoder": "qldpc.decoders._common:WrappedErrorDecoder",
        "as_error_decoder": "qldpc.decoders._common:as_error_decoder",
        "batch_decode_errors": "qldpc.decoders._common:batch_decode_errors",
        "compiles_for_dem": "qldpc.decoders._capabilities:compiles_for_dem",
        "get_decoder_BF": "qldpc._deprecations.decoders:get_decoder_bf",
        "get_decoder_BP_LSD": "qldpc._deprecations.decoders:get_decoder_bp_lsd",
        "get_decoder_BP_OSD": "qldpc._deprecations.decoders:get_decoder_bp_osd",
        "get_decoder_GUF": "qldpc._deprecations.decoders:get_decoder_guf",
        "get_decoder_ILP": "qldpc._deprecations.decoders:get_decoder_ilp",
        "get_decoder_MWPM": "qldpc._deprecations.decoders:get_decoder_mwpm",
        "get_decoder_RBP": "qldpc._deprecations.decoders:get_decoder_rbp",
        "get_legacy_decoder_migration_message": (
            "qldpc._deprecations.decoders:get_legacy_decoder_migration_message"
        ),
        "is_prebuilt_decoder": "qldpc.decoders._capabilities:is_prebuilt_decoder",
        "is_prebuilt_observable_decoder": (
            "qldpc.decoders._capabilities:is_prebuilt_observable_decoder"
        ),
        "lookup_table": "qldpc.decoders.custom.lookup:lookup",
        "match_error_decoder_to_dem": (
            "qldpc.decoders.adapters._dem_matching:match_error_decoder_to_dem"
        ),
        "reject_prebuilt_decoder": (
            "qldpc.decoders.construction._resolution:reject_prebuilt_decoder"
        ),
        "reject_removed_decoder_args": "qldpc._deprecations.decoders:reject_removed_decoder_args",
        "resolve_decoder": "qldpc.decoders.construction._resolution:resolve_decoder",
        "resolve_observable_decoder": (
            "qldpc.decoders.construction._resolution:resolve_observable_decoder"
        ),
        "supports_batch_decoding": "qldpc.decoders._common:supports_batch_decoding",
    },
    "qldpc.decoders.adapters": {
        "BitPackedObservableDecoder": (
            "qldpc.decoders.adapters._observable_decoders:BitPackedObservableDecoder"
        ),
        "ErrorsToFieldObservablesDecoder": (
            "qldpc.decoders.adapters._observable_decoders:ErrorsToFieldObservablesDecoder"
        ),
        "ExpandedErrorDecoder": "qldpc.decoders.adapters._dem_matching:ExpandedErrorDecoder",
        "match_error_decoder_to_dem": (
            "qldpc.decoders.adapters._dem_matching:match_error_decoder_to_dem"
        ),
        "validate_decoder_output": (
            "qldpc.decoders.adapters._observable_decoders:validate_decoder_output"
        ),
        "validate_observable_decoder": (
            "qldpc.decoders.adapters._observable_decoders:validate_observable_decoder"
        ),
    },
    "qldpc.decoders.custom": {
        "BatchDecoder": "qldpc.decoders.protocols:BatchErrorDecoder",
        "Decoder": "qldpc.decoders.protocols:ErrorDecoder",
        "PLACEHOLDER_ERROR_RATE": "qldpc.decoders._common:PLACEHOLDER_ERROR_RATE",
        "RelayBPDecoder": "qldpc.decoders.external.relay_bp:RelayBPDecoder",
    },
    "qldpc.decoders.sinter": {
        "Decoder": "qldpc.decoders.protocols:ErrorDecoder",
        "SequentialSinterDecoder": "qldpc.decoders.sinter.window:SequentialWindowDecoder",
        "SubgraphSinterDecoder": "qldpc.decoders.sinter.subgraph:SubgraphDecoder",
    },
}

# Names served without warning when accessed: deprecated functions, which warn when called instead,
# and objects that moved to another module, which old pickles may still reference by their old path.
SILENT_ALIASES: dict[str, dict[str, str]] = {
    "qldpc.abstract.rings": {
        "RingArray": "qldpc.abstract.ring_array:RingArray",
    },
    "qldpc.decoders": {
        "decode": "qldpc._deprecations.decoders:decode",
        "decode_observables": "qldpc._deprecations.decoders:decode_observables",
        "get_decoder": "qldpc._deprecations.decoders:get_decoder",
        "get_decoder_bf": "qldpc._deprecations.decoders:get_decoder_bf",
        "get_decoder_bp_lsd": "qldpc._deprecations.decoders:get_decoder_bp_lsd",
        "get_decoder_bp_osd": "qldpc._deprecations.decoders:get_decoder_bp_osd",
        "get_decoder_guf": "qldpc._deprecations.decoders:get_decoder_guf",
        "get_decoder_ilp": "qldpc._deprecations.decoders:get_decoder_ilp",
        "get_decoder_lookup": "qldpc._deprecations.decoders:get_decoder_lookup",
        "get_decoder_mwpm": "qldpc._deprecations.decoders:get_decoder_mwpm",
        "get_decoder_rbp": "qldpc._deprecations.decoders:get_decoder_rbp",
        "get_error_decoder": "qldpc._deprecations.decoders:get_error_decoder",
        "get_observable_decoder": "qldpc._deprecations.decoders:get_observable_decoder",
    },
    "qldpc.decoders.adapters.error_decoders": {
        "ExpandedErrorDecoder": "qldpc.decoders.adapters._dem_matching:ExpandedErrorDecoder",
    },
    "qldpc.decoders.protocols": {
        "WrappedErrorDecoder": "qldpc.decoders._common:WrappedErrorDecoder",
    },
}


def get_deprecated_attribute(module_name: str, name: str) -> object:
    """Retrieve a deprecated attribute of a module, warning if ALIASES lists it.

    The warning names the replacement: by its new name if it was renamed, by its full path if it
    only moved, or not at all if its new home is private.

    Args:
        module_name: The name of the module whose attribute is being retrieved.
        name: The name of the attribute being retrieved.

    Returns:
        The object that the deprecated name refers to.

    Raises:
        AttributeError: If the registry lists no such name for the module.
    """
    if (target := SILENT_ALIASES.get(module_name, {}).get(name)) is not None:
        return _import_target(target)
    if (target := ALIASES.get(module_name, {}).get(name)) is None:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    if not _is_import_probe(sys._getframe(2)):
        warn_deprecated(
            _get_alias_warning(name, target), stacklevel=get_external_caller_stacklevel()
        )
    return _import_target(target)


def deprecated_names(module_name: str) -> list[str]:
    """List the deprecated names that the registry serves for a module, sorted."""
    return sorted(ALIASES.get(module_name, {}).keys() | SILENT_ALIASES.get(module_name, {}).keys())


def _import_target(target: str) -> object:
    """Import the object at a "module:attribute" target."""
    module_name, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module_name), attribute)


def _get_alias_warning(name: str, target: str) -> str:
    """Describe the replacement for a deprecated name with the given "module:attribute" target."""
    module_name, _, attribute = target.partition(":")
    replacement = attribute if attribute != name else f"{module_name}.{attribute}"
    if any(part.startswith("_") for part in replacement.split(".")):
        return f"{name} is deprecated and will be removed from the public API"
    return f"{name} is deprecated; use {replacement} instead"


def _is_import_probe(frame: FrameType | None) -> bool:
    """Is a frame the probe that checks for names before a ``from package import ...`` statement?

    Python checks that a package provides each name in ``from package import name`` before it
    retrieves the name, so a module-level __getattr__ is called twice for one such statement.
    """
    return (
        frame is not None
        and frame.f_code.co_name == "_handle_fromlist"
        and frame.f_globals.get("__name__") == "importlib._bootstrap"
    )


####################################################################################################
# warnings and defunct errors


def warn_deprecated(message: str, *, stacklevel: int) -> None:
    """Issue a DeprecationWarning, with a stacklevel relative to the caller of this function."""
    warnings.warn(message, DeprecationWarning, stacklevel=stacklevel + 1)


def defunct(message: str) -> ValueError:
    """Return the error that a defunct method or argument raises when it is used."""
    return ValueError(message)


####################################################################################################
# deprecated methods


def install_deprecated_methods() -> None:
    """Attach deprecated methods to their classes."""
    from qldpc.abstract import TrivialGroup
    from qldpc.codes import ClassicalCode, CSSCode, QuditCode
    from qldpc.decoders.custom import DirectDecoder

    @_deprecated_method(ClassicalCode)
    def puncture(self: ClassicalCode, bits: Collection[int]) -> ClassicalCode:
        """Deprecated alias for ClassicalCode.punctured."""
        warn_deprecated(
            "ClassicalCode.puncture is DEPRECATED; use ClassicalCode.punctured instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return self.punctured(bits)

    @_deprecated_method(ClassicalCode)
    def shorten(self: ClassicalCode, bits: Collection[int]) -> ClassicalCode:
        """Deprecated alias for ClassicalCode.shortened."""
        warn_deprecated(
            "ClassicalCode.shorten is DEPRECATED; use ClassicalCode.shortened instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return self.shortened(bits)

    @_deprecated_method(QuditCode)
    def get_dual_subsystem_code(self: QuditCode) -> QuditCode:
        """Deprecated alias for self.dual()."""
        warn_deprecated(
            "QuditCode.get_dual_subsystem_code is DEPRECATED; use QuditCode.dual instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        return self.dual()

    @_deprecated_method(QuditCode)
    def conjugate(self: QuditCode) -> QuditCode:
        """Deprecated alias for ``conjugated()``."""
        name = type(self).__name__
        warn_deprecated(
            f"{name}.conjugate is DEPRECATED; use {name}.conjugated instead",
            stacklevel=get_external_caller_stacklevel(),
        )
        code = self.conjugated()
        return code.to_css() if isinstance(self, CSSCode) else code

    @_deprecated_method(DirectDecoder)
    def decode(self: DirectDecoder, word: npt.NDArray[np.int_]) -> None:
        """DEFUNCT alias for DirectDecoder.correct."""
        raise defunct(
            "DirectDecoder.decode is DEFUNCT; use DirectDecoder.correct instead.  A DirectDecoder"
            " corrects code words, and is not a syndrome decoder"
        )

    @_deprecated_method(DirectDecoder)
    def decode_batch(self: DirectDecoder, words: npt.NDArray[np.int_]) -> None:
        """DEFUNCT alias for DirectDecoder.correct_batch."""
        raise defunct(
            "DirectDecoder.decode_batch is DEFUNCT; use DirectDecoder.correct_batch instead.  A"
            " DirectDecoder corrects code words, and is not a syndrome decoder"
        )

    @_deprecated_method(TrivialGroup, static=True)
    def to_ring_array(data: npt.NDArray[np.int_] | NestedSequence) -> None:
        """DEFUNCT alias for qldpc.abstract.RingArray.build(data)."""
        raise defunct(
            "TrivialGroup.to_ring_array(data) is DEFUNCT; use"
            " qldpc.abstract.RingArray.build(data) instead"
        )


def _deprecated_method(cls: type, *, static: bool = False) -> Callable[[_FunctionT], _FunctionT]:
    """Attach a function to a class as a method, with the module and qualified name of a member."""

    def attach(function: _FunctionT) -> _FunctionT:
        function.__module__ = cls.__module__
        function.__qualname__ = f"{cls.__qualname__}.{function.__name__}"
        function.__annotations__.pop("self", None)  # methods in class bodies leave self implicit
        setattr(cls, function.__name__, staticmethod(function) if static else function)
        return function

    return attach
