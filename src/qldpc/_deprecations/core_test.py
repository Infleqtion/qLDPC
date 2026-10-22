# SPDX-License-Identifier: Apache-2.0

"""Unit tests for core.py."""

from __future__ import annotations

import importlib
import inspect
import warnings

import numpy as np
import pytest

from qldpc import _deprecations, abstract, codes, decoders


def test_aliases() -> None:
    """Each deprecated name warns once, naming its replacement, and is the replacement itself."""
    for module_name, aliases in _deprecations.ALIASES.items():
        module = importlib.import_module(module_name)
        for name, target in aliases.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                value = getattr(module, name)
            target_module, _, attribute = target.partition(":")
            replacement = importlib.import_module(target_module)
            replacement = getattr(replacement, attribute) if attribute else replacement
            assert value is replacement

            # a private replacement is not named, a renamed object is named by its new name, and a
            # moved one by its full path
            path = f"{target_module}.{attribute}" if attribute else target_module
            if any(part.startswith("_") for part in path.split(".")):
                expected = f"{name} is deprecated and will be removed from the public API"
            elif attribute != name:
                expected = f"{name} is deprecated; use {attribute} instead"
                assert _get_from_package(module_name, attribute) is replacement
            else:
                expected = f"{name} is deprecated; use {path} instead"
            assert [str(warning.message) for warning in caught] == [expected]
            assert caught[0].category is DeprecationWarning
            assert caught[0].filename == __file__


def test_silent_aliases() -> None:
    """Deprecated functions that warn when called do not warn when accessed."""
    for module_name, aliases in _deprecations.SILENT_ALIASES.items():
        module = importlib.import_module(module_name)
        for name, target in aliases.items():
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                value = getattr(module, name)
            target_module, _, attribute = target.partition(":")
            assert value is getattr(importlib.import_module(target_module), attribute)


def test_deprecated_names() -> None:
    """Deprecated names are importable but unlisted, and unknown names are not deprecated."""
    for module_name in _deprecations.ALIASES.keys() | _deprecations.SILENT_ALIASES.keys():
        names = set(_deprecations.ALIASES.get(module_name, {}))
        names |= set(_deprecations.SILENT_ALIASES.get(module_name, {}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            module = importlib.import_module(module_name)
            namespace: dict[str, object] = {}
            exec(f"from {module_name} import {', '.join(sorted(names))}", namespace)  # noqa: S102
        assert names <= namespace.keys()
        assert not names & set(getattr(module, "__all__", ()))
    with pytest.raises(AttributeError, match=r"module 'qldpc\.decoders' has no attribute 'Other'"):
        _deprecations.get_deprecated_attribute("qldpc.decoders", "Other")


def test_deprecated_import_warns_once() -> None:
    """Importing a deprecated name from a package warns once, although Python looks it up twice."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from qldpc.decoders import Decoder  # type: ignore[attr-defined]
    assert Decoder is decoders.ErrorDecoder
    assert [str(warning.message) for warning in caught] == [
        "Decoder is deprecated; use ErrorDecoder instead"
    ]
    assert caught[0].filename == __file__


def test_star_imports_do_not_warn() -> None:
    """Star imports of modules that serve deprecated names do not warn."""
    for module_name in _deprecations.ALIASES.keys() | _deprecations.SILENT_ALIASES.keys():
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # a star import is only valid at module level, so execute it in a fresh namespace
            exec(f"from {module_name} import *", {})  # noqa: S102


def test_deprecated_methods() -> None:
    """Deprecated methods warn and delegate to their replacements."""
    classical_code = codes.RepetitionCode(3)
    qudit_code = codes.QuditCode(codes.FiveQubitCode())
    css_code = codes.SteaneCode()
    cases: list[tuple[object, str, tuple[object, ...], object, str]] = [
        (classical_code, "puncture", ([0],), classical_code.punctured([0]), "ClassicalCode"),
        (classical_code, "shorten", ([0],), classical_code.shortened([0]), "ClassicalCode"),
        (qudit_code, "get_dual_subsystem_code", (), qudit_code.dual(), "QuditCode"),
        (qudit_code, "conjugate", (), qudit_code.conjugated(), "QuditCode"),
        (css_code, "conjugate", (), css_code.conjugated().to_css(), "SteaneCode"),
    ]
    for code, method_name, args, expected, class_name in cases:
        with pytest.warns(DeprecationWarning, match="DEPRECATED") as caught:
            result = getattr(code, method_name)(*args)
        assert len(caught) == 1
        assert caught[0].filename == __file__
        assert str(caught[0].message).startswith(f"{class_name}.{method_name} is DEPRECATED")
        assert result == expected
    assert isinstance(result, codes.CSSCode)

    # an attached method reads like one defined in the class body
    puncture = vars(codes.ClassicalCode)["puncture"]
    assert puncture.__module__ == codes.ClassicalCode.__module__
    assert puncture.__qualname__ == "ClassicalCode.puncture"
    assert str(inspect.signature(puncture)).startswith("(self, bits")


def test_defunct_methods() -> None:
    """Defunct methods raise an error that names their replacements."""
    direct_decoder = decoders.custom.DirectDecoder(lambda word: word)
    words = np.zeros((2, 3), dtype=int)
    for obj, method_name, args, replacement in [
        (direct_decoder, "decode", (words[0],), "DirectDecoder.correct"),
        (direct_decoder, "decode_batch", (words,), "DirectDecoder.correct_batch"),
        (abstract.TrivialGroup(), "to_ring_array", ([[1]],), "RingArray.build"),
    ]:
        with pytest.raises(ValueError, match=f"DEFUNCT; use .*{replacement}"):
            getattr(obj, method_name)(*args)


def _get_from_package(module_name: str, name: str) -> object:
    """Get a name from a module, or else from the closest package in qldpc that provides it."""
    parts = module_name.split(".")
    modules = [importlib.import_module(".".join(parts[:end])) for end in range(len(parts), 1, -1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return next(getattr(module, name) for module in modules if hasattr(module, name))
