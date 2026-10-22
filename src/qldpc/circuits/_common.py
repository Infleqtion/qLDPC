# SPDX-License-Identifier: Apache-2.0

"""Plumbing shared by qLDPC's circuit constructors."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable
from typing import ParamSpec, TypeVar

import galois

from qldpc import codes

_ReturnType = TypeVar("_ReturnType")
_Params = ParamSpec("_Params")


####################################################################################################
# qubit-only guard


def restrict_to_qubits(
    func: Callable[_Params, _ReturnType],
) -> Callable[_Params, _ReturnType]:
    """Restrict a circuit or tableau constructor to qubit-based codes."""

    @functools.wraps(func)
    def qubit_func(*args: _Params.args, **kwargs: _Params.kwargs) -> _ReturnType:
        if any(
            isinstance(arg, codes.QuditCode) and arg.field is not galois.GF2
            for arg in itertools.chain(args, kwargs.values())
        ):
            raise ValueError("Circuit methods are only supported for qubit codes")
        return func(*args, **kwargs)

    return qubit_func
