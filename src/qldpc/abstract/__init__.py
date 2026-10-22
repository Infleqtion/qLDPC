# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from qldpc import _deprecations

from . import ring_array as ring_array
from . import rings as rings
from ._monomials import (
    get_coefficient_and_exponents,
    iter_monomial_terms,
)
from .groups import (
    GL,
    PGL,
    PSL,
    SL,
    AbelianGroup,
    AlternatingGroup,
    CyclicGroup,
    DihedralGroup,
    GeneralLinearGroup,
    Group,
    GroupMember,
    ProjectiveGeneralLinearGroup,
    ProjectiveSpecialLinearGroup,
    QuaternionGroup,
    SmallGroup,
    SpecialLinearGroup,
    SymmetricGroup,
    TrivialGroup,
    WreathProductGroup,
    resolve_field,
)
from .linalg import (
    block_diag,
    get_howell_dual,
    kron,
    matmul,
)
from .ring_array import (
    RingArray,
)
from .rings import (
    GroupRing,
    RingMember,
)
from .wedderburn_artin import (
    WedderburnArtinComponentTransformer,
    WedderburnArtinTransformer,
)

__all__ = [
    "GL",
    "PGL",
    "PSL",
    "SL",
    "AbelianGroup",
    "AlternatingGroup",
    "CyclicGroup",
    "DihedralGroup",
    "GeneralLinearGroup",
    "Group",
    "GroupMember",
    "GroupRing",
    "ProjectiveGeneralLinearGroup",
    "ProjectiveSpecialLinearGroup",
    "QuaternionGroup",
    "RingArray",
    "RingMember",
    "SmallGroup",
    "SpecialLinearGroup",
    "SymmetricGroup",
    "TrivialGroup",
    "WedderburnArtinComponentTransformer",
    "WedderburnArtinTransformer",
    "WreathProductGroup",
    "block_diag",
    "get_coefficient_and_exponents",
    "get_howell_dual",
    "iter_monomial_terms",
    "kron",
    "matmul",
    "resolve_field",
]
__all__.extend(_deprecations.deprecated_names(__name__))

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
