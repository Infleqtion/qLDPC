# SPDX-License-Identifier: Apache-2.0

"""Resolution of finite-field arguments."""

from __future__ import annotations

import galois


def resolve_field(
    field: int | type[galois.FieldArray] | None,
) -> type[galois.FieldArray]:
    """Parse a finite field argument to obtain an actual finite field."""
    if field is None:
        return galois.GF2
    if isinstance(field, int):
        return galois.GF(field)
    return field
