# SPDX-License-Identifier: Apache-2.0

"""Exact binary distance calculations, and distance method and backend types and validation."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

from ._binary import (
    as_binary_matrix,
    assert_binary,
    get_independent_rows,
    get_nested_code_basis,
    symplectic_to_hamming,
)
from ._brouwer_zimmermann import (
    exhaustive_is_cheaper,
    get_distance_brouwer_zimmermann,
)
from ._brute_force import (
    get_distance_quantum_brute_force,
)

DistanceBackend = Literal["auto", "decoder", "gap", "sqetch"]
DistanceMethod = Literal["brouwer_zimmermann", "brute_force", "information_sets"]

# The canonical name of every accepted exact-distance method name, including aliases.  Logical
# counts accept the same names.
_DISTANCE_METHODS: dict[str, Literal["brouwer_zimmermann", "brute_force"]] = {
    "brouwer_zimmermann": "brouwer_zimmermann",
    "brute_force": "brute_force",
    "information_sets": "brouwer_zimmermann",
}


####################################################################################################
# exact binary distance


def get_distance_classical(
    generators: npt.ArrayLike,
    *,
    cutoff: int = 1,
    block_size: int = 15,
    method: DistanceMethod = "brouwer_zimmermann",
) -> int:
    """Distance of a classical linear binary code.

    Args:
        generators: The generator matrix of the classical code whose distance we want to compute.
            Both methods reduce redundant rows to a linearly independent basis.
        cutoff: Exit early and return once an upper bound on distance falls to or below this cutoff.
        block_size: Batch size exponent.  BZ returns to Python after at most
            ``2**min(block_size, 15)`` supports; brute force vectorizes ``2**block_size`` values.
        method: Exact-distance method.  ``"brouwer_zimmermann"`` (default) enumerates fixed-weight
            combinations in several information-set bases and stops once its lower and upper bounds
            meet.  ``"brute_force"`` enumerates every nonzero codeword.  ``"information_sets"`` is
            an alias for ``"brouwer_zimmermann"``.

    Returns:
        The minimum Hamming distance between different code words, or equivalently the minimum
        Hamming weight of a nontrivial code word.
    """
    method = _validate_distance_method(method)
    assert_binary(generators, "generators")
    matrix = as_binary_matrix(generators)
    basis = get_independent_rows(matrix)
    if not len(basis):
        return matrix.shape[1]

    if method == "brouwer_zimmermann":
        return get_distance_brouwer_zimmermann(
            basis,
            labels=None,
            cutoff=cutoff,
            block_size=block_size,
        )

    # The reduced classical basis needs no excluded subspace.
    return get_distance_quantum_brute_force(
        basis,
        np.empty((0, matrix.shape[1]), dtype=np.uint8),
        cutoff=cutoff,
        block_size=block_size,
        homogeneous=True,
    )


def get_distance_quantum(
    logical_ops: npt.ArrayLike,
    stabilizers: npt.ArrayLike,
    *,
    cutoff: int = 1,
    block_size: int = 15,
    homogeneous: bool = False,
    method: DistanceMethod = "brouwer_zimmermann",
) -> int:
    """Distance of a binary quantum code.

    Args:
        logical_ops: Rows that generate the logical space together with the stabilizers.
            Both methods remove redundant rows and exclude the stabilizer row space from witnesses.
        stabilizers: A matrix whose rows represent stabilizers of the code.
        cutoff: Exit early and return once an upper bound on distance falls to or below this cutoff.
        block_size: Batch size exponent.  BZ returns to Python after at most
            ``2**min(block_size, 15)`` supports; brute force vectorizes ``2**block_size`` values.
        homogeneous: If True, all Pauli strings (represented by rows of logical_ops and stabilizers)
            are assumed to have the same homogeneous (X or Z) type.  If False, Pauli strings may
            have mixed (X, Y, or Z) support on different qubits.
        method: Exact-distance method.  ``"brouwer_zimmermann"`` (default) uses an exclusion-aware
            search over the logical space modulo stabilizers.  ``"brute_force"`` enumerates eligible
            words using independent bases constructed internally.  ``"information_sets"`` is an
            alias for ``"brouwer_zimmermann"``.

    Returns:
        The exact minimum weight of a nontrivial logical operator in ``logical_ops`` modulo
        stabilizers (the code distance).  As an optimization, as soon as the lightest operator seen
        so far has weight ``<= cutoff`` the search stops early and returns that weight -- an upper
        bound on the true distance, in ``[distance, cutoff]``.  With the default ``cutoff=1`` this
        still returns the exact distance for every valid input (an early return can exceed the
        distance only when ``cutoff >= 2``).  Pass ``cutoff=0`` to disable the early exit and force
        the exact minimum.

    More specifically, if homogeneous is True, then::

        (a) each Pauli string is represented by a binary vector of length equal to the number of
            data qubits in a code, indicating the nontrivial support of the Pauli string; and
        (b) the weight of a Pauli string is the Hamming weight of the corresponding bitstring.

    If ``homogeneous is False``, then::

        (a) each Pauli string is represented by a symplectic binary vector (of length 2 * data
            qubits) with the first and second halves indicating the X and Z Pauli support; and
        (b) the weight of a Pauli string is the symplectic weight of the corresponding bitstring.
    """
    method = _validate_distance_method(method)
    assert_binary(logical_ops, "logical_ops")
    assert_binary(stabilizers, "stabilizers")
    logical_matrix = as_binary_matrix(logical_ops)
    stabilizer_matrix = as_binary_matrix(stabilizers, num_cols=logical_matrix.shape[1])
    num_bits = logical_matrix.shape[1]
    physical_size = num_bits if homogeneous else num_bits // 2
    if not homogeneous and num_bits % 2:
        raise ValueError("Symplectic operators must have an even number of columns")
    if not len(logical_matrix):
        return physical_size

    if method == "brouwer_zimmermann":
        # Verify independence before bypassing adapted-basis preparation.
        num_rows = len(logical_matrix) + len(stabilizer_matrix)
        if exhaustive_is_cheaper(num_rows, num_bits) and num_rows == len(
            get_independent_rows(np.vstack([stabilizer_matrix, logical_matrix]))
        ):
            return get_distance_quantum_brute_force(
                logical_matrix,
                stabilizer_matrix,
                cutoff=cutoff,
                block_size=block_size,
                homogeneous=homogeneous,
            )

    # Separate an independent excluded basis from a complement representing the quotient.
    basis, labels = get_nested_code_basis(logical_matrix, stabilizer_matrix)
    if labels.shape[1] == 0:
        return 0
    if method == "brute_force":
        num_stabilizers = len(basis) - labels.shape[1]
        return get_distance_quantum_brute_force(
            basis[num_stabilizers:],
            basis[:num_stabilizers],
            cutoff=cutoff,
            block_size=block_size,
            homogeneous=homogeneous,
        )

    divisor = 1
    if not homogeneous:
        basis = symplectic_to_hamming(basis)
        divisor = 2

    distance = get_distance_brouwer_zimmermann(
        basis,
        labels,
        cutoff=cutoff * divisor,
        block_size=block_size,
        weight_divisor=divisor,
    )
    return distance // divisor


####################################################################################################
# method and backend selectors


def _validate_distance_backend(backend: DistanceBackend) -> None:
    """Validate a distance-bound backend selector."""
    if backend not in ("auto", "decoder", "gap", "sqetch"):
        raise ValueError(
            f"Unknown distance backend {backend!r}; choose from 'auto', 'decoder', 'gap', or "
            "'sqetch'."
        )


def _validate_distance_method(
    method: DistanceMethod,
) -> Literal["brouwer_zimmermann", "brute_force"]:
    """Validate an exact-distance method selector, and return its canonical name.

    The alias ``"information_sets"`` resolves to ``"brouwer_zimmermann"``.
    """
    if method not in _DISTANCE_METHODS:
        raise ValueError(
            f"Unknown distance method {method!r}; choose from 'brouwer_zimmermann' or "
            "'brute_force'."
        )
    return _DISTANCE_METHODS[method]
