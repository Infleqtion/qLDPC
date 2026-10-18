# SPDX-License-Identifier: Apache-2.0

"""Methods for computing the (exact) distance of error-correcting codes."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt
from numba import njit
from numba.extending import register_jitable

DistanceBackend = Literal["auto", "decoder", "gap", "sqetch"]
DistanceMethod = Literal["brouwer_zimmermann", "brute_force"]


def validate_distance_backend(backend: DistanceBackend) -> None:
    """Validate a distance-bound backend selector."""
    if backend not in ("auto", "decoder", "gap", "sqetch"):
        raise ValueError(
            f"Unknown distance backend {backend!r}; choose from 'auto', 'decoder', 'gap', or "
            "'sqetch'."
        )


def validate_distance_method(method: DistanceMethod) -> None:
    """Validate an exact-distance method selector."""
    if method not in ("brouwer_zimmermann", "brute_force"):
        raise ValueError(
            f"Unknown distance method {method!r}; choose from 'brouwer_zimmermann' or "
            "'brute_force'."
        )


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
            Brouwer-Zimmermann mode reduces redundant rows to an independent basis.
        cutoff: Exit early and return once an upper bound on distance falls to or below this cutoff.
        block_size: Batch size exponent.  BZ returns to Python after at most
            ``2**min(block_size, 15)`` supports; brute force vectorizes ``2**block_size`` values.
        method: Exact-distance method.  ``"brouwer_zimmermann"`` (default) enumerates fixed-weight
            combinations in several information-set bases and stops once its lower and upper bounds
            meet.  ``"brute_force"`` enumerates every nonzero codeword.

    Returns:
        The minimum Hamming distance between different code words, or equivalently the minimum
        Hamming weight of a nontrivial code word.
    """
    validate_distance_method(method)
    _assert_binary(generators, "generators")
    if method == "brouwer_zimmermann":
        matrix = _as_binary_matrix(generators)
        basis = _get_independent_rows(matrix)
        if not len(basis):
            return matrix.shape[1]
        return _get_distance_brouwer_zimmermann(
            basis,
            labels=None,
            cutoff=cutoff,
            block_size=block_size,
        )

    # Classical brute force is the quantum calculation with no stabilizers.
    return get_distance_quantum(
        logical_ops=generators,
        stabilizers=[],
        cutoff=cutoff,
        block_size=block_size,
        homogeneous=True,
        method=method,
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
        logical_ops: A matrix whose rows represent logical operators of the code.  These rows must
            span a complement of the stabilizers in the logical space.  Brouwer-Zimmermann mode
            reduces redundant rows and excludes the stabilizer row space from candidate witnesses.
        stabilizers: A matrix whose rows represent stabilizers of the code.
        cutoff: Exit early and return once an upper bound on distance falls to or below this cutoff.
        block_size: Batch size exponent.  BZ returns to Python after at most
            ``2**min(block_size, 15)`` supports; brute force vectorizes ``2**block_size`` values.
        homogeneous: If True, all Pauli strings (represented by rows of logical_ops and stabilizers)
            are assumed to have the same homogeneous (X or Z) type.  If False, Pauli strings may
            have mixed (X, Y, or Z) support on different qubits.
        method: Exact-distance method.  ``"brouwer_zimmermann"`` (default) uses an exclusion-aware
            search over the logical space modulo stabilizers.  ``"brute_force"`` enumerates every
            stabilizer and nonzero logical combination, and requires those input rows to be linearly
            independent.

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
    validate_distance_method(method)
    _assert_binary(logical_ops, "logical_ops")
    _assert_binary(stabilizers, "stabilizers")
    if method == "brouwer_zimmermann":
        logical_matrix = _as_binary_matrix(logical_ops)
        stabilizer_matrix = _as_binary_matrix(stabilizers, num_cols=logical_matrix.shape[1])
        num_bits = logical_matrix.shape[1]
        physical_size = num_bits if homogeneous else num_bits // 2
        if not homogeneous and num_bits % 2:
            raise ValueError("Symplectic operators must have an even number of columns")

        # Exhaustive enumeration of the input rows avoids building the quotient basis and
        # information sets, but requires the rows to be linearly independent.
        num_rows = len(logical_matrix) + len(stabilizer_matrix)
        if _exhaustive_is_cheaper(num_rows, num_bits) and num_rows == len(
            _get_independent_rows(np.vstack([stabilizer_matrix, logical_matrix]))
        ):
            return _get_distance_quantum_brute_force(
                logical_matrix,
                stabilizer_matrix,
                cutoff=cutoff,
                block_size=block_size,
                homogeneous=homogeneous,
            )

        basis, labels = _get_nested_code_basis(logical_matrix, stabilizer_matrix)
        if labels.shape[1] == 0:
            return 0 if len(logical_matrix) else physical_size

        divisor = 1
        if not homogeneous:
            basis = _symplectic_to_hamming(basis)
            divisor = 2

        distance = _get_distance_brouwer_zimmermann(
            basis,
            labels,
            cutoff=cutoff * divisor,
            block_size=block_size,
            weight_divisor=divisor,
        )
        return distance // divisor

    return _get_distance_quantum_brute_force(
        logical_ops,
        stabilizers,
        cutoff=cutoff,
        block_size=block_size,
        homogeneous=homogeneous,
    )


def get_distance_css_brouwer_zimmermann(
    sectors: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
    *,
    cutoff: int,
    block_size: int = 15,
    upper_bound: int | None = None,
) -> int:
    """Compute the minimum distance across homogeneous binary CSS sectors.

    Args:
        sectors: Logical operators and stabilizers for every sector to search.  All operators use
            homogeneous binary support vectors rather than symplectic vectors.
        cutoff: Exit early and return once an upper bound on distance falls to or below this cutoff.
        block_size: Batch size exponent, as in :func:`get_distance_quantum`.
        upper_bound: Weight of a known logical operator from an omitted sector, if any.  The result
            is the minimum of this bound and the distances of the supplied sectors.

    Returns:
        The minimum exact distance across the supplied sectors and optional upper bound.  As with
        :func:`get_distance_quantum`, a result at or below ``cutoff`` is an observed upper bound.
    """
    problems: list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]] = []
    best = upper_bound
    for logical_ops, stabilizers in sectors:
        _assert_binary(logical_ops, "logical_ops")
        _assert_binary(stabilizers, "stabilizers")
        logical_matrix = _as_binary_matrix(logical_ops)
        stabilizer_matrix = _as_binary_matrix(stabilizers, num_cols=logical_matrix.shape[1])
        basis, labels = _get_nested_code_basis(logical_matrix, stabilizer_matrix)
        if labels.shape[1]:
            problems.append((basis, labels))
            continue

        # A supplied but trivial quotient has distance zero; an empty sector contributes its width.
        sector_distance = 0 if len(logical_matrix) else logical_matrix.shape[1]
        best = sector_distance if best is None else min(best, sector_distance)

    if best is not None and best <= cutoff:
        return best
    if not problems:
        assert best is not None
        return best
    return _get_distance_brouwer_zimmermann_many(
        problems,
        cutoff=cutoff,
        block_size=block_size,
        upper_bound=best,
    )


####################################################################################################
# helper methods

_MASK55 = np.uint64(0x5555555555555555)
_MASK33 = np.uint64(0x3333333333333333)
_MASK0F = np.uint64(0x0F0F0F0F0F0F0F0F)


class _InformationSet(NamedTuple):
    """Generators, labels, rank, and pivot columns for one systematic column block."""

    generators: npt.NDArray[np.uint8]
    labels: npt.NDArray[np.uint8] | None
    rank: int
    pivots: npt.NDArray[np.int_]


def _assert_binary(vectors: npt.ArrayLike, name: str) -> None:
    """Assert that the given vectors are binary.

    The enumeration below packs each row into the bits of ``uint64`` words and combines rows by
    bitwise XOR, so it is only correct over GF(2).  Non-binary input would otherwise be truncated
    silently and yield a wrong distance rather than an error.
    """
    order = getattr(type(vectors), "order", None)
    if order is not None and order != 2:
        raise ValueError(
            f"Distance calculations only support binary codes, but {name} is defined over"
            f" GF({order})"
        )
    array = np.asarray(vectors)
    if array.size and not np.all((array == 0) | (array == 1)):
        raise ValueError(
            f"Distance calculations only support binary codes, but {name} has entries other"
            " than 0 and 1"
        )


def _as_binary_matrix(
    vectors: npt.ArrayLike, *, num_cols: int | None = None
) -> npt.NDArray[np.uint8]:
    """Convert a collection of binary rows to a two-dimensional uint8 array."""
    matrix = np.asarray(vectors, dtype=np.uint8)
    if matrix.size == 0:
        if num_cols is None:
            if matrix.ndim != 2:
                raise ValueError("Cannot infer the width of an empty binary matrix")
            num_cols = matrix.shape[1]
        return np.empty((0, num_cols), dtype=np.uint8)
    matrix = np.atleast_2d(matrix)
    if matrix.ndim != 2:
        raise ValueError("Binary generators must be a two-dimensional matrix")
    if num_cols is not None and matrix.shape[1] != num_cols:
        raise ValueError(
            f"Binary matrices have incompatible widths {num_cols} and {matrix.shape[1]}"
        )
    return matrix


def _row_reduce_binary(
    matrix: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None = None,
    *,
    columns: npt.NDArray[np.int_] | None = None,
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8] | None,
    npt.NDArray[np.int_],
]:
    """Row-reduce a binary matrix over selected columns, transforming labels in parallel."""
    matrix = matrix.copy()
    labels = None if labels is None else labels.copy()
    columns = np.arange(matrix.shape[1], dtype=int) if columns is None else columns
    pivots: list[int] = []
    pivot_row = 0

    for col in columns:
        candidates = np.flatnonzero(matrix[pivot_row:, col])
        if not len(candidates):
            continue
        source_row = pivot_row + int(candidates[0])
        if source_row != pivot_row:
            matrix[[pivot_row, source_row]] = matrix[[source_row, pivot_row]]
            if labels is not None:
                labels[[pivot_row, source_row]] = labels[[source_row, pivot_row]]

        rows_to_clear = np.flatnonzero(matrix[:, col])
        rows_to_clear = rows_to_clear[rows_to_clear != pivot_row]
        matrix[rows_to_clear] ^= matrix[pivot_row]
        if labels is not None:
            labels[rows_to_clear] ^= labels[pivot_row]

        pivots.append(int(col))
        pivot_row += 1
        if pivot_row == len(matrix):
            break

    return matrix, labels, np.asarray(pivots, dtype=int)


def _get_independent_rows(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Return a row-reduced basis for the row space of a binary matrix."""
    reduced, _, pivots = _row_reduce_binary(matrix)
    return reduced[: len(pivots)]


def _get_nested_code_basis(
    logical_ops: npt.NDArray[np.uint8],
    stabilizers: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Build an adapted basis for ``span(stabilizers, logical_ops) / span(stabilizers)``."""
    stabilizer_basis = _get_independent_rows(stabilizers)
    stabilizer_pivots = [int(np.flatnonzero(row)[0]) for row in stabilizer_basis]
    reduced_logical_ops = logical_ops.copy()
    for row, pivot in zip(stabilizer_basis, stabilizer_pivots, strict=True):
        reduced_logical_ops[reduced_logical_ops[:, pivot] == 1] ^= row

    logical_quotient = _get_independent_rows(reduced_logical_ops)
    basis = np.vstack([stabilizer_basis, logical_quotient]).astype(np.uint8, copy=False)
    labels = np.zeros((len(basis), len(logical_quotient)), dtype=np.uint8)
    labels[len(stabilizer_basis) :] = np.eye(len(logical_quotient), dtype=np.uint8)
    return basis, labels


def _get_information_set_generators(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
) -> list[_InformationSet]:
    """Construct a large deterministic packing of disjoint information sets."""
    nonzero_columns = np.flatnonzero(np.any(basis, axis=0))
    best = _get_information_set_generators_for_columns(
        basis,
        labels,
        nonzero_columns,
    )
    # Each full set consumes one pivot column per basis row.
    max_full_sets = len(nonzero_columns) // len(basis)
    if sum(rank == len(basis) for _, _, rank, _ in best) == max_full_sets:
        return best

    # A fixed seed makes the bounded packing heuristic reproducible.
    rng = np.random.default_rng(0)
    for _ in range(8):
        candidate = _get_information_set_generators_for_columns(
            basis,
            labels,
            rng.permutation(nonzero_columns),
        )
        candidate_ranks = tuple(rank for _, _, rank, _ in candidate)
        best_ranks = tuple(rank for _, _, rank, _ in best)
        if candidate_ranks > best_ranks:
            best = candidate
        if sum(rank == len(basis) for _, _, rank, _ in best) == max_full_sets:
            break
    return best


def _get_information_set_generators_for_columns(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    columns: npt.NDArray[np.int_],
) -> list[_InformationSet]:
    """Construct disjoint information sets in a specified column order."""
    dimension = len(basis)
    active_columns = columns
    working = basis
    working_labels = labels
    information_sets: list[_InformationSet] = []

    while len(active_columns):
        working, working_labels, pivots = _row_reduce_binary(
            working, working_labels, columns=active_columns
        )
        rank = len(pivots)
        assert rank
        information_sets.append(_InformationSet(working, working_labels, rank, pivots))
        if rank < dimension:
            break

        pivot_set = set(pivots)
        active_columns = np.asarray(
            [col for col in active_columns if col not in pivot_set and np.any(working[:, col])],
            dtype=int,
        )

    return information_sets


@register_jitable(inline="always")
def _popcount_word(value: np.uint64) -> np.int64:
    """Count one unsigned word, returning a signed weight without overflow."""
    value = value - ((value >> np.uint64(1)) & _MASK55)
    value = (value & _MASK33) + ((value >> np.uint64(2)) & _MASK33)
    value = (value + (value >> np.uint64(4))) & _MASK0F
    value = value + (value >> np.uint64(8))
    value = value + (value >> np.uint64(16))
    value = value + (value >> np.uint64(32))
    return np.int64(value & np.uint64(127))


@njit(boundscheck=True, cache=True)
def _scan_brouwer_zimmermann(
    generators: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    ranks: npt.NDArray[np.int64],
    widths: npt.NDArray[np.int64],
    label_widths: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
    best: int,
    cutoff: int,
    batch_size: int,
) -> tuple[int, int, int]:
    """Scan one bounded chunk of coefficient supports for Brouwer-Zimmermann search.

    Args:
        generators: Packed information-set rows after pivot columns have been deleted.
        labels: Packed quotient labels whose nonzero XOR marks eligible candidates.
        ranks: Rank of each information-set block before pivot columns were deleted.
        widths: Number of packed generator words used by each information-set block.
        label_widths: Number of packed label words used by each information-set block.
        support: Mutable row indices for the current coefficient support.
        prefixes: Mutable prefix XORs for every support depth and information-set block.
        depth: Current support-tree depth, or ``-1`` when the coefficient layer is complete.
        best: Current upper bound; only strictly lighter eligible candidates replace it.
        cutoff: Weight threshold that stops this chunk once ``best <= cutoff``.
        batch_size: Maximum number of complete supports visited by this chunk.

    Returns:
        Updated ``best``, updated ``depth``, and number of complete supports visited.
    """
    dimension = generators.shape[1]
    weight = len(support)
    visited = 0
    while depth >= 0:
        # Backtrack once the row at this depth leaves too few rows to complete the support.
        if support[depth] > dimension - weight + depth:
            depth -= 1
            if depth >= 0:
                support[depth] += 1
            continue

        # Extend the prefix XOR by the row selected at this depth, in every information set.
        row = support[depth]
        for set_index in range(len(ranks)):
            for word in range(widths[set_index]):
                prefixes[depth + 1, set_index, word] = (
                    prefixes[depth, set_index, word] ^ generators[set_index, row, word]
                )

        # Descend until every row of the support has been selected.
        if depth + 1 < weight:
            depth += 1
            support[depth] = row + 1
            continue

        # Evaluate the completed support in each information set.
        for set_index in range(len(ranks)):
            # Each selected row whose pivot column was deleted contributes one to the Hamming
            # weight; packed words hold the remaining columns.
            candidate_weight = np.int64(weight) if ranks[set_index] == dimension else np.int64(0)
            if ranks[set_index] != dimension:
                for selected in range(weight):
                    if support[selected] < ranks[set_index]:
                        candidate_weight += 1
            for word in range(widths[set_index]):
                candidate_weight += _popcount_word(prefixes[weight, set_index, word])
            if candidate_weight >= best:
                continue

            # A candidate is eligible when no subcode is excluded or its quotient label is nonzero.
            eligible = label_widths[set_index] == 0
            for word in range(label_widths[set_index]):
                combined = labels[set_index, support[0], word]
                for selected in range(1, weight):
                    combined ^= labels[set_index, support[selected], word]
                if combined:
                    eligible = True
                    break
            if eligible:
                best = int(candidate_weight)

        # Move to the next support, returning at the cutoff or at the end of this chunk.
        visited += 1
        support[depth] += 1
        if best <= cutoff or visited >= batch_size:
            return best, depth, visited
    return best, depth, visited


def _get_packed_row_weights(
    rows: npt.NDArray[np.uint64],
    weight_func: Callable[..., npt.NDArray[np.uint64]],
) -> npt.NDArray[np.uint64]:
    """Compute Hamming weights of packed binary rows without narrow-integer overflow."""
    word_weights = np.asarray(weight_func(rows), dtype=np.uint64)
    return word_weights.sum(axis=-1, dtype=np.uint64)


@dataclass
class _BrouwerZimmermannSearch:
    """State for one resumable Brouwer-Zimmermann search.

    ``max_coverage`` bounds the number of pivot blocks containing any physical coordinate.
    Disjoint information sets have coverage one.
    """

    generators: npt.NDArray[np.uint64]
    labels: npt.NDArray[np.uint64]
    ranks: npt.NDArray[np.int64]
    widths: npt.NDArray[np.int64]
    label_widths: npt.NDArray[np.int64]
    cutoff: int
    batch_size: int
    weight_divisor: int
    best: int
    lower_bound: int
    next_weight: int = 1
    max_coverage: int = 1

    @property
    def dimension(self) -> int:
        """Number of basis rows searched."""
        return self.generators.shape[1]

    @property
    def finished(self) -> bool:
        """Whether this sector cannot improve the shared upper bound."""
        return (
            self.best <= self.cutoff
            or self.lower_bound >= self.best
            or self.next_weight > self.dimension
        )

    def update_upper_bound(self, upper_bound: int) -> None:
        """Import a witness found by another search."""
        self.best = min(self.best, upper_bound)

    def advance(self) -> int:
        """Search one coefficient weight and return the best shared upper bound."""
        if self.finished:
            return self.best

        # Start a fresh enumeration of all supports with this coefficient weight.
        weight = self.next_weight
        support = np.zeros(weight, dtype=np.int64)
        prefixes = np.zeros(
            (weight + 1, len(self.ranks), self.generators.shape[2]), dtype=np.uint64
        )
        depth = 0
        # Any negative Python cutoff disables early exit; keep its native value in int64 range.
        cutoff = max(-1, self.cutoff)

        # Scan in chunks, returning early once the upper bound reaches the cutoff.
        while depth >= 0:
            self.best, depth, _ = _scan_brouwer_zimmermann(
                self.generators,
                self.labels,
                self.ranks,
                self.widths,
                self.label_widths,
                support,
                prefixes,
                depth,
                self.best,
                cutoff,
                self.batch_size,
            )
            if self.best <= self.cutoff:
                return self.best

        # The layer is complete, so update the lower bound and move on to the next weight.
        self.lower_bound = _get_brouwer_zimmermann_lower_bound(
            self.dimension,
            self.ranks.tolist(),
            completed_weight=weight,
            weight_divisor=self.weight_divisor,
            max_coverage=self.max_coverage,
        )
        self.next_weight += 1
        return self.best


def _get_brouwer_zimmermann_lower_bound(
    dimension: int,
    ranks: list[int],
    *,
    completed_weight: int,
    weight_divisor: int,
    max_coverage: int = 1,
) -> int:
    """Bound the Hamming weight of eligible words not covered by the completed layers.

    Every systematic basis must have been searched through ``completed_weight``.
    The bound applies only to unsearched words, not to witnesses already found.
    """
    # In an unsearched word, a rank-r pivot block has at least
    # max(0, completed_weight + 1 - (dimension - r)) nonzero pivot columns.
    lower_bound = sum(max(0, completed_weight + 1 - (dimension - rank)) for rank in ranks)

    # Each physical coordinate contributes to at most max_coverage blocks, so divide and round up.
    lower_bound = (lower_bound + max_coverage - 1) // max_coverage
    if weight_divisor > 1:
        # Every candidate Hamming weight is a multiple of weight_divisor, so the bound can round up.
        lower_bound += (-lower_bound) % weight_divisor
    return lower_bound


def _get_brouwer_zimmermann_initial_upper_bound(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
) -> int:
    """Return the lightest eligible row in a nested-code basis."""
    weight_func, _ = _get_hamming_weight_fn()
    eligible = np.ones(len(basis), dtype=bool) if labels is None else np.any(labels, axis=1)
    packed_basis = _rows_to_ints(basis[eligible], dtype=np.uint64)
    return int(_get_packed_row_weights(packed_basis, weight_func).min())


def _select_brouwer_zimmermann_information_sets(
    information_sets: Sequence[_InformationSet],
    *,
    dimension: int,
    upper_bound: int,
    weight_divisor: int,
) -> list[_InformationSet]:
    """Retain partial sets that can strengthen certification of the current upper bound."""
    num_full_sets = sum(rank == dimension for _, _, rank, _ in information_sets)
    if not num_full_sets:
        return list(information_sets)
    full_set_ranks = [dimension] * num_full_sets
    full_set_certifying_weight = next(
        (
            completed_weight
            for completed_weight in range(dimension + 1)
            if _get_brouwer_zimmermann_lower_bound(
                dimension,
                full_set_ranks,
                completed_weight=completed_weight,
                weight_divisor=weight_divisor,
            )
            >= upper_bound
        ),
        dimension + 1,
    )
    # A dropped partial set adds zero lower-bound units until full sets certify the upper bound.
    return [
        information_set
        for information_set in information_sets
        if information_set[2] == dimension
        or dimension - information_set[2] < full_set_certifying_weight
    ]


def _get_overlapping_information_sets(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    information_sets: list[_InformationSet],
    *,
    upper_bound: int,
    weight_divisor: int,
    max_sets: int = 8,
    minimum_work: int = 100_000,
) -> tuple[list[_InformationSet], int] | None:
    """Find a cheaper full-set family and its maximum coordinate coverage.

    The input contains disjoint full or partial sets for one independent binary basis.
    Proposed families contain only full sets, with quotient labels transformed alongside rows.
    Return ``None`` to retain the input family when no cheaper replacement is found.
    """
    # Start from full blocks; skip complete packings and families already at the size limit.
    dimension = len(basis)
    full_sets = [item for item in information_sets if item.rank == dimension]
    columns = np.flatnonzero(np.any(basis, axis=0))
    if not full_sets or len(full_sets) >= max_sets or len(full_sets) * dimension == len(columns):
        return None

    # Avoid construction for cheap searches, using the same work estimate as exhaustive dispatch.
    best_work = _get_brouwer_zimmermann_work(
        dimension,
        [item.rank for item in information_sets],
        upper_bound=upper_bound,
        weight_divisor=weight_divisor,
    )
    if best_work < minimum_work:
        return None

    # Even ideal families repeat some coordinate ceil(count * dimension / width) times.
    optimistic_work = min(
        _get_brouwer_zimmermann_work(
            dimension,
            [dimension] * count,
            upper_bound=upper_bound,
            weight_divisor=weight_divisor,
            max_coverage=(count * dimension + len(columns) - 1) // len(columns),
        )
        for count in range(1, max_sets + 1)
    )
    if optimistic_work >= best_work:
        return None

    # Track pivot loads and distinct sets, with fixed-seed tie-breaking for reproducibility.
    loads = np.bincount(
        np.concatenate([item.pivots for item in full_sets]), minlength=basis.shape[1]
    )
    seen = {tuple(sorted(item.pivots.tolist())) for item in full_sets}
    rng = np.random.default_rng(0)
    best_sets = None
    best_coverage = 1
    for _ in range(2 * (max_sets - len(full_sets))):
        # Prioritize least-covered coordinates and transform quotient labels with the new basis.
        order = columns[np.lexsort((rng.random(len(columns)), loads[columns]))]
        generators, transformed_labels, pivots = _row_reduce_binary(basis, labels, columns=order)
        assert len(pivots) == dimension

        # Skip duplicate pivot sets to keep the bounded search focused on new bases.
        signature = tuple(sorted(pivots.tolist()))
        if signature in seen:
            continue
        seen.add(signature)
        full_sets.append(_InformationSet(generators, transformed_labels, dimension, pivots))
        loads[pivots] += 1
        max_coverage = int(loads.max())

        # Retain the cheapest prefix, stopping once no permitted family could improve its work.
        work = _get_brouwer_zimmermann_work(
            dimension,
            [dimension] * len(full_sets),
            upper_bound=upper_bound,
            weight_divisor=weight_divisor,
            max_coverage=max_coverage,
        )
        if work < best_work:
            best_sets, best_coverage, best_work = full_sets.copy(), max_coverage, work
        if len(full_sets) == max_sets or best_work <= optimistic_work:
            break
    return None if best_sets is None else (best_sets, best_coverage)


def _prepare_brouwer_zimmermann_search(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    *,
    cutoff: int,
    block_size: int,
    weight_divisor: int,
    upper_bound: int,
    initial_upper_bound: int | None = None,
) -> int | _BrouwerZimmermannSearch:
    """Prepare one BZ search, or return immediately when another route is cheaper."""
    dimension = len(basis)
    eligible = np.ones(dimension, dtype=bool) if labels is None else np.any(labels, axis=1)

    def brute_force_distance() -> int:
        distance = _get_distance_quantum_brute_force(
            basis[eligible],
            basis[~eligible],
            cutoff=cutoff,
            block_size=block_size,
            homogeneous=True,
        )
        return min(best, distance)

    best = min(
        upper_bound,
        (
            _get_brouwer_zimmermann_initial_upper_bound(basis, labels)
            if initial_upper_bound is None
            else initial_upper_bound
        ),
    )
    if best <= cutoff:
        return best

    if _exhaustive_is_cheaper(dimension, basis.shape[1]):
        return brute_force_distance()

    # XOR preserves even parity, so this entire ambient space has even Hamming weight.
    if np.all(np.count_nonzero(basis, axis=1) % 2 == 0):
        weight_divisor = math.lcm(weight_divisor, 2)

    information_sets = _get_information_set_generators(basis, labels)
    # Systematic rows are all coefficient-weight-one candidates.
    for generators, set_labels, _, _ in information_sets:
        best = min(best, _get_brouwer_zimmermann_initial_upper_bound(generators, set_labels))
    if best <= cutoff:
        return best

    information_sets = _select_brouwer_zimmermann_information_sets(
        information_sets,
        dimension=dimension,
        upper_bound=best,
        weight_divisor=weight_divisor,
    )
    ranks = [rank for _, _, rank, _ in information_sets]
    completed_weight = 1 if information_sets else 0
    lower_bound = _get_brouwer_zimmermann_lower_bound(
        dimension,
        ranks,
        completed_weight=completed_weight,
        weight_divisor=weight_divisor,
    )
    if lower_bound >= best:
        return best

    # Replace disjoint blocks only when overlap predicts less certification work.
    max_coverage = 1
    overlapping = _get_overlapping_information_sets(
        basis,
        labels,
        information_sets,
        upper_bound=best,
        weight_divisor=weight_divisor,
    )
    if overlapping is not None:
        information_sets, max_coverage = overlapping

        # New systematic rows can expose lighter witnesses without further enumeration.
        for information_set in information_sets:
            best = min(
                best,
                _get_brouwer_zimmermann_initial_upper_bound(
                    information_set.generators, information_set.labels
                ),
            )
        if best <= cutoff:
            return best

        # Recompute the local certificate using the replacement family's actual coverage.
        ranks = [item.rank for item in information_sets]
        lower_bound = _get_brouwer_zimmermann_lower_bound(
            dimension,
            ranks,
            completed_weight=completed_weight,
            weight_divisor=weight_divisor,
            max_coverage=max_coverage,
        )
        if lower_bound >= best:
            return best

    if _brute_force_is_cheaper(
        dimension=dimension,
        num_logical_rows=int(np.count_nonzero(eligible)),
        ranks=ranks,
        upper_bound=best,
        weight_divisor=weight_divisor,
        max_coverage=max_coverage,
    ):
        return brute_force_distance()

    # Remove systematic pivots; advance() restores their known weight arithmetically.
    packed_sets = [
        (
            _rows_to_ints(
                np.delete(information_set.generators, information_set.pivots, axis=1),
                dtype=np.uint64,
            ),
            None
            if information_set.labels is None
            else _rows_to_ints(information_set.labels, dtype=np.uint64),
            information_set.rank,
        )
        for information_set in information_sets
    ]

    # Zero-pad every set to common word widths, so the Numba kernel receives single arrays.
    widths = np.array([rows.shape[1] for rows, _, _ in packed_sets], dtype=np.int64)
    label_widths = np.array(
        [0 if labels is None else labels.shape[1] for _, labels, _ in packed_sets],
        dtype=np.int64,
    )
    packed_generators = np.zeros(
        (len(ranks), dimension, int(widths.max(initial=0))), dtype=np.uint64
    )
    packed_labels = np.zeros(
        (len(ranks), dimension, int(label_widths.max(initial=0))), dtype=np.uint64
    )
    for index, (rows, packed_set_labels, _) in enumerate(packed_sets):
        packed_generators[index, :, : widths[index]] = rows
        if packed_set_labels is not None:
            packed_labels[index, :, : label_widths[index]] = packed_set_labels

    return _BrouwerZimmermannSearch(
        packed_generators,
        packed_labels,
        np.array(ranks, dtype=np.int64),
        widths,
        label_widths,
        cutoff,
        1 << min(block_size, 15),
        weight_divisor,
        best,
        lower_bound,
        completed_weight + 1,
        max_coverage,
    )


def _get_distance_brouwer_zimmermann_many(
    problems: Sequence[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]],
    *,
    cutoff: int,
    block_size: int,
    weight_divisor: int = 1,
    upper_bound: int | None = None,
) -> int:
    """Advance nested-code searches round-robin while sharing their best upper bound."""
    initial_upper_bounds = [
        _get_brouwer_zimmermann_initial_upper_bound(basis, labels) for basis, labels in problems
    ]
    best = min(
        *initial_upper_bounds,
        math.inf if upper_bound is None else upper_bound,
    )
    searches: list[_BrouwerZimmermannSearch] = []
    for (basis, labels), initial_upper_bound in zip(problems, initial_upper_bounds, strict=True):
        prepared = _prepare_brouwer_zimmermann_search(
            basis,
            labels,
            cutoff=cutoff,
            block_size=block_size,
            weight_divisor=weight_divisor,
            upper_bound=int(best),
            initial_upper_bound=initial_upper_bound,
        )
        if isinstance(prepared, int):
            best = min(best, prepared)
        else:
            searches.append(prepared)

    while searches:
        advanced = False
        for search in searches:
            # Share witnesses between sectors, but keep each lower certificate independent.
            search.update_upper_bound(int(best))
            if search.finished:
                continue
            best = min(best, search.advance())
            advanced = True
            if best <= cutoff:
                return int(best)
        if not advanced:
            break

    return int(best)


def _get_distance_brouwer_zimmermann(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    *,
    cutoff: int,
    block_size: int,
    weight_divisor: int = 1,
) -> int:
    """Compute an exact nested-code distance with the Brouwer-Zimmermann algorithm.

    The information-set lower bound follows Algorithm 1 of https://arxiv.org/abs/1603.06757.
    Nonzero ``labels`` identify rows outside an excluded subcode; transforming them alongside the
    generators makes the upper-bound search exact for logical operators modulo stabilizers.
    """
    return _get_distance_brouwer_zimmermann_many(
        [(basis, labels)],
        cutoff=cutoff,
        block_size=block_size,
        weight_divisor=weight_divisor,
    )


def _exhaustive_is_cheaper(dimension: int, length: int) -> bool:
    """Estimate whether enumerating every codeword is cheaper than building information sets.

    Setup costs dominate the Brouwer-Zimmermann search for dimensions up to 20.  For long,
    low-dimensional codes, byte-wise elimination for each information set can cost more than packed
    ``uint64`` enumeration of all ``2**dimension`` codewords.
    """
    return dimension <= 20 or (dimension < 63 and 1 << dimension <= 128 * dimension * length)


def _brute_force_is_cheaper(
    *,
    dimension: int,
    num_logical_rows: int,
    ranks: list[int],
    upper_bound: int,
    weight_divisor: int,
    max_coverage: int = 1,
) -> bool:
    """Estimate whether exhaustive nested-code enumeration will outperform BZ.

    The BZ estimate pessimistically assumes that the search must certify the current upper bound,
    whereas lighter witnesses usually end the search sooner.  Although the vectorized exhaustive
    kernel is a few times cheaper per candidate than the serial Numba BZ kernel, whole-search
    timings near this boundary favor BZ until the exhaustive count reaches roughly the estimated BZ
    count, so the two counts are compared directly.
    """
    bz_candidates = _get_brouwer_zimmermann_work(
        dimension,
        ranks,
        upper_bound=upper_bound,
        weight_divisor=weight_divisor,
        max_coverage=max_coverage,
    )
    exhaustive_candidates = ((1 << num_logical_rows) - 1) << (dimension - num_logical_rows)
    return exhaustive_candidates <= bz_candidates


def _get_brouwer_zimmermann_work(
    dimension: int,
    ranks: list[int],
    *,
    upper_bound: int,
    weight_divisor: int,
    max_coverage: int = 1,
) -> int:
    """Estimate candidate/set work needed to certify the current upper bound.

    This pessimistic estimate includes systematic rows (coefficient weight one).
    It assumes no lighter witness is found before the lower certificate reaches the upper bound.
    """
    candidates = 0
    for weight in range(1, dimension + 1):
        candidates += len(ranks) * math.comb(dimension, weight)
        if (
            _get_brouwer_zimmermann_lower_bound(
                dimension,
                ranks,
                completed_weight=weight,
                weight_divisor=weight_divisor,
                max_coverage=max_coverage,
            )
            >= upper_bound
        ):
            break
    return candidates


def _symplectic_to_hamming(vectors: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Map ``(X | Z)`` to ``(X | Z | X xor Z)``, doubling symplectic weight.

    This is the ``Saved_isometry`` reduction from https://arxiv.org/abs/2408.10743.
    """
    half_width = vectors.shape[1] // 2
    vectors_x = vectors[:, :half_width]
    vectors_z = vectors[:, half_width:]
    return np.hstack([vectors_x, vectors_z, vectors_x ^ vectors_z])


####################################################################################################
# exact distance via brute-force enumeration over logical-op and stabilizer combinations


def _get_distance_quantum_brute_force(
    logical_ops: npt.ArrayLike,
    stabilizers: npt.ArrayLike,
    *,
    cutoff: int,
    block_size: int,
    homogeneous: bool,
) -> int:
    """Brute-force binary quantum distance implementation."""
    num_bits = np.shape(logical_ops)[-1]

    if homogeneous:
        weight_func, num_buffers = _get_hamming_weight_fn()
    else:
        weight_func, num_buffers = _get_symplectic_weight_fn()

        logical_ops = _riffle(logical_ops)
        stabilizers = _riffle(stabilizers)

    int_logical_ops = _rows_to_ints(logical_ops, dtype=np.uint64)
    int_stabilizers = _rows_to_ints(stabilizers, dtype=np.uint64)
    num_stabilizers = len(int_stabilizers)

    # Number of generators to include in the operational array.  Most calculations will then be
    # vectorized over ``2**block_size`` values.  Clamp at 0: when the packed word-count per row
    # exceeds ``block_size + 1`` the first term goes negative, which would make a negative slice and
    # build an uncapped ``2**(S-k)`` array, defeating the block_size cap.
    num_vectorized_ops = max(
        0,
        min(
            block_size + 1 - int_logical_ops.shape[-1],
            len(int_logical_ops) + len(int_stabilizers),
        ),
    )

    # Vectorize all combinations of first `num_vectorized_ops` stabilizers
    array = np.zeros((1, int_logical_ops.shape[-1]), dtype=np.uint64)
    for op in int_stabilizers[:num_vectorized_ops]:
        array = np.vstack([array, array ^ op])

    if num_vectorized_ops > num_stabilizers:
        # fill out block with products of some logical ops
        for op in int_logical_ops[: num_vectorized_ops - num_stabilizers]:
            array = np.vstack([array, array ^ op])

        int_logical_ops = int_logical_ops[num_vectorized_ops - num_stabilizers :]

    int_stabilizers = int_stabilizers[num_vectorized_ops:]

    # Everything below will run much faster if we use Fortran-style ordering
    arrayf = np.asarray(array, order="F", dtype=np.uint64)

    # out is the uint64 buffer for the in-loop weight_func calls: passing out=out keeps
    # np.bitwise_count's uint8 result cast to uint64, avoiding the weight-reduction overflow for
    # codes with >= 256 columns (the pre-loop call below casts explicitly for the same reason).
    out = np.empty_like(arrayf)
    bufs = [np.empty_like(arrayf) for _ in range(num_buffers)]

    # Min weight of the block containing logical ops.  Cast to uint64 because np.bitwise_count
    # (numpy >= 2) returns uint8, which would overflow the reduction below for >= 256 columns.
    weights = np.asarray(weight_func(arrayf[2**num_stabilizers :]), dtype=np.uint64)
    min_weight = _inplace_rowsum(weights).min(initial=num_bits)
    if min_weight <= cutoff:
        return int(min_weight)

    # Sweep over every remaining logical-op combination and, nested, every remaining stabilizer
    # combination, on top of the vectorized block above.  The sweep uses a reflected Gray code:
    # successive Gray-code words differ in one bit, whose position is the trailing-zero count of the
    # step index.  Each step therefore flips a single operator into/out of the running XOR
    # (``arrayf``) with one ^=, so the loop walks the 2**k - 1 nonzero combinations of the remaining
    # operators (the empty combination is the pre-loop state) without rebuilding any.
    # ``min_weight`` tracks the lightest operator seen; ``cutoff`` lets the sweep stop once that
    # bound is reached (see Returns).
    for li in range(1, 2 ** len(int_logical_ops)):
        arrayf ^= int_logical_ops[_count_trailing_zeros(li)]
        weights = weight_func(arrayf, *bufs, out=out)
        min_weight = _inplace_rowsum(weights).min(initial=min_weight)
        if min_weight <= cutoff:
            return int(min_weight)

        for si in range(1, 2 ** len(int_stabilizers)):
            arrayf ^= int_stabilizers[_count_trailing_zeros(si)]
            weights = weight_func(arrayf, *bufs, out=out)
            min_weight = _inplace_rowsum(weights).min(initial=min_weight)
            if min_weight <= cutoff:
                return int(min_weight)

    return int(min_weight)


####################################################################################################
# weight functions (Hamming and symplectic popcount) and backend selection


def _get_hamming_weight_fn() -> tuple[Callable[..., npt.NDArray[np.uint64]], int]:
    if getattr(np, "bitwise_count", None) is not None:
        weight_fn = np.bitwise_count
        return weight_fn, 0

    return _hamming_weight, 1


def _get_symplectic_weight_fn() -> tuple[Callable[..., npt.NDArray[np.uint64]], int]:
    if getattr(np, "bitwise_count", None) is not None:
        np_bitwise_count = np.bitwise_count

        def weight_fn(
            arr: npt.NDArray[np.uint64],
            buf: npt.NDArray[np.uint64] | None = None,
            out: npt.NDArray[np.uint64] | None = None,
        ) -> npt.NDArray[np.uint64]:
            """Symplectic weight of an integer."""
            buf = np.right_shift(arr, 1, out=buf)
            buf |= arr
            buf &= _MASK55
            return np_bitwise_count(buf, out=out)

        return weight_fn, 1

    return _symplectic_weight, 1


####################################################################################################
# bit-packing and array helpers


def _count_trailing_zeros(val: int) -> int:
    """Returns the position of the least significant 1 in the binary representation of `val`."""
    return (val & -val).bit_length() - 1


def _inplace_rowsum(arr: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Destructively compute ``arr.sum(-1)``, placing the result in the first column of ``arr``.

    When complete, the returned sum will be stored in ``arr[..., 0]``, while other entries in
    ``arr[..., 1:]`` will be left in indeterminate states.  This permits a faster sum
    implementation.
    """
    width = arr.shape[-1]
    while width > 1:
        split = width // 2
        arr[..., :split] += arr[..., width - split : width]
        width -= split

    return arr[..., 0]


def _rows_to_ints(
    array: npt.ArrayLike, dtype: npt.DTypeLike = np.uint64, axis: int = -1
) -> npt.NDArray[np.uint64]:
    """Pack rows of a binary array into rows of the given integral type."""
    array = np.asarray(array, dtype=dtype)
    tsize = array.itemsize * 8

    if array.size == 0:
        num_words = int(np.ceil(array.shape[-1] / tsize))
        return np.empty((*array.shape[:-1], num_words), dtype=dtype)

    def _to_int(bits: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
        """Pack `bits` into a single integer (of type `dtype`)."""
        return (bits << np.arange(len(bits) - 1, -1, -1, dtype=dtype)).sum(dtype=dtype)

    def _to_ints(bits: npt.NDArray[np.uint64]) -> list[npt.NDArray[np.uint64]]:
        """Pack a single row of bits into a row of integers."""
        return [_to_int(bits[i : i + tsize]) for i in range(0, np.shape(bits)[-1], tsize)]

    return np.apply_along_axis(_to_ints, axis, array)


def _riffle(array: npt.ArrayLike) -> npt.ArrayLike:
    """'Riffle' Pauli strings, putting X and Z support bits for each qubit next to each other."""
    arr = np.asarray(array)
    num_bits = arr.shape[-1]
    assert num_bits % 2 == 0
    if arr.size == 0:
        return np.empty((0, num_bits), dtype=arr.dtype)  # nothing to riffle (e.g. no stabilizers)
    return np.reshape(arr, (-1, 2, num_bits // 2)).transpose(0, 2, 1).reshape(-1, num_bits)


####################################################################################################
# numpy < 2.0 does not provide np.bitwise_count; the methods below provide a fallback

_MASK01 = np.uint64(0x0101010101010101)


def _hamming_weight(
    arr: npt.NDArray[np.uint64],
    buf: npt.NDArray[np.uint64] | None = None,
    out: npt.NDArray[np.uint64] | None = None,
) -> npt.NDArray[np.uint64]:
    """Somewhat efficient (vectorized) Hamming weight calculation.

    Assumes 64-bit uints.  For `numpy >= 2.0.0`, it's generally better to use `np.bitwise_count`
    (which uses processors' builtin `popcnt` instruction).  Unfortunately this isn't available for
    numpy < 2.0.0.

    The mask-and-shift steps are the classic SWAR (SIMD-within-a-register) popcount; see
    https://en.wikipedia.org/wiki/Hamming_weight.
    """
    out = np.right_shift(arr, 1, out=out)
    out &= _MASK55
    out = np.subtract(arr, out, out=out)

    buf = np.right_shift(out, 2, out=buf)
    buf &= _MASK33
    out &= _MASK33
    out += buf

    buf = np.right_shift(out, 4, out=buf)
    out += buf
    out &= _MASK0F

    out = np.multiply(out, _MASK01, out=out)
    out >>= np.uint64(56)
    return out


def _symplectic_weight(
    arr: npt.NDArray[np.uint64],
    buf: npt.NDArray[np.uint64] | None = None,
    out: npt.NDArray[np.uint64] | None = None,
) -> npt.NDArray[np.uint64]:
    """Somewhat efficient (vectorized) symplectic weight calculation.

    Assumes 64-bit uints.  This function is equivalent to (but slightly more efficient than) the
    expression ``_hamming_weight((arr | (arr >> 1)) & 0x5555555555555555, buf=buf, out=out)``.
    """
    out = np.right_shift(arr, 1, out=out)
    out |= arr
    out &= _MASK55

    buf = np.right_shift(out, 2, out=buf)
    buf &= _MASK33
    out &= _MASK33
    out += buf

    buf = np.right_shift(out, 4, out=buf)
    out += buf
    out &= _MASK0F

    out *= _MASK01
    out >>= np.uint64(56)
    return out
