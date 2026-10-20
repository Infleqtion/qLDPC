# SPDX-License-Identifier: Apache-2.0

"""Exact binary BZ searches using disjoint or overlapping information sets.

The disjoint/rank-deficient certificates follow `Algorithm 994
<https://arxiv.org/abs/1603.06757>`_; the symplectic reduction follows the
`quantum-code adaptation <https://arxiv.org/abs/2408.10743>`_. The overlapping-set
certificate and implementation invariants are derived in ``brouwer_zimmermann.md``
beside this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from numba import njit

from ._binary import (
    _get_hamming_weight_fn,
    _get_packed_row_weights,
    _pack_bits,
    _popcount_word,
    _row_reduce_binary,
)
from .brute_force import (
    _get_distance_quantum_brute_force,
)


class _InformationSet(NamedTuple):
    """Generators, labels, rank, and pivot columns for one systematic column block."""

    generators: npt.NDArray[np.uint8]
    labels: npt.NDArray[np.uint8] | None
    rank: int
    pivots: npt.NDArray[np.int_]


####################################################################################################
# search entry points


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

    The disjoint and rank-deficient bound follows Algorithm 1 of https://arxiv.org/abs/1603.06757.
    The overlapping-set extension divides pivot occupancy by the maximum coordinate coverage.
    Nonzero ``labels`` identify rows outside an excluded subcode; transforming them alongside the
    generators makes the upper-bound search exact for logical operators modulo stabilizers.
    """
    return _get_distance_brouwer_zimmermann_many(
        [(basis, labels)],
        cutoff=cutoff,
        block_size=block_size,
        weight_divisor=weight_divisor,
    )


####################################################################################################
# helper methods


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


@dataclass
class _BrouwerZimmermannSearch:
    """State for one resumable Brouwer-Zimmermann search.

    ``max_coverage`` bounds the number of pivot blocks containing any Hamming coordinate.
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
    """Bound the Hamming weight of unsearched eligible words.

    Search every systematic basis through ``completed_weight`` before using this bound.
    It does not bound the weights of witnesses already found.
    """
    # Rank deficiency limits how many selected rows can have zero pivot support.
    lower_bound = sum(max(0, completed_weight + 1 - (dimension - rank)) for rank in ranks)

    # Correct for repeated Hamming coordinates by dividing occupancy and rounding up.
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
    packed_basis = _pack_bits(basis[eligible])
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
    """Look for a lower-work replacement for disjoint information sets.

    ``information_sets`` contains full or partial sets for one independent binary basis.
    A replacement contains only full sets, with quotient labels transformed alongside rows.
    Return a lower-work family and its maximum Hamming-coordinate coverage, or ``None``.
    """
    # Start from full blocks; skip complete packings and families already at the size limit.
    dimension = len(basis)
    full_sets = [item for item in information_sets if item.rank == dimension]
    columns = np.flatnonzero(np.any(basis, axis=0))
    if not full_sets or len(full_sets) >= max_sets or len(full_sets) * dimension == len(columns):
        return None

    # Skip family construction when the existing search is already cheap.
    best_work = _get_brouwer_zimmermann_work(
        dimension,
        [item.rank for item in information_sets],
        upper_bound=upper_bound,
        weight_divisor=weight_divisor,
    )
    if best_work < minimum_work:
        return None

    # With m full sets, some column has coverage at least ceil(m * dimension / len(columns)).
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

    # Track coverage and distinct sets; fixed-seed tie-breaking keeps choices reproducible.
    loads = np.bincount(
        np.concatenate([item.pivots for item in full_sets]), minlength=basis.shape[1]
    )
    seen = {tuple(sorted(item.pivots.tolist())) for item in full_sets}
    rng = np.random.default_rng(0)
    best_sets = None
    best_coverage = 1
    for _ in range(2 * (max_sets - len(full_sets))):
        # Prefer less-used columns; transform quotient labels with each candidate basis.
        order = columns[np.lexsort((rng.random(len(columns)), loads[columns]))]
        generators, transformed_labels, pivots = _row_reduce_binary(basis, labels, columns=order)
        assert len(pivots) == dimension

        # Do not add duplicate sets; their row reductions still count against the attempt limit.
        signature = tuple(sorted(pivots.tolist()))
        if signature in seen:
            continue
        seen.add(signature)
        full_sets.append(_InformationSet(generators, transformed_labels, dimension, pivots))
        loads[pivots] += 1
        max_coverage = int(loads.max())

        # Keep improvements; stop at the set limit or when the optimistic work bound is attained.
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

        # Check the replacement rows for lighter witnesses before scanning combinations.
        for information_set in information_sets:
            best = min(
                best,
                _get_brouwer_zimmermann_initial_upper_bound(
                    information_set.generators, information_set.labels
                ),
            )
        if best <= cutoff:
            return best

        # Recompute the bound for the replacement ranks and coordinate multiplicity.
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
            _pack_bits(np.delete(information_set.generators, information_set.pivots, axis=1)),
            None if information_set.labels is None else _pack_bits(information_set.labels),
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
