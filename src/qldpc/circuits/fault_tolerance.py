# SPDX-License-Identifier: Apache-2.0

"""Exact bounded-spread certificates for pure Clifford resource-state preparation."""

from __future__ import annotations

import itertools
import math
import numbers
import time
from bisect import bisect_left
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import stim

from qldpc.codes.distance import BoundedKernelSearch, PauliCosetSearch

from .common import get_pauli_product_measurements
from .encoding import get_state_stabilizers


@dataclass(frozen=True)
class StatePrepSpreadCounterexample:
    """An accepted original-DEM pattern violating the stated bounded-spread criterion.

    ``fault_indices`` index error instructions (not all instructions) in the flattened,
    undecomposed diagnostic DEM returned in the certificate.  ``output_syndrome`` is ordered
    like ``target_stabilizers``.  ``effective_weight_lower_bound`` is proved, not estimated.
    ``output_pauli`` is a syndrome representative in the caller's output-qubit frame, not
    necessarily a minimum-weight representative or the propagated physical error.
    This need not be the smallest violating fault order.
    """

    fault_indices: tuple[int, ...]
    output_syndrome: tuple[int, ...]
    effective_weight_lower_bound: int
    output_pauli: str

    @property
    def fault_count(self) -> int:
        """Number of unit-cost original DEM mechanisms in the witness."""
        return len(self.fault_indices)


@dataclass
class StatePrepSpreadStatistics:
    """Deterministic work counters and wall-clock setup/search costs (in seconds).

    For independent-prefix search, ``generated`` counts attempted prefix extensions plus emitted
    closing supports.  ``prefix_extensions`` includes attempts pruned by dependence, whereas
    ``independent_prefixes`` counts prefixes that survive.  Neither counts full fault combinations
    through the original budget.
    """

    mechanisms: int = 0
    distinct_symptoms: int = 0
    flag_rank: int = 0
    effective_budget: int = 0
    generated: int = 0
    bounded_candidates: int = 0
    minimal_candidates: int = 0
    syndrome_queries: int = 0
    cache_hits: int = 0
    inner_generated: int = 0
    inner_preparations: int = 0
    outer_ranks: tuple[int, ...] = ()
    inner_ranks: tuple[int, ...] = ()
    coefficient_depth: int = 0
    setup_seconds: float = 0
    search_seconds: float = 0
    search_method: Literal["singletons", "pairs", "brouwer_zimmermann", "independent_prefix"] = (
        "singletons"
    )
    prefix_extensions: int = 0
    independent_prefixes: int = 0


@dataclass(frozen=True)
class StatePrepSpreadCertificate:
    """Completed bounded-spread decision, including the exact model and witness frame.

    ``certified`` means every accepted pattern of one through ``max_faults`` unit-cost binary
    DEM mechanisms has effective output Pauli weight no greater than its own fault count.
    A negative decision always includes a counterexample.  No incomplete search returns a result.
    """

    certified: bool
    max_faults: int
    output_qubits: tuple[int, ...]
    target_stabilizers: tuple[str, ...]
    detector_error_model: stim.DetectorErrorModel
    counterexample: StatePrepSpreadCounterexample | None
    statistics: StatePrepSpreadStatistics


def certify_state_prep_spread(
    circuit: stim.Circuit,
    *,
    output_qubits: Sequence[int],
    max_faults: int,
    post_select: Collection[int] = (),
) -> StatePrepSpreadCertificate:
    r"""Exactly certify bounded Pauli spread of an already noisy Clifford preparation.

    For the complete pure output-state stabilizers, let ``S`` map DEM faults to their state
    syndrome, ``P`` map them to the selected actual rejection detectors, and ``mu(s)`` be the
    minimum physical Pauli weight with syndrome ``s``.  The certificate proves
    ``mu(S f) <= |f|`` whenever ``P f = 0`` and ``1 <= |f| <= max_faults``.
    A Y affects one physical qubit.  Logical stabilizers fixing the prepared state are included.

    Args:
        circuit: An already noisy Stim Clifford circuit, starting in Stim's all-zero state.
        output_qubits: Nonempty distinct physical output identities, in the desired Pauli frame.
        max_faults: Required positive integer unit-cost DEM fault budget.
        post_select: Indices of actual circuit detectors whose *events* must vanish relative to
            their deterministic ideal reference.  Appended output diagnostics are not rejected.

    Returns:
        A complete certificate or a concrete accepted counterexample with an exact effective-
        weight lower bound.  Witness indices refer to the returned undecomposed diagnostic DEM,
        before any internal joint-symptom compression.

    Raises:
        ValueError: For invalid inputs, mixed/incomplete outputs, nondeterministic annotations
            or diagnostics, or unsupported circuit-to-DEM conversion.

    The selected output must be a definite pure stabilizer state independent of random measurement
    branches.  All existing detector and observable annotations must be deterministic, including
    unselected ones (Stim's exact DEM validation enforces this conservative restriction).
    Intrinsically random post-selected branches are not supported.  Stabilizer signs and physical
    frames are preserved; the appended measurements are noiseless.

    The model is independent binary DEM mechanisms, not mutually exclusive alternatives at
    physical gate locations.  Stim's disjoint-channel probability approximation is enabled only
    to obtain those mechanisms; probabilities play no role in the combinatorial proof.
    Correlated mechanisms are never split.  This is not an exact sampling-distribution claim.

    Proof of the outer reduction: equal complete syndromes differ by a state stabilizer because
    the stabilizer space is maximal isotropic.  Minimum representative weight is subadditive,
    since product support is contained in the union.  Every accepted nonzero pattern partitions
    into support-disjoint minimal accepted patterns: remove a proper accepted subset and repeat.
    Thus it suffices to check minimal patterns.  For accepted support J, minimality is equivalent
    to ``nullity(P[:, J]) = 1``, so ``|J| <= rank(P) + 1``.  A flag-free column occurs only as a
    singleton; a rank-one nonzero flag map needs only pairs.  General dependencies use packed BZ
    coverage or canonical independent-prefix closure, independently of the nonlinear spread
    predicate.  A minimal sorted dependence has an independent prefix and a last column equal
    to the prefix XOR.  Conversely, an independent prefix plus that closing column has nullity
    one.  Requiring the closing index to exceed the prefix indices generates each minimal support
    exactly once.  Dependent prefixes cannot belong to any minimal support and are pruned.
    If the distinct nonzero flag types are independent, acceptance requires even multiplicity
    of each type, so all minimal patterns are equal-flag pairs regardless of the requested budget.

    High-rate independent flag types use the proved pair-only bound directly.  Otherwise a
    measured support-count crossover selects high-rate prefix closure when BZ coefficient
    supports sufficiently outnumber possible prefix extensions.  The disjoint-information-set BZ
    path is retained elsewhere.  The heuristic selects between two complete exact searches; it is
    not part of the proof and is not a universal runtime guarantee.

    Identical joint (flag, state-syndrome) columns may be merged: remove cancelling pairs from
    any pattern, preserving symptoms while decreasing weight.  A violation reduces to a
    representative-pattern violation; every such pattern lifts to original indices unchanged.
    All-zero joint columns can likewise be removed.  Only joint symptoms, never output alone,
    are used.  Inner decisions use the reusable exact
    :class:`qldpc.codes.distance.PauliCosetSearch`.

    This sufficient *resource-state* property composes with downstream gadgets that propagate
    each input Pauli of weight r and additional faults into errors within the appropriate
    r-plus-fault budget (or correct them).  No arbitrary downstream algorithm, acceptance
    probability, or necessary-and-sufficient fault-tolerance criterion is certified here.
    """
    start = time.perf_counter()
    if (
        isinstance(max_faults, bool)
        or not isinstance(max_faults, numbers.Integral)
        or max_faults < 1
    ):
        raise ValueError("max_faults must be a positive integer")
    qubits = tuple(output_qubits)
    selected = tuple(post_select)
    _validate_indices(qubits, circuit.num_qubits, "output_qubits", nonempty=True)
    _validate_indices(selected, circuit.num_detectors, "post_select")
    stabilizers = get_state_stabilizers(circuit, qubits)
    if len(stabilizers) != len(qubits):
        raise ValueError("Selected outputs must have a definite pure, complete stabilizer state")
    diagnostics = get_pauli_product_measurements(stabilizers, qubits=qubits)
    for jj in range(len(stabilizers)):
        diagnostics.append("DETECTOR", [stim.target_rec(jj - len(stabilizers))])
    diagnostic_circuit = circuit + diagnostics
    # Exact anti-commutation analysis, not sampled signed-flow checks, validates determinism.
    diagnostic_circuit.without_noise().detector_error_model()
    dem = diagnostic_circuit.detector_error_model(approximate_disjoint_errors=True).flattened()
    selected_map = {detector: jj for jj, detector in enumerate(selected)}
    flags: list[int] = []
    syndromes: list[int] = []
    for instruction in dem:
        if instruction.type != "error":
            continue
        flag = syndrome = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                detector = target.val
                if detector in selected_map:
                    flag ^= 1 << selected_map[detector]
                if detector >= circuit.num_detectors:
                    syndrome ^= 1 << (detector - circuit.num_detectors)
        flags.append(flag)
        syndromes.append(syndrome)
    matrix = np.array(
        [np.concatenate(stabilizer.to_numpy()) for stabilizer in stabilizers], dtype=np.uint8
    )
    oracle = PauliCosetSearch(matrix)
    statistics = StatePrepSpreadStatistics(mechanisms=len(flags))
    statistics.setup_seconds = time.perf_counter() - start
    counterexample = _certify_maps(flags, syndromes, oracle, int(max_faults), statistics)
    return StatePrepSpreadCertificate(
        counterexample is None,
        int(max_faults),
        tuple(map(int, qubits)),
        tuple(map(str, stabilizers)),
        dem,
        counterexample,
        statistics,
    )


def _validate_indices(
    indices: tuple[int, ...], size: int, name: str, *, nonempty: bool = False
) -> None:
    if (
        (nonempty and not indices)
        or any(isinstance(jj, bool) or not isinstance(jj, numbers.Integral) for jj in indices)
        or any(not 0 <= jj < size for jj in indices)
        or len(set(indices)) != len(indices)
    ):
        raise ValueError(f"{name} must contain distinct in-range integer indices")


def _certify_maps(
    flags: Sequence[int],
    syndromes: Sequence[int],
    oracle: PauliCosetSearch,
    budget: int,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    start = time.perf_counter()
    representatives = {
        (flag, syndrome): jj
        for jj, (flag, syndrome) in enumerate(zip(flags, syndromes, strict=True))
        if flag or syndrome
    }
    statistics.distinct_symptoms = len(representatives)
    statistics.flag_rank = _binary_rank([flag for flag, _ in representatives])
    statistics.effective_budget = min(budget, statistics.flag_rank + 1)
    remaining: list[tuple[int, int, int]] = []

    def check(indices: tuple[int, ...], syndrome: int) -> StatePrepSpreadCounterexample | None:
        statistics.minimal_candidates += 1
        decision = oracle.decide(syndrome, len(indices))
        if decision.within_bound:
            return None
        n = oracle.num_qubits
        pauli = stim.PauliString.from_numpy(
            xs=np.array([(decision.representative >> jj) & 1 for jj in range(n)], dtype=bool),
            zs=np.array([(decision.representative >> (jj + n)) & 1 for jj in range(n)], dtype=bool),
        )
        return StatePrepSpreadCounterexample(
            indices,
            tuple((syndrome >> jj) & 1 for jj in range(n)),
            decision.lower_bound,
            str(pauli),
        )

    def search() -> StatePrepSpreadCounterexample | None:
        for (flag, syndrome), original in representatives.items():
            if flag == 0:
                statistics.generated += 1
                statistics.bounded_candidates += 1
                failure = check((original,), syndrome)
                if failure is not None:
                    return failure
            else:
                remaining.append((flag, syndrome, original))
        limit = statistics.effective_budget
        if limit < 2 or not remaining:
            return None
        if statistics.flag_rank == 1:
            statistics.search_method = "pairs"
            for left, right in itertools.combinations(remaining, 2):
                statistics.generated += 1
                statistics.bounded_candidates += 1
                failure = check((left[2], right[2]), left[1] ^ right[1])
                if failure is not None:
                    return failure
            return None
        independent_flags = (
            len(remaining) > 2 * statistics.flag_rank
            and len({flag for flag, _, _ in remaining}) == statistics.flag_rank
        )
        if _use_prefix_search(
            len(remaining), statistics.flag_rank, limit, independent_flags=independent_flags
        ):
            statistics.search_method = "independent_prefix"
            if independent_flags:
                limit = statistics.effective_budget = 2
            for indices, syndrome in _iter_minimal_patterns(remaining, limit, statistics):
                statistics.bounded_candidates += 1
                failure = check(indices, syndrome)
                if failure is not None:
                    return failure
            return None
        statistics.search_method = "brouwer_zimmermann"
        checks = np.array(
            [
                [(flag >> jj) & 1 for flag, _, _ in remaining]
                for jj in range(max(flags).bit_length())
            ],
            dtype=np.uint8,
        )
        outer = BoundedKernelSearch(checks, limit)
        statistics.outer_ranks = outer.ranks
        try:
            for word in outer.words():
                statistics.bounded_candidates += 1
                support = tuple(jj for jj in range(len(remaining)) if (word >> jj) & 1)
                if _binary_rank([remaining[jj][0] for jj in support]) != len(support) - 1:
                    continue
                syndrome = 0
                for jj in support:
                    syndrome ^= remaining[jj][1]
                failure = check(tuple(remaining[jj][2] for jj in support), syndrome)
                if failure is not None:
                    return failure
        finally:
            statistics.generated += outer.generated
            statistics.coefficient_depth = outer.coefficient_depth
        return None

    failure = search()
    statistics.syndrome_queries = oracle.queries
    statistics.cache_hits = oracle.cache_hits
    statistics.inner_generated = oracle.generated
    statistics.inner_preparations = oracle.preparations
    statistics.inner_ranks = oracle.ranks
    statistics.search_seconds = time.perf_counter() - start
    return failure


def _use_prefix_search(
    num_columns: int, rank: int, budget: int, *, independent_flags: bool = False
) -> bool:
    """Choose high-rate prefix closure by pair-only structure or measured support counts."""
    dimension = num_columns - rank
    if budget < 2 or dimension <= rank:
        return False
    if independent_flags:
        return True
    coefficient_count = sum(
        math.comb(dimension, depth) for depth in range(1, min(dimension, budget) + 1)
    )
    prefix_count = sum(
        math.comb(num_columns, depth) for depth in range(1, min(num_columns, budget - 1) + 1)
    )
    return coefficient_count >= 1024 and coefficient_count >= 2 * prefix_count


def _iter_minimal_patterns(
    mechanisms: Sequence[tuple[int, int, int]],
    budget: int,
    statistics: StatePrepSpreadStatistics,
) -> Iterator[tuple[tuple[int, ...], int]]:
    """Yield minimal patterns of nonzero-flag (flag, syndrome, original-index) columns.

    Prefixes are independent and increasing in *internal* column order, not necessarily original
    DEM order.  The largest internal index is the unique closing column.  Carrying its prefix XOR
    and state syndrome avoids a full word reconstruction or repeated output-map multiplication.
    """
    groups: dict[int, list[int]] = {}
    for index, (flag, _, _) in enumerate(mechanisms):
        groups.setdefault(flag, []).append(index)

    def visit(
        support: tuple[int, ...],
        start: int,
        flag_sum: int,
        syndrome: int,
        basis: tuple[int, ...],
    ) -> Iterator[tuple[tuple[int, ...], int]]:
        if support:
            statistics.independent_prefixes += 1
            matches = groups.get(flag_sum, [])
            for position in range(bisect_left(matches, start), len(matches)):
                _, closing_syndrome, original = mechanisms[matches[position]]
                statistics.generated += 1
                yield support + (original,), syndrome ^ closing_syndrome
        if len(support) >= budget - 1:
            return
        for index in range(start, len(mechanisms)):
            statistics.generated += 1
            statistics.prefix_extensions += 1
            flag, column_syndrome, original = mechanisms[index]
            reduced = flag
            for row in basis:
                reduced = min(reduced, reduced ^ row)
            if reduced:
                yield from visit(
                    support + (original,),
                    index + 1,
                    flag_sum ^ flag,
                    syndrome ^ column_syndrome,
                    tuple(sorted((*basis, reduced), reverse=True)),
                )

    yield from visit((), 0, 0, 0, ())


def _binary_rank(columns: Sequence[int]) -> int:
    pivots: dict[int, int] = {}
    for column in columns:
        while column:
            pivot = column.bit_length()
            if pivot in pivots:
                column ^= pivots[pivot]
            else:
                pivots[pivot] = column
                break
    return len(pivots)
