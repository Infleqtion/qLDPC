# SPDX-License-Identifier: Apache-2.0

"""Exact bounded-spread certificates for pure Clifford state preparation."""

from __future__ import annotations

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

from ..common import get_pauli_product_measurements
from ..encoding import get_state_stabilizers

####################################################################################################
# state-preparation certification


def certify_state_prep_spread(
    circuit: stim.Circuit,
    *,
    output_qubits: Sequence[int],
    max_faults: int,
    post_select: Collection[int] = (),
    css_weight: bool | None = None,
) -> StatePrepSpreadCertificate:
    r"""Exactly certify bounded error spread of an already noisy Clifford preparation.

    For the complete pure output-state stabilizers, let ``S`` map DEM faults to their state
    syndrome, ``P`` map them to the selected actual rejection detectors, and ``mu(s)`` be the
    minimum chosen weight with syndrome ``s``, jointly over a single representative.
    The certificate proves
    ``mu(S f) <= |f|`` whenever ``P f = 0`` and ``1 <= |f| <= max_faults``.
    Physical weight counts ``|support(x) union support(z)|``, with Y affecting one qubit.
    CSS weight counts ``max(|x|, |z|)``, in the supplied physical frame.
    Logical stabilizers fixing the prepared state are included.

    Args:
        circuit: An already noisy Stim Clifford circuit, starting in Stim's all-zero state.
        output_qubits: Nonempty distinct physical output identities, in the desired Pauli frame.
        max_faults: Required positive integer unit-cost DEM fault budget.
        post_select: Indices of actual circuit detectors whose *events* must vanish relative to
            their deterministic ideal reference.  Appended output diagnostics are not rejected.
        css_weight: True selects CSS-sector weight; False selects physical Pauli support.
            None selects CSS weight exactly when the complete prepared-state stabilizer row
            space has a pure-X/pure-Z decomposition in the physical output basis.
            The resolved boolean is returned in the certificate.

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
    Automatic selection is independent of the chosen stabilizer generators, but not of the
    physical Pauli basis: it does not use local-Clifford equivalence.
    For the complete rank-n row space W, the pure-X and pure-Z intersections have dimensions
    ``n - rank(W_Z)`` and ``n - rank(W_X)``.
    They span W exactly when their dimension sum is n.
    This detects state equivalences, not compatibility with an unspecified downstream protocol.
    A CSS code may prepare a non-CSS logical stabilizer state, requiring an explicit True
    override for its sector-wise application.
    Both metrics always minimize jointly, never by separately minimizing the X and Z projections.

    The model is independent binary DEM mechanisms, not mutually exclusive alternatives at
    physical gate locations.  Stim's disjoint-channel probability approximation is enabled only
    to obtain those mechanisms; probabilities play no role in the combinatorial proof.
    Correlated mechanisms are never split.  This is not an exact sampling-distribution claim.

    Proof of the outer reduction: equal complete syndromes differ by a state stabilizer because
    the stabilizer space is maximal isotropic.  Minimum representative weight is subadditive,
    since each sector's product support is contained in its union, and
    ``max(a+c,b+d) <= max(a,b)+max(c,d)``.  Every accepted nonzero pattern partitions
    into support-disjoint minimal accepted patterns: remove a proper accepted subset and repeat.
    Thus it suffices to check minimal patterns.  For accepted support J, minimality is equivalent
    to ``nullity(P[:, J]) = 1``, so ``|J| <= rank(P) + 1``.  A flag-free column occurs only as a
    singleton.  General dependencies use packed BZ
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
    each input error of the selected weight r and additional faults within the corresponding
    r-plus-fault budget (or correct them).  CSS-sector use requires suitable sector-preserving
    downstream operations; a general Clifford can mix sectors.  An explicit CSS override on a
    non-CSS application is the caller's responsibility, not an automatic correction theorem.
    No arbitrary downstream algorithm, acceptance
    probability, or necessary-and-sufficient fault-tolerance criterion is certified here.
    """
    start = time.perf_counter()
    if css_weight is not None and not isinstance(css_weight, bool):
        raise ValueError("css_weight must be a bool or None")
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

    # Complete state stabilizers, including logical stabilizers, identify output-error classes.
    stabilizers = get_state_stabilizers(circuit, qubits)
    if len(stabilizers) != len(qubits):
        raise ValueError("Selected outputs must have a definite pure, complete stabilizer state")
    matrix = np.array(
        [np.concatenate(stabilizer.to_numpy()) for stabilizer in stabilizers], dtype=np.uint8
    )
    resolved_css_weight = _is_css_state(matrix) if css_weight is None else css_weight
    dem, flags, syndromes = _get_fault_maps(circuit, stabilizers, qubits, selected)

    # Reuse one prepared coset oracle for every fault pattern and syndrome encountered below.
    oracle = PauliCosetSearch(matrix, weight="css" if resolved_css_weight else "pauli")
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
        resolved_css_weight,
    )


####################################################################################################
# certificate results and search statistics


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
    search_method: Literal["singletons", "brouwer_zimmermann", "independent_prefix"] = "singletons"
    prefix_extensions: int = 0
    independent_prefixes: int = 0


@dataclass(frozen=True)
class StatePrepSpreadCertificate:
    """Completed bounded-spread decision, including the exact model and witness frame.

    ``certified`` means every accepted pattern of one through ``max_faults`` unit-cost binary
    DEM mechanisms has effective output error weight no greater than its own fault count.
    ``css_weight`` records the resolved metric: True for CSS-sector weight, False for physical
    Pauli support.
    A negative decision always includes a counterexample.  No incomplete search returns a result.
    """

    certified: bool
    max_faults: int
    output_qubits: tuple[int, ...]
    target_stabilizers: tuple[str, ...]
    detector_error_model: stim.DetectorErrorModel
    counterexample: StatePrepSpreadCounterexample | None
    statistics: StatePrepSpreadStatistics
    css_weight: bool = False


####################################################################################################
# circuit diagnostics and input validation


def _get_fault_maps(
    circuit: stim.Circuit,
    stabilizers: Sequence[stim.PauliString],
    qubits: Sequence[int],
    selected: Sequence[int],
) -> tuple[stim.DetectorErrorModel, list[int], list[int]]:
    """Compile the undecomposed DEM and its packed rejection/output-syndrome columns.

    Each returned list has one integer per DEM error instruction.  Bit positions follow the
    selected-detector order and stabilizer order, respectively; the error-instruction order is
    retained so search witnesses can name mechanisms in the returned model.
    """
    # Appended detectors diagnose the output state; only selected original detectors reject.
    diagnostics = get_pauli_product_measurements(stabilizers, qubits=qubits)
    for jj in range(len(stabilizers)):
        diagnostics.append("DETECTOR", [stim.target_rec(jj - len(stabilizers))])
    diagnostic_circuit = circuit + diagnostics

    # Exact anti-commutation analysis, not sampled signed-flow checks, validates determinism.
    diagnostic_circuit.without_noise().detector_error_model()
    dem = diagnostic_circuit.detector_error_model(approximate_disjoint_errors=True).flattened()

    # Keep each correlated mechanism intact and pack its two distinct kinds of detector effects.
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
    return dem, flags, syndromes


def _is_css_state(stabilizers: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]) -> bool:
    """Test pure-sector intersection dimensions of an already complete rank-n row space."""
    n = len(stabilizers)
    # The pure-X/Z intersections have dimensions n-rank(W_Z) and n-rank(W_X).
    # They span W exactly when rank(W_X)+rank(W_Z)=n, regardless of the generator basis.
    ranks = [
        _binary_rank([int.from_bytes(np.packbits(row).tobytes(), "little") for row in sector])
        for sector in (stabilizers[:, :n], stabilizers[:, n:])
    ]
    return sum(ranks) == n


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


####################################################################################################
# bounded search over accepted fault patterns


def _certify_maps(
    flags: Sequence[int],
    syndromes: Sequence[int],
    oracle: PauliCosetSearch,
    budget: int,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Compress joint symptoms, search minimal accepted supports, and collect oracle statistics."""
    start = time.perf_counter()

    # Equal joint symptoms cancel in pairs without changing acceptance or the output syndrome.
    # Removing those pairs cannot hide a violation; retain original DEM indices for witnesses.
    representatives = {
        (flag, syndrome): jj
        for jj, (flag, syndrome) in enumerate(zip(flags, syndromes, strict=True))
        if flag or syndrome
    }
    statistics.distinct_symptoms = len(representatives)
    statistics.flag_rank = _binary_rank([flag for flag, _ in representatives])

    # A minimal accepted support has nullity one, hence at most rank(P)+1 mechanisms.
    statistics.effective_budget = min(budget, statistics.flag_rank + 1)
    try:
        return _search_minimal_patterns(representatives, oracle, statistics)
    finally:
        statistics.syndrome_queries = oracle.queries
        statistics.cache_hits = oracle.cache_hits
        statistics.inner_generated = oracle.generated
        statistics.inner_preparations = oracle.preparations
        statistics.inner_ranks = oracle.ranks
        statistics.search_seconds = time.perf_counter() - start


def _search_minimal_patterns(
    representatives: dict[tuple[int, int], int],
    oracle: PauliCosetSearch,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Check flag-free singletons, then dispatch the remaining minimal dependencies."""
    remaining: list[tuple[int, int, int]] = []

    # A flag-free mechanism is minimal by itself and cannot belong to a larger minimal support.
    for (flag, syndrome), original in representatives.items():
        if flag == 0:
            statistics.generated += 1
            statistics.bounded_candidates += 1
            failure = _check_fault_pattern((original,), syndrome, oracle, statistics)
            if failure is not None:
                return failure
        else:
            remaining.append((flag, syndrome, original))
    limit = statistics.effective_budget
    if limit < 2 or not remaining:
        return None

    # Independent distinct flag types require even multiplicities, so their minimal supports
    # are equal-flag pairs.  This reduction applies at any flag rank, not just one flag.
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
            failure = _check_fault_pattern(indices, syndrome, oracle, statistics)
            if failure is not None:
                return failure
        return None

    return _search_bz_patterns(remaining, limit, oracle, statistics)


def _check_fault_pattern(
    indices: tuple[int, ...],
    syndrome: int,
    oracle: PauliCosetSearch,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Prove this pattern safe or package a violation in the original DEM/output frames."""
    statistics.minimal_candidates += 1
    # Compare with this pattern's own fault count, not the maximum allowed search budget.
    decision = oracle.decide(syndrome, len(indices))
    if decision.within_bound:
        return None

    # The oracle supplies a representative and a proved lower bound, not necessarily a minimizer.
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


####################################################################################################
# exact minimal-dependency search strategies


def _use_prefix_search(
    num_columns: int, rank: int, budget: int, *, independent_flags: bool = False
) -> bool:
    """Choose high-rate prefix closure by pair-only structure or measured support counts."""
    dimension = num_columns - rank
    if budget < 2 or dimension <= rank:
        return False
    if independent_flags:
        return True
    # This cost heuristic chooses between complete searches; it is not a correctness bound.
    coefficient_count = sum(
        math.comb(dimension, depth) for depth in range(1, min(dimension, budget) + 1)
    )
    prefix_count = sum(
        math.comb(num_columns, depth) for depth in range(1, min(num_columns, budget - 1) + 1)
    )
    return coefficient_count >= 1024 and coefficient_count >= 2 * prefix_count


def _search_bz_patterns(
    mechanisms: Sequence[tuple[int, int, int]],
    budget: int,
    oracle: PauliCosetSearch,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Enumerate the bounded rejection-map kernel with BZ, testing only minimal supports."""
    statistics.search_method = "brouwer_zimmermann"
    checks = np.array(
        [
            [(flag >> jj) & 1 for flag, _, _ in mechanisms]
            for jj in range(max(flag for flag, _, _ in mechanisms).bit_length())
        ],
        dtype=np.uint8,
    )
    outer = BoundedKernelSearch(checks, budget)
    statistics.outer_ranks = outer.ranks
    try:
        for word in outer.words():
            statistics.bounded_candidates += 1
            support = tuple(jj for jj in range(len(mechanisms)) if (word >> jj) & 1)

            # BZ emits nonminimal kernel words too.  Nullity one excludes accepted supports
            # that can be decomposed into smaller, disjoint accepted patterns.
            if _binary_rank([mechanisms[jj][0] for jj in support]) != len(support) - 1:
                continue
            syndrome = 0
            for jj in support:
                syndrome ^= mechanisms[jj][1]
            indices = tuple(mechanisms[jj][2] for jj in support)
            failure = _check_fault_pattern(indices, syndrome, oracle, statistics)
            if failure is not None:
                return failure
    finally:
        # Preserve work counts even if a witness ends the search before BZ exhausts its iterator.
        statistics.generated += outer.generated
        statistics.coefficient_depth = outer.coefficient_depth
    return None


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

    # Each recursive prefix is independent.  Its full flag XOR is the only possible closing
    # column; requiring a later closing index emits each minimal dependence exactly once.
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
            # Descending leading bits give an echelon basis.  A zero remainder means the prefix
            # already contains a dependence and cannot extend to a minimal accepted support.
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


####################################################################################################
# binary linear algebra


def _binary_rank(columns: Sequence[int]) -> int:
    """Rank of binary columns packed into Python integers, using leading-bit elimination."""
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
