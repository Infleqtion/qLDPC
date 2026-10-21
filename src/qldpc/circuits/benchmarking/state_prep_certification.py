# SPDX-License-Identifier: Apache-2.0

"""Exact bounded-spread certification for pure Clifford states.

Let ``P`` map faults to rejection flags, ``S`` map them to complete state syndromes, and ``mu(s)``
be the minimum chosen Pauli weight with syndrome ``s``.  The test proves ``mu(S f) <= |f|`` for
``P f = 0`` and ``1 <= |f| <= max_faults``:

- Equal complete syndromes differ by a prepared-state stabilizer, including logical stabilizers,
  since that space is maximal isotropic.  Thus ``mu`` depends only on the syndrome.
- Both weights are subadditive: product support lies in the union of supports, and CSS weight
  satisfies ``max(a+c, b+d) <= max(a, b) + max(c, d)``.
- Removing proper accepted subsets partitions any accepted pattern into disjoint minimal ones.
  Subadditivity then extends their bounds to the whole pattern.  An accepted support ``J`` is
  minimal exactly when ``nullity(P[:, J]) = 1``, so ``|J| <= rank(P) + 1``.  Flag-free faults are
  minimal only as singletons.
- Identical joint (flag, syndrome) columns cancel in pairs; zero columns have no effect.  Merging
  them cannot hide violations, and merged patterns still name original DEM mechanisms.
- If distinct nonzero flag types are independent, acceptance requires even multiplicities.
  The flagged minimal patterns are therefore equal-flag pairs, at any rank and budget.

Two complete searches cover the remaining patterns.  Brouwer--Zimmermann (BZ) visits all words of
``ker(P)`` through the weight bound, regardless of the nonlinear spread predicate; a rank test
removes nonminimal words.  Independent-prefix closure uses the fact that a sorted minimal dependence
is exactly an independent prefix plus its XOR as the last column.  Requiring a later closing index
emits each minimal support once; dependent prefixes cannot extend to minimal supports.  A cost
heuristic chooses between these complete searches without a runtime guarantee.  One shared exact
``_PauliCosetSearch`` oracle decides all patterns.
"""

from __future__ import annotations

import math
import numbers
from bisect import bisect_left
from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import stim

from qldpc.codes.distance._bounded_weight_search import _BoundedKernelSearch, _PauliCosetSearch

from ..common import get_pauli_product_measurements
from ..encoding import get_state_stabilizers

# Avoid switching searches for small BZ jobs or marginal estimated savings.
_MIN_COEFFICIENT_SUPPORTS = 1024
_PREFIX_ADVANTAGE = 2

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
    r"""Certify bounded error spread of a noisy Clifford state preparation.

    Check that every accepted pattern of one through ``max_faults`` faults leaves an output error
    equivalent, modulo the complete prepared-state stabilizers (including logical stabilizers), to
    an error whose weight is at most the number of faults.  Physical weight counts qubits with X, Y,
    or Z; CSS-sector weight is ``max(|x|, |z|)``.  Both minimize jointly over one representative,
    not separately over X and Z.

    Faults are independent unit-cost mechanisms of the undecomposed binary detector error model
    (DEM), not mutually exclusive physical alternatives.  Correlated mechanisms stay intact.  Stim's
    disjoint-channel approximation supplies the mechanisms; probabilities play no role, and the
    certificate makes no sampling-distribution claim.

    The selected output must be a definite pure stabilizer state, independent of random measurement
    branches.  All detector and observable annotations, even unselected ones, must be deterministic.
    Random post-selected branches are unsupported.  Stabilizer signs and physical frames are
    preserved; appended diagnostics are noiseless.

    Automatic CSS detection uses the complete stabilizer row space in the physical Pauli basis, not
    local-Clifford equivalence.  A CSS code can prepare a non-CSS logical state, so sector-wise use
    may require explicit ``css_weight=True``.

    This is a sufficient resource-state property, not a downstream-composition certificate.
    Downstream gadgets must propagate weight-r inputs and added faults within an r-plus-fault
    budget, or correct them; CSS use also requires suitable sector-preserving operations.  The
    caller must justify a CSS override.  Acceptance probability and necessary-and-sufficient fault
    tolerance are not certified.  The exact, nonsampling search can take exponential time.  See the
    module docstring for the proof.

    Args:
        circuit: Noisy Stim Clifford circuit starting in Stim's all-zero state.
        output_qubits: Nonempty, distinct physical output indices in the desired Pauli frame.
        max_faults: Positive integer unit-cost DEM fault budget.
        post_select: Circuit detector indices whose events must vanish relative to their ideal
            reference; appended output diagnostics do not reject.
        css_weight: True for CSS-sector weight, False for physical weight, or None to select CSS
            weight exactly when the complete state has a pure-X/pure-Z stabilizer decomposition.

    Returns:
        The decision, model, and resolved metric.  Failure includes an accepted counterexample
        with a proved weight lower bound exceeding its fault count; its indices refer to error
        instructions in the returned DEM, before compression.

    Raises:
        ValueError: Invalid inputs, mixed or incomplete outputs, nondeterministic annotations or
            diagnostics, or unsupported circuit-to-DEM conversion.
    """
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

    oracle = _PauliCosetSearch(matrix, metric="css" if resolved_css_weight else "pauli")
    statistics = StatePrepSpreadStatistics(mechanisms=len(flags))
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
    """An accepted DEM pattern violating the spread bound.

    ``fault_indices`` index error instructions, not all instructions, in the returned flattened,
    undecomposed DEM.  ``output_syndrome`` follows ``target_stabilizers`` order.
    ``effective_weight_lower_bound`` is proved.  ``output_pauli`` is a Stim string in output-qubit
    order and the caller's frame, not necessarily a minimizer or the propagated error.  The witness
    need not have the smallest violating fault count.
    """

    fault_indices: tuple[int, ...]
    output_syndrome: tuple[int, ...]
    effective_weight_lower_bound: int
    output_pauli: str

    @property
    def fault_count(self) -> int:
        """Number of original DEM mechanisms in the witness."""
        return len(self.fault_indices)


@dataclass
class StatePrepSpreadStatistics:
    """Deterministic search-effort counters for one certification."""

    mechanisms: int = 0
    """Unit-cost error mechanisms in the diagnostic DEM."""

    flag_rank: int = 0
    """GF(2) rank of the selected rejection flags."""

    effective_budget: int = 0
    """Fault budget capped at ``flag_rank + 1``, or 2 when searching independent flag types."""

    search_method: Literal["singletons", "brouwer_zimmermann", "independent_prefix"] = "singletons"
    """Flagged-pattern search, or ``singletons`` if no flagged patterns are searched."""

    generated: int = 0
    """Flag-free singletons plus search-dependent work, not all fault combinations.

    BZ counts coefficient supports; prefix search counts attempted extensions (including dependent
    ones) and emitted closing supports.
    """

    minimal_candidates: int = 0
    """Minimal accepted patterns checked for output weight."""


@dataclass(frozen=True)
class StatePrepSpreadCertificate:
    """Completed spread decision with its model and witness frame.

    Failure always includes a counterexample; incomplete searches return no result.
    """

    certified: bool
    """Whether accepted patterns of one through ``max_faults`` faults meet the spread bound."""

    max_faults: int
    """Requested unit-cost fault budget."""

    output_qubits: tuple[int, ...]
    """Physical output indices in target-stabilizer and witness-Pauli order."""

    target_stabilizers: tuple[str, ...]
    """Signed Stim strings for the complete state stabilizers, one per output qubit."""

    detector_error_model: stim.DetectorErrorModel
    """Flattened, undecomposed DEM including noiseless output diagnostics."""

    counterexample: StatePrepSpreadCounterexample | None
    """Accepted violating pattern on failure, or None on success."""

    statistics: StatePrepSpreadStatistics
    """Deterministic search-effort counters."""

    css_weight: bool
    """Resolved metric: True for CSS-sector weight, False for physical Pauli support."""


####################################################################################################
# helper methods


def _get_fault_maps(
    circuit: stim.Circuit,
    stabilizers: Sequence[stim.PauliString],
    qubits: Sequence[int],
    selected: Sequence[int],
) -> tuple[stim.DetectorErrorModel, list[int], list[int]]:
    """Compile the undecomposed DEM with packed flag and syndrome columns.

    Lists follow DEM error-instruction order; bits follow selected-detector and stabilizer order.
    """
    # Appended detectors diagnose the output state; only selected original detectors reject.
    diagnostics = get_pauli_product_measurements(stabilizers, qubits=qubits)
    for jj in range(len(stabilizers)):
        diagnostics.append("DETECTOR", [stim.target_rec(jj - len(stabilizers))])
    diagnostic_circuit = circuit + diagnostics

    # Exact anticommutation analysis validates determinism without sampling.
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
    return dem, flags, syndromes


def _is_css_state(stabilizers: np.ndarray[tuple[int, ...], np.dtype[np.uint8]]) -> bool:
    """Test whether a complete rank-n stabilizer space is CSS in the physical basis."""
    n = len(stabilizers)
    # The pure-X/Z intersections have dimensions n-rank(W_Z) and n-rank(W_X).  They span W exactly
    # when rank(W_X)+rank(W_Z)=n, regardless of the generator basis.
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


def _certify_maps(
    flags: Sequence[int],
    syndromes: Sequence[int],
    oracle: _PauliCosetSearch,
    budget: int,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Merge joint columns and search minimal accepted supports."""
    # Equal joint columns cancel without changing acceptance or syndrome; removing pairs cannot hide
    # violations.  Keep original DEM indices for witnesses.
    representatives = {
        (flag, syndrome): jj
        for jj, (flag, syndrome) in enumerate(zip(flags, syndromes, strict=True))
        if flag or syndrome
    }
    statistics.flag_rank = _binary_rank([flag for flag, _ in representatives])

    # A minimal accepted support has nullity one, hence at most rank(P)+1 mechanisms.
    statistics.effective_budget = min(budget, statistics.flag_rank + 1)
    return _search_minimal_patterns(representatives, oracle, statistics)


def _search_minimal_patterns(
    representatives: dict[tuple[int, int], int],
    oracle: _PauliCosetSearch,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Check flag-free singletons, then the remaining minimal dependencies."""
    remaining: list[tuple[int, int, int]] = []

    # A flag-free mechanism is minimal by itself and cannot belong to a larger minimal support.
    for (flag, syndrome), original in representatives.items():
        if flag == 0:
            statistics.generated += 1
            failure = _check_fault_pattern((original,), syndrome, oracle, statistics)
            if failure is not None:
                return failure
        else:
            remaining.append((flag, syndrome, original))
    limit = statistics.effective_budget
    if limit < 2 or not remaining:
        return None

    # Independent flag types require even multiplicities: only equal-flag pairs are minimal.
    if len({flag for flag, _, _ in remaining}) == statistics.flag_rank:
        limit = statistics.effective_budget = 2
    elif not _use_prefix_search(len(remaining), statistics.flag_rank, limit):
        return _search_bz_patterns(remaining, limit, oracle, statistics)

    statistics.search_method = "independent_prefix"
    for indices, syndrome in _iter_minimal_patterns(remaining, limit, statistics):
        failure = _check_fault_pattern(indices, syndrome, oracle, statistics)
        if failure is not None:
            return failure
    return None


def _check_fault_pattern(
    indices: tuple[int, ...],
    syndrome: int,
    oracle: _PauliCosetSearch,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Check a pattern or return its violation in the original DEM and output frames."""
    statistics.minimal_candidates += 1
    # Compare with this pattern's own fault count, not the maximum allowed search budget.
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


def _use_prefix_search(num_columns: int, rank: int, budget: int) -> bool:
    """Choose between complete searches by estimated work, not correctness."""
    dimension = num_columns - rank
    if dimension <= rank:
        return False

    # Prefix XOR determines the closing column, saving one support layer.
    coefficient_count = sum(
        math.comb(dimension, depth) for depth in range(1, min(dimension, budget) + 1)
    )
    prefix_count = sum(
        math.comb(num_columns, depth) for depth in range(1, min(num_columns, budget - 1) + 1)
    )
    return (
        coefficient_count >= _MIN_COEFFICIENT_SUPPORTS
        and coefficient_count >= _PREFIX_ADVANTAGE * prefix_count
    )


def _search_bz_patterns(
    mechanisms: Sequence[tuple[int, int, int]],
    budget: int,
    oracle: _PauliCosetSearch,
    statistics: StatePrepSpreadStatistics,
) -> StatePrepSpreadCounterexample | None:
    """Enumerate the bounded rejection kernel with BZ and check minimal supports."""
    statistics.search_method = "brouwer_zimmermann"
    checks = np.array(
        [
            [(flag >> jj) & 1 for flag, _, _ in mechanisms]
            for jj in range(max(flag for flag, _, _ in mechanisms).bit_length())
        ],
        dtype=np.uint8,
    )
    outer = _BoundedKernelSearch(checks, budget)
    try:
        for word in outer.words():
            support = tuple(jj for jj in range(len(mechanisms)) if (word >> jj) & 1)

            # Nullity one excludes BZ words that split into smaller accepted patterns.
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
        # Count work even when a witness ends iteration early.
        statistics.generated += outer.generated
    return None


def _iter_minimal_patterns(
    mechanisms: Sequence[tuple[int, int, int]],
    budget: int,
    statistics: StatePrepSpreadStatistics,
) -> Iterator[tuple[tuple[int, ...], int]]:
    """Yield minimal patterns of nonzero-flag (flag, syndrome, original-index) columns.

    Prefixes are independent and increase in internal, not original DEM, order.  The largest
    internal index uniquely closes the pattern; carrying XORs avoids reconstructing full words.
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
            matches = groups.get(flag_sum, [])
            for position in range(bisect_left(matches, start), len(matches)):
                _, closing_syndrome, original = mechanisms[matches[position]]
                statistics.generated += 1
                yield support + (original,), syndrome ^ closing_syndrome
        if len(support) >= budget - 1:
            return
        for index in range(start, len(mechanisms)):
            statistics.generated += 1
            flag, column_syndrome, original = mechanisms[index]
            # Descending leading bits give an echelon basis; a zero remainder means dependence,
            # which cannot extend to a minimal accepted support.
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
    """Find the rank of packed binary columns by leading-bit elimination."""
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
