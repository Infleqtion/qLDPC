# SPDX-License-Identifier: Apache-2.0

"""Internal exact bounded-kernel enumeration and affine Pauli-coset decisions.

These searches serve ``circuits.certify_state_prep_spread`` and are not part of the public API.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ._binary import (
    _as_binary_matrix,
    _assert_binary,
    _pack_bits,
    _row_reduce_binary,
    _symplectic_to_hamming,
)
from .brouwer_zimmermann import _get_brouwer_zimmermann_lower_bound, _get_information_set_generators

####################################################################################################
# bounded kernel and affine Pauli searches


class _BoundedKernelSearch:
    """Enumerate every binary kernel word through a weight bound using packed BZ coverage.

    Args:
        checks: Binary parity-check matrix, including redundant or zero rows.
        max_weight: Only yield nonzero words of at most this Hamming weight.
        batch_size: Maximum number of coefficient supports processed together.

    Words are Python integers, with column zero in the least significant bit.  A word can be
    yielded repeatedly in different information sets; consumers may apply arbitrary predicates.
    No eligible-word upper bound or ordinary distance-search fallback is used.

    After coefficient weights through ``a`` in dimension ``k``, an unseen word has weight at least
    ``sum(max(0, a + 1 - (k - r)) for r in ranks)``: its coefficient vector in each basis has more
    than ``a`` bits, of which at most ``k-r`` lie outside that set's pivots.  The pivot sets are
    disjoint, so their bounds add.  Enumeration stops only when this bound exceeds ``max_weight``
    or all coefficients have been exhausted.  Filtering yielded words cannot weaken this proof.
    The ``lower_bound`` attribute bounds unseen words only, not the minimum over all kernel words.

    When the kernel dimension exceeds the check rank, store only the pivot-coordinate parts of
    the systematic generators, not a quadratic-size dense kernel matrix.  Coefficient support
    supplies the free coordinates and their exact weight.  In the lower-rate case, reuse the
    distance implementation's disjoint information-set construction.
    """

    def __init__(self, checks: npt.ArrayLike, max_weight: int, *, batch_size: int = 4096) -> None:
        _assert_binary(checks, "checks")
        matrix = _as_binary_matrix(checks)
        if max_weight < 0 or batch_size < 1:
            raise ValueError("max_weight must be nonnegative and batch_size must be positive")
        reduced, _, pivots = _row_reduce_binary(matrix)
        self.rank = len(pivots)
        self.length = matrix.shape[1]
        self.dimension = self.length - self.rank
        self.max_weight = min(max_weight, self.length)
        self.batch_size = batch_size
        self.generated = 0
        self.coefficient_depth = 0
        self.lower_bound = 1
        self._pivots = tuple(map(int, pivots))
        self._free = tuple(col for col in range(self.length) if col not in self._pivots)
        pivot_parts = reduced[: self.rank, self._free].T
        self._systematic = self.dimension > self.rank
        self.ranks: tuple[int, ...]
        if self._systematic:
            self.ranks = (self.dimension,)
            self._sets = [_pack_bits(pivot_parts)]
        elif self.dimension:
            basis = np.zeros((self.dimension, self.length), dtype=np.uint8)
            basis[:, self._pivots] = pivot_parts
            basis[:, self._free] = np.eye(self.dimension, dtype=np.uint8)
            information_sets = _get_information_set_generators(basis, None)
            self.ranks = tuple(rank for _, _, rank, _ in information_sets)
            self._sets = [_pack_bits(rows) for rows, _, _, _ in information_sets]
        else:
            self.ranks = ()
            self._sets = []

    def words(self) -> Iterator[int]:
        """Yield bounded-weight words; exhaustion is the coverage certificate."""
        for depth in range(1, min(self.dimension, self.max_weight) + 1):
            for rows in self._sets:
                for supports in _iter_fixed_weight_supports(self.dimension, depth, self.batch_size):
                    self.generated += len(supports)
                    candidates = np.bitwise_xor.reduce(rows[supports], axis=1)
                    weights = np.bitwise_count(candidates).sum(axis=-1, dtype=np.uint64)
                    if self._systematic:
                        weights += depth
                    for index in np.flatnonzero(weights <= self.max_weight):
                        word = _packed_row_int(candidates[index])
                        if self._systematic:
                            word = sum(
                                ((word >> jj) & 1) << col for jj, col in enumerate(self._pivots)
                            ) | sum(1 << self._free[int(jj)] for jj in supports[index])
                        yield word
            self.coefficient_depth = depth
            self.lower_bound = _get_brouwer_zimmermann_lower_bound(
                self.dimension, list(self.ranks), completed_weight=depth, weight_divisor=1
            )
            if self.lower_bound > self.max_weight:
                return


@dataclass(frozen=True)
class _PauliWeightDecision:
    """A checked coset representative or a global lower bound in the oracle's fixed metric.

    ``representative`` packs X in the low ``n`` bits and Z in the next ``n`` bits.
    It is not asserted to be a minimum representative.  If ``within_bound`` is False,
    ``lower_bound`` strictly exceeds the queried threshold.  In every decision it bounds the
    minimum of the entire coset, so it remains valid when reused at a different threshold.
    """

    within_bound: bool
    representative: int
    lower_bound: int


class _PauliCosetSearch:
    """Amortized exact joint error-weight oracle for a pure stabilizer state.

    Args:
        stabilizers: An independent commuting binary ``n x 2n`` matrix in ``[X|Z]`` order.
        weight: Fixed metric: ``"pauli"`` counts physical support, while ``"css"`` counts
            ``max(|x|, |z|)`` in the supplied physical frame.  Minimize jointly over one coset
            representative, including when the complete state stabilizers are non-CSS.
        cache_size: Maximum number of cached syndrome decisions (at least one).
        batch_size: Maximum coefficient supports in a packed affine BZ batch.

    The full syndrome identifies a Pauli coset because a maximal isotropic space equals its
    symplectic orthogonal.  Physical weight uses the embedding ``(x|z) -> (x|z|x+z)``, whose
    Hamming weight is twice physical weight, including Y.  CSS weight uses ``[X|Z]`` directly:
    its Hamming weight is at most twice ``max(|x|, |z|)``.
    For each information-set basis, zero the affine offset on its pivots by adding rows.
    Pivot coordinates then equal coefficient bits.  The same disjoint-set coverage bound as
    :class:`_BoundedKernelSearch` applies, starting at coefficient weight zero.  Absence of a
    threshold witness is reported only after all visited metric weights and the unseen metric
    bound exceed the threshold.  If ``v`` is the minimum visited metric weight and ``L`` bounds
    unseen binary Hamming weights, the global metric bound is ``min(v, ceil(L / 2))``.
    Exhausting all coefficients gives the exact visited minimum.  An unseen-word bound alone
    is never published as a global coset bound.

    Elimination is performed once.  Information sets are prepared lazily, at most once.
    The unconditional syndrome tables have linear size: identity and single-qubit Paulis.
    A CSS-weight-one representative may be ``X_i Z_j`` on distinct qubits.  Per-query matching
    of X-single and Z-single syndromes checks all such representatives in linear work without
    a quadratic table.  Absence from that complete search, not just physical singles, proves
    a CSS lower bound two.
    The bounded cache stores checked upper bounds and proved lower bounds.  A recomputed decision
    is merged into its entry, so a proved bound is never lost.  Eviction changes only runtime,
    never correctness.  No heuristic or probabilistic negative decision is used.
    """

    def __init__(
        self,
        stabilizers: npt.ArrayLike,
        *,
        weight: Literal["pauli", "css"] = "pauli",
        cache_size: int = 4096,
        batch_size: int = 4096,
    ) -> None:
        if weight not in ("pauli", "css"):
            raise ValueError("weight must be 'pauli' or 'css'")
        self._weight_metric = weight
        _assert_binary(stabilizers, "stabilizers")
        matrix = _as_binary_matrix(stabilizers)
        self.num_qubits = len(matrix)
        n = self.num_qubits
        if not n or matrix.shape[1] != 2 * n:
            raise ValueError("Require a complete nonempty n by 2n stabilizer matrix")
        checks = np.roll(matrix, n, axis=1)
        _, labels, pivots = _row_reduce_binary(checks, np.eye(n, dtype=np.uint8))
        if len(pivots) != n or np.any((checks @ matrix.T) % 2):
            raise ValueError("Stabilizers must be independent and commute")
        if cache_size < 1 or batch_size < 1:
            raise ValueError("cache_size and batch_size must be positive")
        assert labels is not None
        inverse = np.zeros((n, 2 * n), dtype=np.uint8)
        inverse[:, pivots] = labels.T
        self._inverse = tuple(_binary_row_int(row) for row in inverse)
        self._checks = tuple(_binary_row_int(row) for row in checks)
        self._embedded = _symplectic_to_hamming(matrix) if weight == "pauli" else matrix.copy()
        mask = (1 << n) - 1
        self._sector_masks = tuple(
            np.array(
                [(sector >> (64 * jj)) & ((1 << 64) - 1) for jj in range((2 * n + 63) // 64)],
                dtype=np.uint64,
            )
            for sector in (mask, mask << n)
        )
        self._information_sets: list[
            tuple[npt.NDArray[np.uint64], tuple[int, ...], tuple[int, ...]]
        ] = []
        self.ranks: tuple[int, ...] = ()
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.queries = 0
        self.cache_hits = 0
        self.generated = 0
        self.preparations = 0
        self.coefficient_depth = 0
        self._cache: dict[int, _PauliWeightDecision] = {}
        self._singles = {0: 0}
        self._x_singles = {0: 0}
        self._z_singles = {0: 0}
        for qubit in range(n):
            x, z = 1 << qubit, 1 << (qubit + n)
            sx, sz = self.syndrome(x), self.syndrome(z)
            self._x_singles.setdefault(sx, x)
            self._z_singles.setdefault(sz, z)
            self._singles.setdefault(sx, x)
            self._singles.setdefault(sz, z)
            self._singles.setdefault(sx ^ sz, x ^ z)

    @property
    def weight(self) -> Literal["pauli", "css"]:
        """The read-only metric shared by all witnesses, bounds, and cache entries."""
        return self._weight_metric

    def syndrome(self, pauli: int) -> int:
        """Compute the complete stabilizer syndrome of a packed symplectic Pauli."""
        return sum(((row & pauli).bit_count() & 1) << jj for jj, row in enumerate(self._checks))

    def decide(self, syndrome: int, max_weight: int) -> _PauliWeightDecision:
        """Prove or disprove that this syndrome has a representative within the threshold."""
        if not 0 <= syndrome < 1 << self.num_qubits or max_weight < 0:
            raise ValueError("Syndrome out of range or negative weight threshold")
        self.queries += 1
        known = self._cache.get(syndrome)
        if known is not None:
            upper = self._weight(known.representative)
            if upper <= max_weight or known.lower_bound > max_weight:
                self.cache_hits += 1
                return _PauliWeightDecision(
                    upper <= max_weight, known.representative, known.lower_bound
                )
        pauli = self._weight_one_representative(syndrome)
        if pauli is not None:
            decision = _PauliWeightDecision(
                self._weight(pauli) <= max_weight, pauli, int(bool(syndrome))
            )
        elif max_weight < 2:
            decision = _PauliWeightDecision(False, self._representative(syndrome), 2)
        else:
            decision = self._search(syndrome, max_weight)
        decision = self._merge_decisions(known, decision)
        self._cache[syndrome] = decision
        if len(self._cache) > self.cache_size:
            del self._cache[next(iter(self._cache))]
        return decision

    def _merge_decisions(
        self, known: _PauliWeightDecision | None, found: _PauliWeightDecision
    ) -> _PauliWeightDecision:
        """Keep the lightest checked representative and strongest proved bound for one coset."""
        if known is None:
            return found
        representative = min((known.representative, found.representative), key=self._weight)
        return _PauliWeightDecision(
            found.within_bound, representative, max(known.lower_bound, found.lower_bound)
        )

    def _representative(self, syndrome: int) -> int:
        return _xor_selected(self._inverse, syndrome)

    def _weight_one_representative(self, syndrome: int) -> int | None:
        if syndrome in self._singles:
            return self._singles[syndrome]
        if self.weight == "css":
            for sx, x in self._x_singles.items():
                z = self._z_singles.get(syndrome ^ sx)
                if z is not None:
                    return x ^ z
        return None

    def _weight(self, pauli: int) -> int:
        mask = (1 << self.num_qubits) - 1
        x, z = pauli & mask, (pauli >> self.num_qubits) & mask
        return max(x.bit_count(), z.bit_count()) if self.weight == "css" else (x | z).bit_count()

    def _candidate_weights(
        self,
        candidates: npt.NDArray[np.uint64],
    ) -> npt.NDArray[np.uint64]:
        if self.weight == "pauli":
            return np.bitwise_count(candidates).sum(axis=-1, dtype=np.uint64) // 2
        return np.maximum(
            *(
                np.bitwise_count(candidates & mask).sum(axis=-1, dtype=np.uint64)
                for mask in self._sector_masks
            )
        )

    def _search(self, syndrome: int, threshold: int) -> _PauliWeightDecision:
        if not self._information_sets:
            for generators, _, rank, pivot_columns in _get_information_set_generators(
                self._embedded, None
            ):
                self._information_sets.append(
                    (
                        _pack_bits(generators),
                        tuple(_binary_row_int(row) for row in generators),
                        tuple(map(int, pivot_columns)),
                    )
                )
                self.ranks += (rank,)
            self.preparations += 1
        pauli = self._representative(syndrome)
        n = self.num_qubits
        x = pauli & ((1 << n) - 1)
        z = pauli >> n
        embedded = pauli if self.weight == "css" else x | (z << n) | ((x ^ z) << (2 * n))
        offsets: list[npt.NDArray[np.uint64]] = []
        for packed, rows, pivots in self._information_sets:
            offset_word = embedded
            for row, pivot in zip(rows, pivots):
                if (offset_word >> pivot) & 1:
                    offset_word ^= row
            offsets.append(
                np.array(
                    [(offset_word >> (64 * jj)) & ((1 << 64) - 1) for jj in range(packed.shape[1])],
                    dtype=np.uint64,
                )
            )
        visited_min = n + 1
        best_pauli = pauli
        depth = 0
        while True:
            for (packed, _, _), offset in zip(self._information_sets, offsets, strict=True):
                supports_iter = (
                    [np.empty((1, 0), dtype=int)]
                    if depth == 0
                    else _iter_fixed_weight_supports(n, depth, self.batch_size)
                )
                for supports in supports_iter:
                    candidates = np.bitwise_xor.reduce(packed[supports], axis=1) ^ offset
                    self.generated += len(supports)
                    weights = self._candidate_weights(candidates)
                    good = np.flatnonzero(weights <= threshold)
                    if len(good):
                        word = _packed_row_int(candidates[good[0]])
                        representative = word & ((1 << (2 * n)) - 1)
                        return _PauliWeightDecision(True, representative, 2)
                    batch_min = int(weights.min())
                    if batch_min < visited_min:
                        visited_min = batch_min
                        word = _packed_row_int(candidates[int(weights.argmin())])
                        best_pauli = word & ((1 << (2 * n)) - 1)
            self.coefficient_depth = max(self.coefficient_depth, depth)
            unseen_hamming = _get_brouwer_zimmermann_lower_bound(
                n, list(self.ranks), completed_weight=depth, weight_divisor=1
            )
            # Either metric is at least half of the embedded Hamming weight.
            unseen_lower = (unseen_hamming + 1) // 2
            if depth == n:
                return _PauliWeightDecision(False, best_pauli, visited_min)
            if unseen_lower > threshold:
                global_lower = min(visited_min, unseen_lower)
                return _PauliWeightDecision(False, best_pauli, global_lower)
            depth += 1


####################################################################################################
# helper methods


def _iter_fixed_weight_supports(
    dimension: int, weight: int, batch_size: int
) -> Iterator[npt.NDArray[np.int_]]:
    """Yield coefficient supports for witness enumeration and affine-coset decisions.

    Unlike the compiled distance minimizer, these consumers need the individual supports and
    candidate words, including words that do not improve a distance upper bound.
    """
    num_supports = math.comb(dimension, weight)
    max_rank = np.iinfo(np.int64).max
    if num_supports <= max_rank:
        # Unrank consecutive combinadic indices as whole NumPy batches.
        binomial_tables = [
            np.asarray(
                [min(math.comb(value, index), max_rank) for value in range(dimension)],
                dtype=np.int64,
            )
            for index in range(1, weight + 1)
        ]
        for start in range(0, num_supports, batch_size):
            remainders = np.arange(
                start,
                min(start + batch_size, num_supports),
                dtype=np.int64,
            )
            supports = np.empty((len(remainders), weight), dtype=np.int_)
            for index in range(weight, 0, -1):
                table = binomial_tables[index - 1]
                choices = np.searchsorted(table, remainders, side="right") - 1
                supports[:, index - 1] = choices
                remainders -= table[choices]
            yield supports
        return

    # Stream supports when their ranks cannot be represented exactly by int64.
    combinations = itertools.combinations(range(dimension), weight)
    flattened = itertools.chain.from_iterable(combinations)
    while True:
        flat_batch = np.fromiter(
            itertools.islice(flattened, batch_size * weight),
            dtype=np.int_,
        )
        if not len(flat_batch):
            return
        yield flat_batch.reshape(-1, weight)


def _binary_row_int(row: npt.NDArray[np.uint8]) -> int:
    return int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")


def _packed_row_int(row: npt.NDArray[np.uint64]) -> int:
    return sum(int(word) << (64 * jj) for jj, word in enumerate(row))


def _xor_selected(rows: Sequence[int], support: int) -> int:
    word = 0
    while support:
        low = support & -support
        word ^= rows[low.bit_length() - 1]
        support ^= low
    return word
