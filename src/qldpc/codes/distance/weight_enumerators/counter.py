# SPDX-License-Identifier: Apache-2.0

"""Exact binary CSS logical spectra and minimum-weight physical Pauli counts.

The public entry point is ``CSSLogicalWeightCounter``.  It counts physical representatives,
ignoring global phase, rather than equivalence classes modulo stabilizers.  Its backends live in
sibling modules:

- ``dual``: stabilizer enumeration and the MacWilliams identity, which enumerate
  stabilizer words rather than the generally much larger logical space.
- ``supports``: minimum-weight supports from Brouwer--Zimmermann information sets or an
  exhaustive walk.  BZ enumeration stops only when its unseen-word bound is strictly greater than
  the requested weight.
- ``collision``: opt-in discovery of supports by joining physical half-shells.
- ``lifting``: lifting of the supports to full physical-Pauli counts.
- ``rows``: packed rows, local Clifford transport and storage budgets.

References:

- https://en.wikipedia.org/wiki/MacWilliams_identity
- https://arxiv.org/abs/1603.06757 (BZ information-set coverage)
- https://arxiv.org/abs/2607.19574 (the annealing objective and physical representatives)
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from .. import _binary as binary
from .. import brouwer_zimmermann as bz
from .collision import recover_supports
from .dual import (
    get_macwilliams_minimum,
    get_macwilliams_spectrum,
    get_stabilizer_histograms,
)
from .lifting import count_minimum_paulis
from .rows import BinaryMatrix, PackedMatrix, StorageBudget, pack, transform_rows, unpack
from .supports import (
    get_bz_supports,
    get_exhaustive_minimum,
    get_exhaustive_supports,
)

WeightEnumeratorMethod = Literal["auto", "dual", "brouwer_zimmermann", "brute_force", "collision"]


####################################################################################################
# exact logical weight results


@dataclass(frozen=True)
class CSSLogicalWeightSpectrum:
    """Exact nontrivial homogeneous representative counts through ``max_weight``.

    ``counts_x[w]`` counts X-only words in ``ker(H_Z) \\ row(H_X)`` of weight w, and similarly
    for Z.  These are not the full Pauli spectrum.  A truncated all-zero result proves absence
    only through ``max_weight``.
    """

    counts_x: tuple[int, ...]
    counts_z: tuple[int, ...]

    @property
    def max_weight(self) -> int:
        """Largest weight certified by this spectrum."""
        return len(self.counts_x) - 1


@dataclass(frozen=True)
class CSSMinimumLogicalWeights:
    """Exact global minimum and its multiplicities.

    ``distance`` is None when no nontrivial logical exists (k=0).  ``count_x`` and ``count_z``
    refer to the global minimum, not each sector's individual minimum.  ``count_pauli`` includes
    Y and mixed representatives, or is None when full counting was not requested.
    """

    distance: int | None
    count_x: int
    count_z: int
    count_pauli: int | None

    @property
    def count_homogeneous(self) -> int:
        """Total number of X-only and Z-only representatives at the minimum weight."""
        return self.count_x + self.count_z


####################################################################################################
# prepared CSS counting and conjugation


class CSSLogicalWeightCounter:
    """Prepared exact counter for a binary CSS stabilizer code.

    Args:
        checks_x: Binary X-stabilizer checks, including optional redundant rows.
        checks_z: Binary Z-stabilizer checks.  The two row spaces must commute.
        memory_limit_bytes: Conservative working-storage budget for support enumeration.
            This is not a limit on the interpreter's total resident memory.
        batch_size: Maximum native enumeration chunk, providing interruption opportunities.

    Instances own their matrices.  Conjugation returns a new counter without changing the source;
    ranks and precomputed transforms can be reused across annealing proposals.  Constructor
    checks are stabilizers, never noncommuting subsystem gauge generators.

    Exact enumeration remains exponential.  ``"auto"`` uses dual enumeration for small stabilizer
    ranks, recognizes cheap low-weight BZ searches at moderate ranks, and uses BZ above rank 24.
    Explicit selection is available for reproducible comparisons.
    Full generic-CSS Pauli counts require distinct pure minimum supports and can be output-sized.
    BZ packs each information set once per call and scans coefficient leaves in bounded batches;
    neither minimum scores nor support outputs are cached across calls.
    The opt-in ``"collision"`` method discovers distance by joining physical half-shells of
    increasing weight, retaining all distinct pure minimum supports for full-Pauli lifting.
    It is suited to low-distance codes with large logical dimension, but its half-shell storage
    and collision-chain work can still be exponential.
    It is never selected by ``"auto"``.
    """

    def __init__(
        self,
        checks_x: npt.ArrayLike,
        checks_z: npt.ArrayLike,
        *,
        memory_limit_bytes: int = 256 << 20,
        batch_size: int = 1 << 16,
    ) -> None:
        binary._assert_binary(checks_x, "checks_x")
        binary._assert_binary(checks_z, "checks_z")
        x = binary._as_binary_matrix(checks_x)
        z = binary._as_binary_matrix(checks_z, num_cols=x.shape[1])
        if np.any((x @ z.T) % 2):
            raise ValueError("CSS stabilizer checks must commute; subsystem gauges are unsupported")
        if any(
            not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1
            for value in (memory_limit_bytes, batch_size)
        ):
            raise ValueError("memory_limit_bytes and batch_size must be positive integers")
        x, z = binary._get_independent_rows(x), binary._get_independent_rows(z)
        self.num_qubits = x.shape[1]
        self.dimension = self.num_qubits - len(x) - len(z)
        self.memory_limit_bytes = int(memory_limit_bytes)
        self.batch_size = int(batch_size)
        self._x, self._z = pack(x), pack(z)
        self._symmetric = np.array_equal(x, z)
        self._basis_cache: tuple[tuple[BinaryMatrix, BinaryMatrix], ...] | None = None

    @property
    def checks_x(self) -> BinaryMatrix:
        """Copy of an independent X-stabilizer basis."""
        return unpack(self._x, self.num_qubits)

    @property
    def checks_z(self) -> BinaryMatrix:
        """Copy of an independent Z-stabilizer basis."""
        return unpack(self._z, self.num_qubits)

    @functools.cached_property
    def is_self_dual(self) -> bool:
        """Whether the actual X/Z row spaces coincide, independent of distance promises."""
        return self._symmetric or np.array_equal(
            binary._get_independent_rows(self.checks_x),
            binary._get_independent_rows(self.checks_z),
        )

    def get_spectrum(self, max_weight: int | None = None) -> CSSLogicalWeightSpectrum:
        """Compute exact homogeneous spectra using stabilizer enumeration and MacWilliams.

        Only coefficients through max_weight are returned; None requests the complete spectrum.
        Transform accumulation and final coefficients use arbitrary-precision integers.
        A dual rank above 62 is rejected because a single native histogram cannot represent its
        enumeration count; use BZ minimum counting instead.
        """
        limit = self.num_qubits if max_weight is None else max_weight
        if not isinstance(limit, (int, np.integer)) or not 0 <= limit <= self.num_qubits:
            raise ValueError("max_weight must be between zero and num_qubits")
        limit = int(limit)
        if not self.dimension:
            zeros = (0,) * (limit + 1)
            return CSSLogicalWeightSpectrum(zeros, zeros)
        histogram_x, histogram_z = self._get_histograms()
        counts_x, counts_z = get_macwilliams_spectrum(
            histogram_x, histogram_z, len(self._x), len(self._z), self.num_qubits, limit
        )
        return CSSLogicalWeightSpectrum(tuple(counts_x), tuple(counts_z))

    def get_minimum(
        self,
        *,
        method: WeightEnumeratorMethod = "auto",
        full_pauli: bool = True,
    ) -> CSSMinimumLogicalWeights:
        """Find the exact global distance and multiplicities.

        ``full_pauli=False`` requests only homogeneous counts and returns count_pauli=None.
        For a genuinely self-dual code, the full count is three times one sector count.
        Otherwise shortened-space ranks lift distinct minimum pure supports to full Paulis.
        A memory-budget failure raises MemoryError, never a partial count.
        """
        if method not in ("auto", "dual", "brouwer_zimmermann", "brute_force", "collision"):
            raise ValueError(f"Unknown weight-enumerator method: {method!r}")
        if not self.dimension:
            return CSSMinimumLogicalWeights(None, 0, 0, 0 if full_pauli else None)

        # Phase 1: the global distance and the X-only and Z-only counts at that weight.
        selected = self._choose_method() if method == "auto" else method
        found = self._find_minimum(selected)
        distance, count_x, count_z = found.distance, found.count_x, found.count_z
        if not full_pauli:
            return CSSMinimumLogicalWeights(distance, count_x, count_z, None)
        if self._symmetric:
            return CSSMinimumLogicalWeights(distance, count_x, count_z, 3 * count_x)

        # Phase 2: lift the distinct minimum supports of both sectors to physical Paulis.
        supports_x, supports_z = found.supports_x, found.supports_z
        if supports_x is None or supports_z is None:
            (basis_x, labels_x), (basis_z, labels_z) = self._get_sector_bases()
            supports_x = self._get_supports_at(basis_x, labels_x, distance, count_x)
            supports_z = self._get_supports_at(basis_z, labels_z, distance, count_z)
        full = count_minimum_paulis(
            supports_x, supports_z, self._x, self._z, distance, self._get_budget()
        )
        return CSSMinimumLogicalWeights(distance, count_x, count_z, full)

    def conjugated_cnot(self, control: int, target: int) -> CSSLogicalWeightCounter:
        """Return a CNOT-conjugated counter; qubit indices are zero-based."""
        return self.conjugated([control, target], [[1, 0], [1, 1]], [[1, 1], [0, 1]])

    def conjugated_swel(self, targets: Sequence[int]) -> CSSLogicalWeightCounter:
        """Apply the four-qubit SWEL-preserving gate to four ordered targets.

        Its binary local rows are (7,11,13,14), not the transvection with an omitted reversal.
        The gate is also valid on general CSS inputs.
        """
        if len(targets) != 4:
            raise ValueError("The SWEL-preserving gate requires four targets")
        matrix = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]
        return self.conjugated(targets, matrix, matrix)

    def conjugated(
        self,
        targets: Sequence[int],
        matrix_x: npt.ArrayLike,
        matrix_z: npt.ArrayLike,
    ) -> CSSLogicalWeightCounter:
        """Apply a local CSS-preserving Clifford specified by its X/Z binary matrices.

        Matrices act on column supports.  They must obey matrix_x @ matrix_z.T = I over GF(2).
        Between one and 64 distinct, in-range targets are supported; physical widths can exceed 64.
        """
        positions = tuple(targets)
        size = len(positions)
        if (
            not 1 <= size <= 64
            or any(
                not isinstance(index, (int, np.integer))
                or isinstance(index, bool)
                or not 0 <= index < self.num_qubits
                for index in positions
            )
            or len(set(positions)) != size
        ):
            raise ValueError("Clifford targets must be 1..64 distinct in-range qubit indices")
        binary._assert_binary(matrix_x, "matrix_x")
        binary._assert_binary(matrix_z, "matrix_z")
        x, z = np.asarray(matrix_x, dtype=np.uint8), np.asarray(matrix_z, dtype=np.uint8)
        if x.shape != (size, size) or z.shape != (size, size):
            raise ValueError("Local Clifford matrices must be square and match the target count")
        if not np.array_equal((x @ z.T) % 2, np.eye(size, dtype=np.uint8)):
            raise ValueError("Local X/Z matrices must preserve the CSS commutation pairing")

        # Transport the packed stabilizers through the local matrices.
        indices = np.asarray(positions, dtype=np.int64)
        new_x = transform_rows(self._x, indices, pack(x)[:, 0])
        new_z = (
            new_x
            if self._symmetric and np.array_equal(x, z)
            else transform_rows(self._z, indices, pack(z)[:, 0])
        )
        result = object.__new__(CSSLogicalWeightCounter)
        result.num_qubits, result.dimension = self.num_qubits, self.dimension
        result.memory_limit_bytes, result.batch_size = self.memory_limit_bytes, self.batch_size
        result._x, result._z = new_x, new_z
        result._symmetric = np.array_equal(new_x, new_z)
        result._basis_cache = None

        # A physical Clifford transports the cached bases but does not change their quotient labels.
        if self._basis_cache is not None:
            transformed = []
            for (basis, labels), matrix in zip(self._basis_cache, (x, z), strict=True):
                rows = transform_rows(pack(basis), indices, pack(matrix)[:, 0])
                transformed.append((unpack(rows, self.num_qubits), labels))
            result._basis_cache = tuple(transformed)
        return result

    ################################################################################################
    # method dispatch

    def _choose_method(self) -> Literal["dual", "brouwer_zimmermann"]:
        """Compare conservative coefficient work only where dual scans are moderately costly."""
        rank = max(len(self._x), len(self._z))
        if rank <= 20:
            return "dual"
        if rank > 24:
            return "brouwer_zimmermann"
        if self._get_basis_storage() > self.memory_limit_bytes:
            return "dual"
        sectors = self._get_sector_bases()
        upper = min(
            int(np.count_nonzero(basis[np.any(labels, axis=1)], axis=1).min())
            for basis, labels in sectors
        )
        work = sum(
            math.comb(len(basis), weight)
            for basis, _ in sectors
            for weight in range(1, min(upper, len(basis)) + 1)
        )
        dual_work = (1 << len(self._x)) + (1 << len(self._z))
        # Unlike distance-only search, counting must exhaust ties and pays for label/dedup work.
        return "brouwer_zimmermann" if 8 * work < dual_work else "dual"

    def _find_minimum(self, method: str) -> _MinimumSearch:
        """Run one backend; only some of them also retain the minimum supports they find."""
        if method == "collision":
            return self._find_minimum_collision()
        if method == "dual":
            return self._find_minimum_dual()
        if method == "brute_force":
            return self._find_minimum_exhaustive()
        return self._find_minimum_bz()

    def _find_minimum_dual(self) -> _MinimumSearch:
        """Count by MacWilliams, which does not produce the supports."""
        histogram_x, histogram_z = self._get_histograms()
        distance, count_x, count_z = get_macwilliams_minimum(
            histogram_x, histogram_z, len(self._x), len(self._z), self.num_qubits
        )
        return _MinimumSearch(distance, count_x, count_z, None, None)

    def _find_minimum_exhaustive(self) -> _MinimumSearch:
        """Walk every coefficient combination of each sector basis."""
        (basis_x, labels_x), (basis_z, labels_z) = self._get_sector_bases()
        budget = self._get_budget()
        distance_x, words_x = get_exhaustive_minimum(basis_x, labels_x, budget)
        if self._symmetric:
            distance_z, words_z = distance_x, words_x
        else:
            distance_z, words_z = get_exhaustive_minimum(basis_z, labels_z, budget)
        # Counts refer to the global minimum, so a sector with a heavier minimum contributes none.
        distance = min(distance_x, distance_z)
        supports_x = words_x if distance_x == distance else self._get_no_supports()
        supports_z = words_z if distance_z == distance else self._get_no_supports()
        return _MinimumSearch(distance, len(supports_x), len(supports_z), supports_x, supports_z)

    def _find_minimum_bz(self) -> _MinimumSearch:
        """Find the distance by BZ, then collect all supports of that weight in each sector."""
        sectors = self._get_sector_bases()
        distance = bz._get_distance_brouwer_zimmermann_many(
            sectors[:1] if self._symmetric else sectors, cutoff=0, block_size=15
        )
        budget = self._get_budget()
        supports_x = get_bz_supports(*sectors[0], distance, budget)
        supports_z = (
            supports_x if self._symmetric else get_bz_supports(*sectors[1], distance, budget)
        )
        return _MinimumSearch(distance, len(supports_x), len(supports_z), supports_x, supports_z)

    def _find_minimum_collision(self) -> _MinimumSearch:
        """Exhaust increasing physical weights, without cached/family distance assumptions."""
        x, z = self._get_sector_bases()

        # A complement of H_Z in ker(H_X) pairs nondegenerately with ker(H_Z)/row(H_X).
        # The omitted H_Z rows already annihilate every X normalizer word (and conversely).
        storage = self._get_basis_storage() + 4 * self.num_qubits * (self.num_qubits + 128)
        if storage > self.memory_limit_bytes:
            raise MemoryError("Collision map setup exceeds memory_limit_bytes")
        maps = (
            (self.checks_z, z[0][len(self._z) :]),
            (self.checks_x, x[0][len(self._x) :]),
        )
        columns = [
            (pack(np.ascontiguousarray(checks.T)), pack(np.ascontiguousarray(logicals.T)))
            for checks, logicals in maps
        ]

        # The first weight with any support in either sector is the global distance.
        for weight in range(1, self.num_qubits + 1):
            supports_x = recover_supports(
                *columns[0],
                weight,
                memory_limit_bytes=self.memory_limit_bytes,
                reserved_bytes=storage,
                batch_size=self.batch_size,
            )
            supports_z = (
                supports_x
                if self._symmetric
                else recover_supports(
                    *columns[1],
                    weight,
                    memory_limit_bytes=self.memory_limit_bytes,
                    reserved_bytes=storage + 12 * supports_x.nbytes,
                    batch_size=self.batch_size,
                )
            )
            if len(supports_x) or len(supports_z):
                return _MinimumSearch(
                    weight, len(supports_x), len(supports_z), supports_x, supports_z
                )
        raise ArithmeticError("A positive-dimensional CSS code has no nontrivial logicals")

    def _get_supports_at(
        self, basis: BinaryMatrix, labels: BinaryMatrix, distance: int, count: int
    ) -> PackedMatrix:
        """Recover one sector's supports at a known minimum weight, if it has any."""
        if not count:
            return self._get_no_supports()
        if len(basis) <= 20:
            return get_exhaustive_supports(basis, labels, distance, self._get_budget())
        return get_bz_supports(basis, labels, distance, self._get_budget())

    ################################################################################################
    # cached bases and storage accounting

    def _get_sector_bases(self) -> tuple[tuple[BinaryMatrix, BinaryMatrix], ...]:
        """Cache coefficient bases, not scores; conjugation transports these bases exactly."""
        if self._basis_cache is not None:
            return self._basis_cache
        if self._get_basis_storage() > self.memory_limit_bytes:
            raise MemoryError("Logical-basis setup exceeds memory_limit_bytes")
        x, z = self.checks_x, self.checks_z
        sector_x = binary._get_nested_code_basis(_get_kernel(z), x)
        sector_z = sector_x if self._symmetric else binary._get_nested_code_basis(_get_kernel(x), z)
        self._basis_cache = sector_x, sector_z
        return self._basis_cache

    def _get_histograms(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        return get_stabilizer_histograms(
            self._x,
            self._z,
            self.num_qubits,
            symmetric=self._symmetric,
            batch_size=self.batch_size,
        )

    def _get_basis_storage(self) -> int:
        dimensions = self.num_qubits - len(self._z)
        if not self._symmetric:
            dimensions += self.num_qubits - len(self._x)
        return 4 * dimensions * (self.num_qubits + self.dimension)

    def _get_budget(self) -> StorageBudget:
        return StorageBudget(self.memory_limit_bytes, self._get_basis_storage(), self.batch_size)

    def _get_no_supports(self) -> PackedMatrix:
        return np.empty((0, self._x.shape[1]), dtype=np.uint64)


class _MinimumSearch(NamedTuple):
    """Distance and sector counts, with the supports when the backend retained them."""

    distance: int
    count_x: int
    count_z: int
    supports_x: PackedMatrix | None
    supports_z: PackedMatrix | None


def get_minimum_logical_weights_css(
    checks_x: npt.ArrayLike,
    checks_z: npt.ArrayLike,
    *,
    method: WeightEnumeratorMethod = "auto",
    full_pauli: bool = True,
    memory_limit_bytes: int = 256 << 20,
    batch_size: int = 1 << 16,
) -> CSSMinimumLogicalWeights:
    """Compute exact binary CSS distance and minimum multiplicities.

    See CSSLogicalWeightCounter for method, field, memory and representative-count conventions.
    Prepare a counter once when evaluating multiple Clifford-conjugated proposals.
    """
    return CSSLogicalWeightCounter(
        checks_x, checks_z, memory_limit_bytes=memory_limit_bytes, batch_size=batch_size
    ).get_minimum(method=method, full_pauli=full_pauli)


####################################################################################################
# helper methods


def _get_kernel(matrix: BinaryMatrix) -> BinaryMatrix:
    """Basis of the GF(2) null space, from the reduced row echelon form."""
    reduced, _, pivots = binary._row_reduce_binary(matrix)
    free = np.setdiff1d(np.arange(matrix.shape[1]), pivots)
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    basis[:, free] = np.eye(len(free), dtype=np.uint8)
    basis[:, pivots] = reduced[: len(pivots), free].T
    return basis
