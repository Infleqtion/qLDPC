# SPDX-License-Identifier: Apache-2.0

"""Exact binary CSS logical spectra and minimum-weight physical Pauli counts.

The dual backend enumerates stabilizer words, not the generally much larger logical space, and
uses the MacWilliams identity.  BZ support enumeration stops only when its unseen-word bound is
strictly greater than the requested weight.  All methods count physical representatives, ignoring
global phase, rather than equivalence classes modulo stabilizers.

References:

- https://en.wikipedia.org/wiki/MacWilliams_identity
- https://arxiv.org/abs/1603.06757 (BZ information-set coverage)
- https://arxiv.org/abs/2607.19574 (the annealing objective and physical representatives)
"""

from __future__ import annotations

import functools
import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import njit

from . import _binary as binary
from . import brouwer_zimmermann as bz
from ._binary_numba import popcount_word, trailing_zeros
from ._collision import recover_supports

BinaryMatrix = npt.NDArray[np.uint8]
PackedMatrix = npt.NDArray[np.uint64]
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
        """The X/Z-only multiplicity used by the original Julia annealing search."""
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
        self._x, self._z = _pack(x), _pack(z)
        self._symmetric = np.array_equal(x, z)
        self._basis_cache: tuple[tuple[BinaryMatrix, BinaryMatrix], ...] | None = None

    @property
    def checks_x(self) -> BinaryMatrix:
        """Copy of an independent X-stabilizer basis."""
        return _unpack(self._x, self.num_qubits)

    @property
    def checks_z(self) -> BinaryMatrix:
        """Copy of an independent Z-stabilizer basis."""
        return _unpack(self._z, self.num_qubits)

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
        bx, bz = self._histograms()
        x, z = [], []
        for weight, coefficients in enumerate(
            itertools.islice(_iter_krawtchouk(self.num_qubits), limit + 1)
        ):
            ax = _dual_coefficient(bz, coefficients, len(self._z)) - int(bx[weight])
            az = _dual_coefficient(bx, coefficients, len(self._x)) - int(bz[weight])
            if ax < 0 or az < 0:
                raise ArithmeticError("Negative logical multiplicity")
            x.append(ax)
            z.append(az)
        return CSSLogicalWeightSpectrum(tuple(x), tuple(z))

    def get_minimum(
        self,
        *,
        method: WeightEnumeratorMethod = "auto",
        full_pauli: bool = True,
    ) -> CSSMinimumLogicalWeights:
        """Find the exact global distance and multiplicities, with no distance-only cutoff.

        ``full_pauli=False`` requests only homogeneous counts and returns count_pauli=None.
        For a genuinely self-dual code, the full count is three times one sector count.
        Otherwise shortened-space ranks lift distinct minimum pure supports to full Paulis.
        A memory-budget failure raises MemoryError, never a partial count.
        """
        if method not in ("auto", "dual", "brouwer_zimmermann", "brute_force", "collision"):
            raise ValueError(f"Unknown weight-enumerator method: {method!r}")
        if not self.dimension:
            return CSSMinimumLogicalWeights(None, 0, 0, 0 if full_pauli else None)
        selected = method
        if selected == "auto":
            selected = self._choose_method()
        supports_x: PackedMatrix | None = None
        supports_z: PackedMatrix | None = None
        if selected == "collision":
            minimum, supports_x, supports_z = self._collision_minimum()
            ax, az = len(supports_x), len(supports_z)
        elif selected == "dual":
            minimum, ax, az = self._dual_minimum()
        elif selected == "brute_force":
            x, z = self._sector_bases()
            dx, words_x = self._exhaustive_minimum(*x)
            dz, words_z = (dx, words_x) if self._symmetric else self._exhaustive_minimum(*z)
            minimum = min(dx, dz)
            supports_x, supports_z = (
                words_x if dx == minimum else np.empty((0, self._x.shape[1]), dtype=np.uint64),
                words_z if dz == minimum else np.empty((0, self._x.shape[1]), dtype=np.uint64),
            )
            ax, az = len(supports_x), len(supports_z)
        else:
            x, z = self._sector_bases()
            minimum = bz._get_distance_brouwer_zimmermann_many(
                (x,) if self._symmetric else (x, z),
                cutoff=0,
                block_size=15,
            )
            supports_x = self._supports(*x, minimum, exhaustive=False)
            supports_z = (
                supports_x if self._symmetric else self._supports(*z, minimum, exhaustive=False)
            )
            ax, az = len(supports_x), len(supports_z)
        if not full_pauli:
            return CSSMinimumLogicalWeights(minimum, ax, az, None)
        if self._symmetric:
            return CSSMinimumLogicalWeights(minimum, ax, az, 3 * ax)
        if supports_x is None or supports_z is None:
            x, z = self._sector_bases()
            supports_x = (
                self._supports(*x, minimum, exhaustive=len(x[0]) <= 20)
                if ax
                else np.empty((0, self._x.shape[1]), dtype=np.uint64)
            )
            supports_z = (
                self._supports(*z, minimum, exhaustive=len(z[0]) <= 20)
                if az
                else np.empty((0, self._x.shape[1]), dtype=np.uint64)
            )
        full = self._lift(supports_x, supports_z, minimum)
        return CSSMinimumLogicalWeights(minimum, ax, az, full)

    def conjugated_cnot(self, control: int, target: int) -> CSSLogicalWeightCounter:
        """Return a CNOT-conjugated counter; qubit indices are zero-based."""
        return self.conjugated([control, target], [[1, 0], [1, 1]], [[1, 1], [0, 1]])

    def conjugated_swel(self, targets: Sequence[int]) -> CSSLogicalWeightCounter:
        """Apply the four-qubit SWEL-preserving gate, with Julia's ordered-target convention.

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
        indices = np.asarray(positions, dtype=np.int64)
        new_x = _transform(self._x, indices, _pack(x)[:, 0])
        new_z = (
            new_x
            if self._symmetric and np.array_equal(x, z)
            else _transform(self._z, indices, _pack(z)[:, 0])
        )
        result = object.__new__(CSSLogicalWeightCounter)
        result.num_qubits, result.dimension = self.num_qubits, self.dimension
        result.memory_limit_bytes, result.batch_size = self.memory_limit_bytes, self.batch_size
        result._x, result._z = new_x, new_z
        result._symmetric = np.array_equal(new_x, new_z)
        result._basis_cache = None
        if self._basis_cache is not None:
            transformed = []
            # A physical Clifford transports the rows but does not change their quotient labels.
            for (basis, labels), matrix in zip(self._basis_cache, (x, z), strict=True):
                rows = _transform(_pack(basis), indices, _pack(matrix)[:, 0])
                transformed.append((_unpack(rows, self.num_qubits), labels))
            result._basis_cache = tuple(transformed)
        return result

    def _choose_method(self) -> Literal["dual", "brouwer_zimmermann"]:
        """Compare conservative coefficient work only where dual scans are moderately costly."""
        rank = max(len(self._x), len(self._z))
        if rank <= 20:
            return "dual"
        if rank > 24:
            return "brouwer_zimmermann"
        if self._basis_storage() > self.memory_limit_bytes:
            return "dual"
        sectors = self._sector_bases()
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

    def _histograms(self) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        if max(len(self._x), len(self._z)) > 62:
            raise OverflowError("Dual ranks above 62 require BZ instead of native histograms")
        counts = []
        for rows in (self._x,) if self._symmetric else (self._x, self._z):
            histogram = np.zeros(self.num_qubits + 1, dtype=np.int64)
            histogram[0] = 1
            current = np.zeros(rows.shape[1], dtype=np.uint64)
            for start in range(1, 1 << len(rows), self.batch_size):
                _histogram_chunk(
                    rows,
                    current,
                    histogram,
                    start,
                    min(start + self.batch_size, 1 << len(rows)),
                )
            counts.append(histogram)
        return counts[0], counts[0] if self._symmetric else counts[1]

    def _dual_minimum(self) -> tuple[int, int, int]:
        bx, bz = self._histograms()
        coefficients = _safe_coefficients(self.num_qubits, max(len(self._x), len(self._z)))
        minimum, ax, az = _minimum_native(bx, bz, len(self._x), len(self._z), coefficients)
        if minimum >= 0:
            return int(minimum), int(ax), int(az)
        for weight, row in enumerate(_iter_krawtchouk(self.num_qubits)):
            ax = _dual_coefficient(bz, row, len(self._z)) - int(bx[weight])
            az = _dual_coefficient(bx, row, len(self._x)) - int(bz[weight])
            if ax < 0 or az < 0:
                raise ArithmeticError("Negative logical multiplicity")
            if ax or az:
                return weight, ax, az
        raise ArithmeticError("A positive-dimensional CSS code has no nontrivial logicals")

    def _sector_bases(self) -> tuple[tuple[BinaryMatrix, BinaryMatrix], ...]:
        """Cache coefficient bases, not scores; conjugation transports these bases exactly."""
        if self._basis_cache is not None:
            return self._basis_cache
        if self._basis_storage() > self.memory_limit_bytes:
            raise MemoryError("Logical-basis setup exceeds memory_limit_bytes")
        x, z = self.checks_x, self.checks_z
        sector_x = binary._get_nested_code_basis(_kernel(z), x)
        sector_z = sector_x if self._symmetric else binary._get_nested_code_basis(_kernel(x), z)
        self._basis_cache = sector_x, sector_z
        return self._basis_cache

    def _basis_storage(self) -> int:
        dimensions = self.num_qubits - len(self._z)
        if not self._symmetric:
            dimensions += self.num_qubits - len(self._x)
        return 4 * dimensions * (self.num_qubits + self.dimension)

    def _collision_minimum(self) -> tuple[int, PackedMatrix, PackedMatrix]:
        """Exhaust increasing physical weights, without cached/family distance assumptions."""
        x, z = self._sector_bases()
        # A complement of H_Z in ker(H_X) pairs nondegenerately with ker(H_Z)/row(H_X).
        # The omitted H_Z rows already annihilate every X normalizer word (and conversely).
        storage = self._basis_storage() + 4 * self.num_qubits * (self.num_qubits + 128)
        if storage > self.memory_limit_bytes:
            raise MemoryError("Collision map setup exceeds memory_limit_bytes")
        maps = (
            (self.checks_z, z[0][len(self._z) :]),
            (self.checks_x, x[0][len(self._x) :]),
        )
        columns = [
            (_pack(np.ascontiguousarray(checks.T)), _pack(np.ascontiguousarray(logicals.T)))
            for checks, logicals in maps
        ]
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
                return weight, supports_x, supports_z
        raise ArithmeticError("A positive-dimensional CSS code has no nontrivial logicals")

    def _supports(
        self, basis: BinaryMatrix, labels: BinaryMatrix, weight: int, *, exhaustive: bool
    ) -> PackedMatrix:
        if exhaustive and len(basis) > 62:
            raise OverflowError("Exhaustive coefficient ranks above 62 are unsupported; use BZ")
        raw_sets = (
            [bz._InformationSet(basis, labels, len(basis), np.empty(0, dtype=int))]
            if exhaustive
            else bz._get_information_set_generators(basis, labels)
        )
        divisor = 2 if np.all(np.count_nonzero(basis, axis=1) % 2 == 0) else 1
        width = (self.num_qubits + 63) // 64
        max_coverage = 1
        if not exhaustive:
            # The distance selector uses >=, so target+1 requests the strict counting certificate.
            raw_sets = bz._select_brouwer_zimmermann_information_sets(
                raw_sets,
                dimension=len(basis),
                upper_bound=weight + 1,
                weight_divisor=divisor,
            )
            set_storage = (
                basis.nbytes
                + labels.nbytes
                + 8 * len(basis) * (width + (labels.shape[1] + 63) // 64)
            )
            # Reserve both the retained family and transient basis-construction workspace.
            available = self.memory_limit_bytes - self._basis_storage() - 1024
            max_sets = min(8, available // max(1, 2 * set_storage))
            if max_sets > sum(item.rank == len(basis) for item in raw_sets):
                overlapping = bz._get_overlapping_information_sets(
                    basis,
                    labels,
                    raw_sets,
                    upper_bound=weight + 1,
                    weight_divisor=divisor,
                    max_sets=max_sets,
                )
                if overlapping is not None:
                    raw_sets, max_coverage = overlapping
        sets = []
        for rows, tags, rank, pivots in raw_sets:
            assert tags is not None
            sets.append((rows, tags, rank, pivots))
        ranks = [rank for _, _, rank, _ in sets]
        result = bytearray()
        storage = sum(
            rows.nbytes + tags.nbytes + 8 * len(rows) * (width + (tags.shape[1] + 63) // 64)
            for rows, tags, _, _ in sets
        )
        output, max_words = self._support_buffer(
            storage, max_candidates=(1 << len(basis)) - 1 if exhaustive else None
        )
        chunk_size = len(output)
        if exhaustive:
            packed = _pack(basis)
            current = np.zeros(width, dtype=np.uint64)
            stabilizer_rank = len(basis) - labels.shape[1]
            for start in range(1, 1 << len(basis), chunk_size):
                count = _exhaustive_supports(
                    packed,
                    current,
                    stabilizer_rank,
                    start,
                    min(start + chunk_size, 1 << len(basis)),
                    weight,
                    output,
                )
                _append_supports(result, output, count, max_words)
            return _support_array(result, width)
        packed_sets = [(_pack(rows), _pack(tags)) for rows, tags, _, _ in sets]
        for depth_weight in range(1, len(basis) + 1):
            for packed, packed_tags in packed_sets:
                support = np.zeros(depth_weight, dtype=np.int64)
                prefixes = np.zeros((depth_weight + 1, width), dtype=np.uint64)
                tag_prefixes = np.zeros((depth_weight + 1, packed_tags.shape[1]), dtype=np.uint64)
                depth = 0
                while depth >= 0:
                    depth, count = _scan_supports(
                        packed,
                        packed_tags,
                        support,
                        prefixes,
                        tag_prefixes,
                        depth,
                        weight,
                        chunk_size,
                        output,
                    )
                    _append_supports(result, output, count, max_words)
            lower = bz._get_brouwer_zimmermann_lower_bound(
                len(basis),
                ranks,
                completed_weight=depth_weight,
                weight_divisor=divisor,
                max_coverage=max_coverage,
            )
            # Equality can still hide more minimum-weight representatives.
            if lower > weight:
                break
        words = _support_array(result, width)
        if len(sets) == 1:
            return words
        if width == 1:
            return np.unique(words[:, 0]).reshape(-1, 1)
        return np.unique(_support_keys(words)).view(np.uint64).reshape(-1, width)

    def _exhaustive_minimum(
        self, basis: BinaryMatrix, labels: BinaryMatrix
    ) -> tuple[int, PackedMatrix]:
        if len(basis) > 62:
            raise OverflowError("Exhaustive ranks above 62 are unsupported; use BZ")
        rows = _pack(basis)
        output, max_words = self._support_buffer(
            basis.nbytes + labels.nbytes + rows.nbytes, max_candidates=(1 << len(basis)) - 1
        )
        current = np.zeros(rows.shape[1], dtype=np.uint64)
        best = self.num_qubits + 1
        result = bytearray()
        for start in range(1, 1 << len(rows), len(output)):
            previous = best
            best, count = _scan_exhaustive_minimum(
                rows,
                current,
                len(basis) - labels.shape[1],
                start,
                min(start + len(output), 1 << len(rows)),
                best,
                output,
            )
            if best < previous:
                result.clear()
            _append_supports(result, output, count, max_words)
        return int(best), _support_array(result, rows.shape[1])

    def _support_buffer(
        self, storage: int, *, max_candidates: int | None = None
    ) -> tuple[PackedMatrix, int]:
        """Reserve bounded native output and conservative bulk deduplication workspace."""
        width = self._x.shape[1]
        allowance = self.memory_limit_bytes - self._basis_storage() - storage - 1024
        chunk_size = min(self.batch_size, allowance // max(1, 16 * (width + 1)))
        if max_candidates is not None:
            chunk_size = min(chunk_size, max_candidates)
        if chunk_size < 1:
            raise MemoryError("Support enumeration setup exceeds memory_limit_bytes")
        return np.empty((chunk_size, width), dtype=np.uint64), allowance // max(1, 96 * width)

    def _lift(self, supports_x: PackedMatrix, supports_z: PackedMatrix, minimum: int) -> int:
        width = self._x.shape[1]
        keys_x, keys_z = _support_keys(supports_x), _support_keys(supports_z)
        intersection = np.intersect1d(keys_x, keys_z, assume_unique=True)
        union_size = len(supports_x) + len(supports_z) - len(intersection)
        if union_size * (96 * width) + self._basis_storage() > self.memory_limit_bytes:
            raise MemoryError("Full-Pauli support union exceeds memory_limit_bytes")
        total = 3 * len(intersection)
        # A shared minimum support has only X, Z and Y; an X-only support can be dressed by
        # Z stabilizers contained inside it, and conversely.  This needs one rank query, not four.
        for keys, stabilizers in (
            (np.setdiff1d(keys_x, keys_z, assume_unique=True), self._z),
            (np.setdiff1d(keys_z, keys_x, assume_unique=True), self._x),
        ):
            supports = keys.view(np.uint64).reshape(-1, width)
            for start in range(0, len(supports), self.batch_size):
                words = supports[start : start + self.batch_size]
                histogram = _shortened_histogram(stabilizers, words, minimum)
                total += sum(
                    int(count) << exponent for exponent, count in enumerate(histogram) if count
                )
        return total


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


def _append_supports(buffer: bytearray, output: PackedMatrix, count: int, max_words: int) -> None:
    """Copy matching rows in bulk, never creating one Python object per representative."""
    word_bytes = output.shape[1] * 8
    if len(buffer) + count * word_bytes > max_words * word_bytes:
        raise MemoryError("Minimum-support output exceeds memory_limit_bytes")
    if count:
        buffer.extend(output[:count].tobytes())


def _support_array(buffer: bytearray, width: int) -> PackedMatrix:
    return np.frombuffer(bytes(buffer), dtype=np.uint64).reshape(-1, width)


def _support_keys(words: PackedMatrix) -> npt.NDArray[np.void]:
    """Compare complete packed rows, not individual words or collision-prone hashes."""
    return words.view(np.dtype((np.void, 8 * words.shape[1]))).reshape(-1)


def _pack(matrix: BinaryMatrix) -> PackedMatrix:
    padded = np.pad(matrix, ((0, 0), (0, -matrix.shape[1] % 64)))
    return np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little").view("<u8"))


def _unpack(matrix: PackedMatrix, width: int) -> BinaryMatrix:
    return np.unpackbits(matrix.view(np.uint8), axis=1, bitorder="little")[:, :width].copy()


def _kernel(matrix: BinaryMatrix) -> BinaryMatrix:
    reduced, _, pivots = binary._row_reduce_binary(matrix)
    free = np.setdiff1d(np.arange(matrix.shape[1]), pivots)
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    basis[:, free] = np.eye(len(free), dtype=np.uint8)
    basis[:, pivots] = reduced[: len(pivots), free].T
    return basis


def _iter_krawtchouk(width: int) -> Iterator[tuple[int, ...]]:
    """Stream exact degrees so a low-weight query does not allocate a quadratic full table."""
    previous, current = (0,) * (width + 1), (1,) * (width + 1)
    for weight in range(width + 1):
        yield current
        if weight < width:
            current, previous = (
                tuple(
                    ((width - 2 * jj) * current[jj] - (width - weight + 1) * previous[jj])
                    // (weight + 1)
                    for jj in range(width + 1)
                ),
                current,
            )


@functools.lru_cache(maxsize=32)
def _safe_coefficients(width: int, rank: int) -> npt.NDArray[np.int64]:
    """Certify every intermediate partial sum by 2**rank * binomial(width, weight)."""
    limit = next(
        (
            weight
            for weight in range(width + 1)
            if (1 << rank) * math.comb(width, weight) > np.iinfo(np.int64).max
        ),
        width + 1,
    )
    rows = tuple(itertools.islice(_iter_krawtchouk(width), limit))
    return np.array(rows, dtype=np.int64).reshape(limit, width + 1)


def _dual_coefficient(
    histogram: npt.NDArray[np.int64], coefficients: tuple[int, ...], rank: int
) -> int:
    numerator = sum(int(count) * value for count, value in zip(histogram, coefficients))
    value, remainder = divmod(numerator, 1 << rank)
    if remainder or value < 0:
        raise ArithmeticError("Invalid exact MacWilliams coefficient")
    return value


@njit(cache=True)
def _histogram_chunk(
    rows: PackedMatrix,
    current: npt.NDArray[np.uint64],
    counts: npt.NDArray[np.int64],
    start: int,
    stop: int,
) -> None:
    """Continue a Gray walk using the caller-owned current word and histogram."""
    if rows.shape[1] == 1:
        value = current[0]
        for index in range(start, stop):
            value ^= rows[trailing_zeros(np.uint64(index)), 0]
            counts[popcount_word(value)] += 1
        current[0] = value
        return
    for index in range(start, stop):
        changed = trailing_zeros(np.uint64(index))
        weight = np.int64(0)
        for word in range(rows.shape[1]):
            current[word] ^= rows[changed, word]
            weight += popcount_word(current[word])
        counts[weight] += 1


@njit(cache=True)
def _minimum_native(
    bx: npt.NDArray[np.int64],
    bz: npt.NDArray[np.int64],
    rank_x: int,
    rank_z: int,
    coefficients: npt.NDArray[np.int64],
) -> tuple[int, int, int]:
    """Transform only the previously certified int64-safe coefficient prefix."""
    for weight in range(len(coefficients)):
        nx, nz = np.int64(0), np.int64(0)
        for jj in range(len(bx)):
            nx += bz[jj] * coefficients[weight, jj]
            nz += bx[jj] * coefficients[weight, jj]
        if nx % (1 << rank_z) or nz % (1 << rank_x):
            raise ArithmeticError("Nonintegral MacWilliams coefficient")
        ax, az = nx // (1 << rank_z) - bx[weight], nz // (1 << rank_x) - bz[weight]
        if ax < 0 or az < 0:
            raise ArithmeticError("Negative logical multiplicity")
        if ax or az:
            return weight, ax, az
    return -1, 0, 0


@njit(cache=True)
def _scan_supports(
    rows: PackedMatrix,
    tags: PackedMatrix,
    support: npt.NDArray[np.int64],
    prefixes: PackedMatrix,
    tag_prefixes: PackedMatrix,
    depth: int,
    target: int,
    batch_size: int,
    output: PackedMatrix,
) -> tuple[int, int]:
    """Validate resumable buffers once, then scan a bounded coefficient-layer chunk.

    The checked dimensions and increasing-support invariant make all native indices valid.
    Avoiding repeated bounds checks inside the prefix tree is important on small packed words.
    """
    weight = len(support)
    if (
        not 1 <= weight <= len(rows)
        or not -1 <= depth < weight
        or len(tags) != len(rows)
        or prefixes.shape != (weight + 1, rows.shape[1])
        or tag_prefixes.shape != (weight + 1, tags.shape[1])
        or output.shape[0] < batch_size
        or output.shape[1] != rows.shape[1]
        or batch_size < 1
    ):
        raise IndexError("Incompatible BZ enumeration buffers")
    for index in support:
        if index < 0 or index > len(rows):
            raise IndexError("Invalid resumable coefficient support")
    if rows.shape[1] == 1 and tags.shape[1] == 1:
        return _scan_supports_one(
            rows, tags, support, prefixes, tag_prefixes, depth, target, batch_size, output
        )
    weight, visited, matched = len(support), 0, 0
    while depth >= 0:
        if support[depth] > len(rows) - weight + depth:
            depth -= 1
            if depth >= 0:
                support[depth] += 1
            continue
        if depth + 1 < weight:
            row = support[depth]
            for word in range(rows.shape[1]):
                prefixes[depth + 1, word] = prefixes[depth, word] ^ rows[row, word]
            for word in range(tags.shape[1]):
                tag_prefixes[depth + 1, word] = tag_prefixes[depth, word] ^ tags[row, word]
            depth += 1
            support[depth] = row + 1
            continue
        stop = min(len(rows), support[depth] + batch_size - visited)
        for row in range(support[depth], stop):
            candidate = np.int64(0)
            for word in range(rows.shape[1]):
                candidate += popcount_word(prefixes[depth, word] ^ rows[row, word])
            if candidate == target:
                eligible = False
                for word in range(tags.shape[1]):
                    eligible = eligible or tag_prefixes[depth, word] ^ tags[row, word] != 0
                if eligible:
                    for word in range(rows.shape[1]):
                        output[matched, word] = prefixes[depth, word] ^ rows[row, word]
                    matched += 1
        for word in range(rows.shape[1]):
            prefixes[weight, word] = prefixes[depth, word] ^ rows[stop - 1, word]
        for word in range(tags.shape[1]):
            tag_prefixes[weight, word] = tag_prefixes[depth, word] ^ tags[stop - 1, word]
        visited += stop - support[depth]
        support[depth] = stop
        if visited >= batch_size:
            break
    return depth, matched


@njit(cache=True)
def _scan_supports_one(
    rows: PackedMatrix,
    tags: PackedMatrix,
    support: npt.NDArray[np.int64],
    prefixes: PackedMatrix,
    tag_prefixes: PackedMatrix,
    depth: int,
    target: int,
    batch_size: int,
    output: PackedMatrix,
) -> tuple[int, int]:
    """Scan sibling leaves together, keeping their common prefix in scalar registers."""
    weight, visited, matched = len(support), 0, 0
    while depth >= 0:
        if support[depth] > len(rows) - weight + depth:
            depth -= 1
            if depth >= 0:
                support[depth] += 1
            continue
        if depth + 1 < weight:
            row = support[depth]
            prefixes[depth + 1, 0] = prefixes[depth, 0] ^ rows[row, 0]
            tag_prefixes[depth + 1, 0] = tag_prefixes[depth, 0] ^ tags[row, 0]
            depth += 1
            support[depth] = row + 1
            continue
        stop = min(len(rows), support[depth] + batch_size - visited)
        prefix, tag = prefixes[depth, 0], tag_prefixes[depth, 0]
        for row in range(support[depth], stop):
            value = prefix ^ rows[row, 0]
            if popcount_word(value) == target and tag ^ tags[row, 0] != 0:
                output[matched, 0] = value
                matched += 1
        # Keep the resumable scratch state identical to a leaf-by-leaf traversal.
        prefixes[weight, 0] = prefix ^ rows[stop - 1, 0]
        tag_prefixes[weight, 0] = tag ^ tags[stop - 1, 0]
        visited += stop - support[depth]
        support[depth] = stop
        if visited >= batch_size:
            break
    return depth, matched


@njit(cache=True)
def _exhaustive_supports(
    rows: PackedMatrix,
    current: npt.NDArray[np.uint64],
    stabilizer_rank: int,
    start: int,
    stop: int,
    target: int,
    output: PackedMatrix,
) -> int:
    """Visit a Gray-code chunk, excluding combinations supported only on stabilizer rows."""
    matched = 0
    if rows.shape[1] == 1:
        value = current[0]
        for index in range(start, stop):
            value ^= rows[trailing_zeros(np.uint64(index)), 0]
            if popcount_word(value) == target and (index ^ (index >> 1)) >> stabilizer_rank:
                output[matched, 0] = value
                matched += 1
        current[0] = value
        return matched
    for index in range(start, stop):
        changed = trailing_zeros(np.uint64(index))
        weight = np.int64(0)
        for word in range(rows.shape[1]):
            current[word] ^= rows[changed, word]
            weight += popcount_word(current[word])
        if weight == target and (index ^ (index >> 1)) >> stabilizer_rank:
            output[matched] = current
            matched += 1
    return matched


@njit(cache=True)
def _scan_exhaustive_minimum(
    rows: PackedMatrix,
    current: npt.NDArray[np.uint64],
    stabilizer_rank: int,
    start: int,
    stop: int,
    best: int,
    output: PackedMatrix,
) -> tuple[int, int]:
    """Find the minimum and all ties in one Gray-code pass, excluding the stabilizer subspace."""
    matched = 0
    if rows.shape[1] == 1:
        value = current[0]
        for index in range(start, stop):
            value ^= rows[trailing_zeros(np.uint64(index)), 0]
            weight_one = popcount_word(value)
            if weight_one <= best and (index ^ (index >> 1)) >> stabilizer_rank:
                if weight_one < best:
                    best = int(weight_one)
                    matched = 0
                output[matched, 0] = value
                matched += 1
        current[0] = value
        return best, matched
    for index in range(start, stop):
        changed = trailing_zeros(np.uint64(index))
        weight = np.int64(0)
        for word in range(rows.shape[1]):
            current[word] ^= rows[changed, word]
            weight += popcount_word(current[word])
        if weight <= best and (index ^ (index >> 1)) >> stabilizer_rank:
            if weight < best:
                best = int(weight)
                matched = 0
            output[matched] = current
            matched += 1
    return best, matched


@njit(cache=True)
def _rank_masked(rows: PackedMatrix, mask: npt.NDArray[np.uint64]) -> int:
    """Rank the restricted row space without unpacking its physical coordinates."""
    if rows.shape[1] == 1:
        return _rank_word(rows.reshape(-1), mask[0], np.empty(64, dtype=np.uint64))
    pivots = np.zeros((64 * rows.shape[1], rows.shape[1]), dtype=np.uint64)
    current = np.empty(rows.shape[1], dtype=np.uint64)
    rank = 0
    for row in range(len(rows)):
        for word in range(rows.shape[1]):
            current[word] = rows[row, word] & mask[word]
        while True:
            word = 0
            while word < rows.shape[1] and current[word] == 0:
                word += 1
            if word == rows.shape[1]:
                break
            pivot = 64 * word + trailing_zeros(current[word])
            if pivots[pivot, word] == 0:
                for jj in range(rows.shape[1]):
                    pivots[pivot, jj] = current[jj]
                rank += 1
                break
            for jj in range(rows.shape[1]):
                current[jj] ^= pivots[pivot, jj]
    return rank


@njit(cache=True, inline="always")
def _rank_word(
    rows: npt.NDArray[np.uint64],
    mask: np.uint64,
    pivots: npt.NDArray[np.uint64],
) -> int:
    """Use scalar masks and caller-owned scratch space for repeated single-word rank queries."""
    if not len(rows) or mask == 0:
        return 0
    pivots[:] = 0
    rank = 0
    # An array iterator would retain/release rows on every inlined support query.
    for index in range(len(rows)):
        value = rows[index] & mask
        while value:
            pivot = trailing_zeros(value)
            if pivots[pivot] == 0:
                pivots[pivot] = value
                rank += 1
                break
            value ^= pivots[pivot]
    return rank


@njit(cache=True)
def _shortened_histogram(
    stabilizers: PackedMatrix, supports: PackedMatrix, minimum: int
) -> npt.NDArray[np.int64]:
    """Count opposite-type stabilizers contained in minimum supports.

    A minimum X support cannot contain a nonzero X stabilizer: multiplying by it would give a
    lighter nontrivial logical.  On a support with no minimum Z word, all allowed Z components
    are therefore Z stabilizers.  Their number is 2**(rank(H_Z)-rank(H_Z[:,outside])).
    Group dimensions natively and exponentiate in Python to retain arbitrary-size cardinalities.
    """
    if stabilizers.shape[1] == 1:
        return _shortened_histogram_one(stabilizers.reshape(-1), supports.reshape(-1), minimum)
    histogram = np.zeros(minimum + 1, dtype=np.int64)
    for support in supports:
        dimension = len(stabilizers) - _rank_masked(stabilizers, ~support)
        histogram[dimension] += 1
    return histogram


@njit(cache=True)
def _shortened_histogram_one(
    stabilizers: npt.NDArray[np.uint64],
    supports: npt.NDArray[np.uint64],
    minimum: int,
) -> npt.NDArray[np.int64]:
    """Reuse one pivot buffer; no per-support views, mask vectors, or rank scratch allocations."""
    histogram = np.zeros(minimum + 1, dtype=np.int64)
    pivots = np.empty(64, dtype=np.uint64)
    for support in supports:
        outside = ~support
        dimension = len(stabilizers) - _rank_word(stabilizers, outside, pivots)
        histogram[dimension] += 1
    return histogram


@njit(cache=True)
def _transform(
    rows: PackedMatrix, targets: npt.NDArray[np.int64], matrix: npt.NDArray[np.uint64]
) -> PackedMatrix:
    """Propagate all packed rows through a small local binary matrix on ordered targets."""
    result = rows.copy()
    for index, row in enumerate(rows):
        local = np.uint64(0)
        for jj, target in enumerate(targets):
            local |= ((row[target // 64] >> np.uint64(target % 64)) & np.uint64(1)) << np.uint64(jj)
        for jj, target in enumerate(targets):
            mask = np.uint64(1) << np.uint64(target % 64)
            result[index, target // 64] &= ~mask
            if popcount_word(local & matrix[jj]) & 1:
                result[index, target // 64] |= mask
    return result
