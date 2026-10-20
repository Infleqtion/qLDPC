# SPDX-License-Identifier: Apache-2.0

"""Exact counts of the logical operators of a given weight in binary CSS codes.

For a CSS stabilizer code with checks ``H_X`` and ``H_Z``, the nontrivial X logicals are the words
in ``ker(H_Z)`` outside ``row(H_X)``, and likewise for Z with X and Z exchanged.  At a weight ``w``
that defaults to the distance ``d``, this module counts:

- ``count_x`` and ``count_z``: the X-only and Z-only nontrivial logical operators of weight ``w``,
- optionally ``count_pauli``: all nontrivial logical operators of weight ``w``, including Y and
  mixed operators.  This count is supported only for ``w <= d``, and is zero for ``w < d``.

Counts are of physical operators ignoring phase, not of logical classes.  At the default weight,
``d`` is the minimum over both sectors, so ``count_x`` or ``count_z`` may be zero.  Counting
generally takes an exponentially long time.

Each X/Z sector uses a normalizer basis whose quotient labels mark nontrivial words (Section 3.4 of
``brouwer_zimmermann.md``).  Words are enumerated in layers over Brouwer--Zimmermann (BZ)
information sets.  Unlike a distance search, the scan stops only when the BZ lower bound exceeds
``w`` (Section 5 of ``brouwer_zimmermann.md``), and a word visited by several information sets is
counted by only one of them.  Full Pauli counts are derived from the homogeneous words of weight
``d``, with one masked rank per word.  Self-dual codes scan only one sector, and have
``count_pauli = 3 * count_x``.  Counters for codes conjugated by local CSS-preserving Cliffords
transform the stored sector bases rather than recomputing them.

In a subsystem code, the checks ``G_X`` and ``G_Z`` are gauge generators, and the stabilizers
``S_X`` and ``S_Z`` are the gauge operators that commute with all gauge operators of the other
type.  Nontrivial logicals are those outside the gauge group (dressed logicals), so the X sector is
``ker(S_Z)`` modulo ``row(G_X)``, and likewise for Z.

References:

- https://arxiv.org/abs/1603.06757 (Brouwer--Zimmermann information sets)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from numba import njit

from ._binary import (
    _get_independent_rows,
    _get_kernel,
    _get_nested_code_basis,
    _pack_bits,
    _popcount_word,
    _row_reduce_binary,
)
from .brouwer_zimmermann import (
    _get_brouwer_zimmermann_lower_bound,
    _get_distance_brouwer_zimmermann_many,
    _get_information_set_generators,
    _get_overlapping_information_sets,
    _select_brouwer_zimmermann_information_sets,
)
from .brute_force import _get_distance_quantum_brute_force
from .core import DistanceMethod, validate_distance_method

# codes.common imports this package, so CSSCode is imported only where it is used.
if TYPE_CHECKING:
    from ..common import CSSCode

_BinaryMatrix = npt.NDArray[np.uint8]
_PackedMatrix = npt.NDArray[np.uint64]

# Coefficient combinations visited per native call, which bounds the time between interruptions.
_CHUNK_SIZE = 1 << 16

# Batch size exponent passed to the distance searches, as in get_distance_quantum.
_BLOCK_SIZE = 15

# The basis of a homogeneous sector, and its quotient labels.
_Sector = tuple[_BinaryMatrix, _BinaryMatrix]


####################################################################################################
# results and entry points


@dataclass(frozen=True)
class CSSLogicalCounts:
    """Counts of the nontrivial logical operators of one weight in a binary CSS code."""

    weight: int | None
    """Weight of the counted operators.

    It is the distance of the code if no weight was requested, and None if no weight was requested
    and the code encodes no qubits.
    """

    count_x: int
    """Number of X-only nontrivial logical operators of weight ``weight``."""

    count_z: int
    """Number of Z-only nontrivial logical operators of weight ``weight``."""

    count_pauli: int | None
    """Number of nontrivial logical Pauli operators of weight ``weight``, ignoring phase.

    It includes Y and mixed operators, and is None if it was not requested.
    """


class CSSLogicalCounter:
    """Exact counter of the logical operators of a given weight in a binary CSS code.

    See the ``logical_counts`` module docstring for the conventions and algorithms.  Counters for
    codes conjugated by local CSS-preserving Cliffords reuse the work of this counter.

    Args:
        code: A binary CSS code, which may be a subsystem code and may have redundant checks.

    Raises:
        TypeError: If the code is not a :class:`~qldpc.codes.CSSCode`.
        ValueError: If the code is not binary.
    """

    def __init__(self, code: CSSCode) -> None:
        from ..common import CSSCode

        if not isinstance(code, CSSCode):
            raise TypeError(f"Expected a CSSCode, not {type(code).__name__}")
        if code.field.order != 2:
            raise ValueError(f"Logical counts require a binary code, not GF({code.field.order})")
        self._code: CSSCode | None = code
        checks_x = code.matrix_x.view(np.ndarray).astype(np.uint8)
        checks_z = code.matrix_z.view(np.ndarray).astype(np.uint8)
        self.num_qubits = checks_x.shape[1]
        self._gauge_x = _get_independent_rows(checks_x)
        self._gauge_z = _get_independent_rows(checks_z)
        # Reduced row echelon form is unique, so the X and Z checks span the same space exactly when
        # these arrays are equal.  A self-dual code then shares one array for each X/Z pair.
        self._self_dual = np.array_equal(self._gauge_x, self._gauge_z)
        if self._self_dual:
            self._gauge_z = self._gauge_x
        self._stabilizers_x, self._stabilizers_z = _get_stabilizers(self._gauge_x, self._gauge_z)
        if self._self_dual:
            self._stabilizers_z = self._stabilizers_x
        self.dimension = self.num_qubits - len(self._gauge_x) - len(self._stabilizers_z)
        self._sectors: tuple[_Sector, _Sector] | None = None
        self._canonical: tuple[_BinaryMatrix, _BinaryMatrix] | None = (self._gauge_x, self._gauge_z)

    @property
    def code(self) -> CSSCode:
        """The code whose logical operators are counted.

        For a counter built by a conjugation method, this is a new :class:`~qldpc.codes.CSSCode`
        with the counter's independent checks, built on first access.
        """
        if self._code is None:
            from ..common import CSSCode

            self._code = CSSCode(self._gauge_x.astype(np.int_), self._gauge_z.astype(np.int_))
        return self._code

    @property
    def checks_x(self) -> npt.NDArray[np.uint8]:
        """A copy of an independent basis of the X checks, in reduced row echelon form."""
        return self._get_canonical_checks()[0].copy()

    @property
    def checks_z(self) -> npt.NDArray[np.uint8]:
        """A copy of an independent basis of the Z checks, in reduced row echelon form."""
        return self._get_canonical_checks()[1].copy()

    @property
    def is_self_dual(self) -> bool:
        """Whether the X and Z checks span the same row space."""
        canonical_x, canonical_z = self._get_canonical_checks()
        return self._self_dual or np.array_equal(canonical_x, canonical_z)

    def count_logicals(
        self,
        weight: int | None = None,
        *,
        method: DistanceMethod = "brouwer_zimmermann",
        full_pauli: bool = False,
    ) -> CSSLogicalCounts:
        """Count the nontrivial logical operators of a given weight.

        Args:
            weight: Weight of the operators to count, between zero and the number of qubits.  If
                None (the default), count the operators whose weight is the distance of the code.
            method: ``"brouwer_zimmermann"`` (the default) or ``"brute_force"``.  Brute force
                enumerates all ``2**rank`` combinations of each normalizer basis, and is a reference
                for small codes.
            full_pauli: If True, also count full Pauli operators, including Y and mixed operators.
                Each homogeneous word of the counted weight costs one rank calculation.  These
                counts are exact up to the distance and zero below it, and are not supported above
                it.  Defaults to False.

        Returns:
            The counted weight and the counts.  If no weight is given and the code encodes no
            qubits, the returned weight is None and every count is zero.

        Raises:
            ValueError: If the weight is not an integer between zero and the number of qubits, if
                the method is unknown, or if full Pauli operators are requested at a weight above
                the distance.
        """
        validate_distance_method(method)
        if weight is not None and (
            not isinstance(weight, (int, np.integer))
            or isinstance(weight, bool)
            or not 0 <= weight <= self.num_qubits
        ):
            raise ValueError("The weight must be an integer between zero and the number of qubits")
        if not self.dimension:
            empty_pauli = 0 if full_pauli else None
            return CSSLogicalCounts(None if weight is None else int(weight), 0, 0, empty_pauli)

        sector_x, sector_z = self._get_sectors()
        # The two sectors of a self-dual code are the same, so one of them suffices.
        sectors = [sector_x] if self._self_dual else [sector_x, sector_z]
        target = int(weight) if weight is not None else _get_distance(sectors, method)

        # A nontrivial logical Pauli (x, z) of weight d has one of two disjoint forms.  Either x is
        # a nontrivial X logical with support T of size d, and z is one of 2**(d - rank(S_X[:, T]))
        # Z words supported in T that commute with S_X, or x is an X gauge operator, and z is a
        # nontrivial Z logical of weight d whose support T contains that of x, which leaves
        # 2**(rank(G_X) - rank(G_X[:, ~T])) choices of x.  Each support T of a self-dual code
        # carries exactly three operators, so self-dual codes need no ranks.  Above d, a logical
        # Pauli need not have the support of a homogeneous logical, so the scans reject any
        # nontrivial word lighter than the target.
        weigh = full_pauli and not self._self_dual
        no_rows = self._gauge_x[:0]
        counts_x = _count_words(
            *sector_x,
            target,
            method,
            _pack_bits(self._stabilizers_x if weigh else no_rows),
            complement=False,
            max_exponent=target if weigh else 0,
            reject_lighter=full_pauli,
        )
        if self._self_dual or counts_x is None:
            counts_z = counts_x
        else:
            counts_z = _count_words(
                *sector_z,
                target,
                method,
                _pack_bits(self._gauge_x if weigh else no_rows),
                complement=True,
                max_exponent=len(self._gauge_x) if weigh else 0,
                reject_lighter=full_pauli,
            )
        if counts_x is None or counts_z is None:
            raise ValueError(
                "Full Pauli operators can be counted only up to the distance, which is less than"
                f" weight {target}"
            )
        count_x, count_z = int(counts_x.sum()), int(counts_z.sum())
        count_pauli = None
        if full_pauli and self._self_dual:
            count_pauli = 3 * count_x
        elif full_pauli:
            count_pauli = sum(
                int(count) << exponent
                for counts in (counts_x, counts_z)
                for exponent, count in enumerate(counts)
            )
        return CSSLogicalCounts(target, count_x, count_z, count_pauli)

    def conjugated(
        self, targets: Sequence[int], matrix_x: npt.ArrayLike, matrix_z: npt.ArrayLike
    ) -> CSSLogicalCounter:
        """Return a counter for this code conjugated by a local CSS-preserving Clifford.

        Args:
            targets: Between one and 64 distinct qubit indices on which the Clifford acts, in order.
            matrix_x: Binary matrix that maps the X support on the targets, as a column vector, to
                its image.
            matrix_z: The same for Z supports.  It must satisfy ``matrix_x @ matrix_z.T = I``
                over GF(2), so that commutation is preserved.

        Returns:
            A new counter.  This counter is not modified.

        Raises:
            ValueError: If the targets or matrices are invalid.
        """
        positions = self._get_positions(targets)
        local_x, local_z = _pack_local_matrices(matrix_x, matrix_z, len(positions))
        return self._conjugated(positions, local_x, local_z)

    def conjugated_cnot(self, control: int, target: int) -> CSSLogicalCounter:
        """Return a counter for this code conjugated by a CNOT on zero-based qubit indices."""
        return self._conjugated(self._get_positions([control, target]), *_CNOT)

    def conjugated_swel(
        self, qubit_0: int, qubit_1: int, qubit_2: int, qubit_3: int
    ) -> CSSLogicalCounter:
        """Return a counter for this code conjugated by a four-qubit SWEL-preserving Clifford.

        The Clifford applies the same binary matrix, whose rows have weight three, to X and Z
        supports on the four targets.  It therefore maps self-dual codes to self-dual codes, and
        preserves the self-dual-with-equivalent-logicals (SWEL) property of
        :meth:`qldpc.codes.CSSCode.is_swel`.  It is valid for any CSS code.  The four qubits are
        distinct zero-based indices, in the order of the matrix rows.

        Returns:
            A new counter.  This counter is not modified.

        Raises:
            ValueError: If the qubits are not distinct, in-range indices.
        """
        positions = self._get_positions([qubit_0, qubit_1, qubit_2, qubit_3])
        return self._conjugated(positions, *_SWEL)

    def _get_positions(self, targets: Sequence[int]) -> npt.NDArray[np.int64]:
        """Validate conjugation targets."""
        positions = list(targets)
        if (
            not 1 <= len(positions) <= 64
            or any(
                not isinstance(index, (int, np.integer))
                or isinstance(index, bool)
                or not 0 <= index < self.num_qubits
                for index in positions
            )
            or len(set(positions)) != len(positions)
        ):
            raise ValueError("Clifford targets must be 1 to 64 distinct, in-range qubit indices")
        return np.array(positions, dtype=np.int64)

    def _conjugated(
        self,
        positions: npt.NDArray[np.int64],
        local_x: npt.NDArray[np.uint64],
        local_z: npt.NDArray[np.uint64],
    ) -> CSSLogicalCounter:
        """Apply validated, packed local matrices to every stored X-type and Z-type matrix.

        The Clifford maps each gauge, stabilizer and sector space onto that of the new code, and
        keeps the quotient labels, so nothing is recomputed.  The checks are no longer in reduced
        row echelon form, which is recomputed only if needed.
        """
        counter = CSSLogicalCounter.__new__(CSSLogicalCounter)
        counter.num_qubits, counter.dimension = self.num_qubits, self.dimension
        counter._code, counter._canonical = None, None
        # A gate that acts identically on X and Z keeps a self-dual code self-dual.
        shared = self._self_dual and (local_x is local_z or np.array_equal(local_x, local_z))
        counter._self_dual = shared
        counter._gauge_x = _apply_local(self._gauge_x, positions, local_x)
        counter._gauge_z = (
            counter._gauge_x if shared else _apply_local(self._gauge_z, positions, local_z)
        )
        # Stabilizer codes share their gauge and stabilizer arrays.
        counter._stabilizers_x = (
            counter._gauge_x
            if self._stabilizers_x is self._gauge_x
            else _apply_local(self._stabilizers_x, positions, local_x)
        )
        if self._stabilizers_z is self._gauge_z:
            counter._stabilizers_z = counter._gauge_z
        elif shared:
            counter._stabilizers_z = counter._stabilizers_x
        else:
            counter._stabilizers_z = _apply_local(self._stabilizers_z, positions, local_z)
        counter._sectors = None
        if self._sectors is not None:
            (basis_x, labels_x), (basis_z, labels_z) = self._sectors
            sector_x = (_apply_local(basis_x, positions, local_x), labels_x)
            sector_z = sector_x if shared else (_apply_local(basis_z, positions, local_z), labels_z)
            counter._sectors = (sector_x, sector_z)
        return counter

    def _get_sectors(self) -> tuple[_Sector, _Sector]:
        """The X and Z sectors: normalizer bases whose first rows span the gauge operators."""
        if self._sectors is None:
            sector_x = _get_nested_code_basis(_get_kernel(self._stabilizers_z), self._gauge_x)
            sector_z = (
                sector_x
                if self._self_dual
                else _get_nested_code_basis(_get_kernel(self._stabilizers_x), self._gauge_z)
            )
            self._sectors = (sector_x, sector_z)
        return self._sectors

    def _get_canonical_checks(self) -> tuple[_BinaryMatrix, _BinaryMatrix]:
        """The checks in reduced row echelon form, which conjugation does not preserve."""
        if self._canonical is None:
            self._canonical = (
                _get_independent_rows(self._gauge_x),
                _get_independent_rows(self._gauge_z),
            )
        return self._canonical


def count_logicals_css(
    code: CSSCode,
    weight: int | None = None,
    *,
    method: DistanceMethod = "brouwer_zimmermann",
    full_pauli: bool = False,
) -> CSSLogicalCounts:
    """Count the nontrivial logical operators of a given weight in a binary CSS code.

    This is shorthand for ``CSSLogicalCounter(code).count_logicals(...)``; see
    :meth:`CSSLogicalCounter.count_logicals` for the arguments and conventions.
    """
    return CSSLogicalCounter(code).count_logicals(weight, method=method, full_pauli=full_pauli)


####################################################################################################
# counting words of one weight


def _get_distance(sectors: Sequence[_Sector], method: DistanceMethod) -> int:
    """Minimum weight of a nontrivial word in any of the given sectors."""
    if method == "brute_force":
        return min(_get_brute_force_distance(sector) for sector in sectors)
    return _get_distance_brouwer_zimmermann_many(sectors, cutoff=0, block_size=_BLOCK_SIZE)


def _get_brute_force_distance(sector: _Sector) -> int:
    """Minimum weight of a nontrivial word of one sector, by exhaustive enumeration."""
    basis, labels = sector
    eligible = np.any(labels, axis=1)
    return _get_distance_quantum_brute_force(
        basis[eligible], basis[~eligible], cutoff=0, block_size=_BLOCK_SIZE, homogeneous=True
    )


def _count_words(
    basis: _BinaryMatrix,
    labels: _BinaryMatrix,
    weight: int,
    method: DistanceMethod,
    rank_rows: _PackedMatrix,
    *,
    complement: bool,
    max_exponent: int,
    reject_lighter: bool,
) -> npt.NDArray[np.int64] | None:
    """Visit each nontrivial word of one sector with the given weight exactly once.

    Returns a histogram over exponents ``max_exponent - rank(rank_rows[:, mask])``, where the mask
    is the support of the word, or its complement.  The histogram sums to the number of words, and
    the caller weighs each exponent by ``2**exponent``, which could overflow a native integer.  The
    scan also visits every lighter word, so with ``reject_lighter`` it returns None as soon as it
    finds a nontrivial one.
    """
    get_lower_bound: Callable[[int], int]
    if method == "brute_force":
        sets, get_lower_bound = [(basis, labels)], lambda layer: 0
    else:
        sets, get_lower_bound = _get_counting_information_sets(basis, labels, weight)

    rows = [_pack_bits(generators) for generators, _ in sets]
    tags = [_pack_bits(set_labels) for _, set_labels in sets]
    # Ownership compares sets, so a single set needs no decoder.
    decoders = (
        np.stack([_get_decoder(generators) for generators, _ in sets])
        if len(sets) > 1
        else np.zeros((1, 0, rows[0].shape[1]), dtype=np.uint64)
    )
    histogram = np.zeros(max_exponent + 1, dtype=np.int64)
    for layer in range(1, len(basis) + 1):
        for index in range(len(sets)):
            support = np.zeros(layer, dtype=np.int64)
            depth = 0
            while depth >= 0:
                depth, lighter = _count_chunk(
                    rows[index],
                    tags[index],
                    decoders,
                    index,
                    support,
                    depth,
                    weight,
                    rank_rows,
                    complement,
                    max_exponent,
                    histogram,
                    reject_lighter,
                    _CHUNK_SIZE,
                )
                if lighter:
                    return None
        # Equality would still allow unvisited words of the target weight.
        if get_lower_bound(layer) > weight:
            break
    return histogram


def _get_counting_information_sets(
    basis: _BinaryMatrix, labels: _BinaryMatrix, weight: int
) -> tuple[list[_Sector], Callable[[int], int]]:
    """Select BZ information sets that certify every word of the given weight.

    Returns the sets and the lower bound on the weight of unvisited words after each layer.
    """
    # XOR preserves even parity, so an even basis spans only even weights.
    divisor = 2 if np.all(np.count_nonzero(basis, axis=1) % 2 == 0) else 1
    candidates = _get_information_set_generators(basis, labels)
    # Aim the family at the counting stop L(t) >= weight + 1; any family gives an exact count.
    candidates = _select_brouwer_zimmermann_information_sets(
        candidates, dimension=len(basis), upper_bound=weight + 1, weight_divisor=divisor
    )
    max_coverage = 1
    overlapping = _get_overlapping_information_sets(
        basis, labels, candidates, upper_bound=weight + 1, weight_divisor=divisor
    )
    if overlapping is not None:
        candidates, max_coverage = overlapping
    ranks = [candidate.rank for candidate in candidates]

    def get_lower_bound(layer: int) -> int:
        return _get_brouwer_zimmermann_lower_bound(
            len(basis),
            ranks,
            completed_weight=layer,
            weight_divisor=divisor,
            max_coverage=max_coverage,
        )

    sets = []
    for candidate in candidates:
        assert candidate.labels is not None
        sets.append((candidate.generators, candidate.labels))
    return sets, get_lower_bound


def _get_decoder(generators: _BinaryMatrix) -> _PackedMatrix:
    """Packed rows ``E.T`` such that ``generators @ E`` is the identity over GF(2).

    Bit ``b`` of the coefficient vector of a word ``w`` is then the parity of ``w & E.T[b]``.
    """
    # Row reduction finds T with T @ generators = R, where R is the identity on its pivot columns P,
    # so that w[P] @ T recovers the coefficients of w.
    size = len(generators)
    _, transform, pivots = _row_reduce_binary(generators, np.eye(size, dtype=np.uint8))
    assert transform is not None
    decoder = np.zeros((generators.shape[1], size), dtype=np.uint8)
    decoder[pivots] = transform
    return _pack_bits(decoder.T)


####################################################################################################
# stabilizers of subsystem codes


def _get_stabilizers(
    gauge_x: _BinaryMatrix, gauge_z: _BinaryMatrix
) -> tuple[_BinaryMatrix, _BinaryMatrix]:
    """Independent bases of the stabilizers among the given gauge generators.

    The stabilizers are the gauge operators that commute with every gauge operator of the other
    type.  If all gauge generators commute, the given arrays are returned.
    """
    overlaps = (gauge_x.astype(int) @ gauge_z.T.astype(int) % 2).astype(np.uint8)
    if not np.any(overlaps):
        return gauge_x, gauge_z
    # A combination c @ gauge_x commutes with every Z gauge generator exactly when c @ overlaps = 0.
    stabilizers_x = _get_kernel(overlaps.T).astype(int) @ gauge_x % 2
    stabilizers_z = _get_kernel(overlaps).astype(int) @ gauge_z % 2
    return (
        _get_independent_rows(stabilizers_x.astype(np.uint8)),
        _get_independent_rows(stabilizers_z.astype(np.uint8)),
    )


####################################################################################################
# local Cliffords

# Packed local matrices, in which bit j of row i is entry (i, j).  The CNOT, with its control first,
# acts by [[1, 0], [1, 1]] on X supports and by [[1, 1], [0, 1]] on Z supports.  The SWEL-preserving
# Clifford acts by [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]] on both.
_CNOT = (np.array([1, 3], dtype=np.uint64), np.array([3, 2], dtype=np.uint64))
_SWEL = (np.array([7, 11, 13, 14], dtype=np.uint64),) * 2


def _pack_local_matrices(
    matrix_x: npt.ArrayLike, matrix_z: npt.ArrayLike, size: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Validate local CSS-preserving matrices, and pack each row into a word (bit j is column j)."""
    packed = []
    for matrix in (matrix_x, matrix_z):
        order = getattr(type(matrix), "order", None)
        local = np.asarray(matrix)
        if local.shape != (size, size):
            raise ValueError("Local Clifford matrices must be square and match the target count")
        if (order is not None and order != 2) or not np.all((local == 0) | (local == 1)):
            raise ValueError("Local Clifford matrices must be binary")
        # At most 64 targets, so each row packs into a single word.
        packed.append(_pack_bits(local.astype(np.uint8))[:, 0])
    if not _is_css_pairing(packed[0], packed[1]):
        raise ValueError("Local X/Z matrices must preserve the CSS commutation pairing")
    return packed[0], packed[1]


####################################################################################################
# native kernels


@njit(cache=True)
def _is_css_pairing(local_x: npt.NDArray[np.uint64], local_z: npt.NDArray[np.uint64]) -> bool:
    """Whether packed local matrices satisfy ``local_x @ local_z.T = I`` over GF(2)."""
    for row in range(len(local_x)):
        for column in range(len(local_z)):
            if _popcount_word(local_x[row] & local_z[column]) & 1 != (row == column):
                return False
    return True


@njit(cache=True)
def _apply_local(
    rows: _BinaryMatrix, positions: npt.NDArray[np.int64], local: npt.NDArray[np.uint64]
) -> _BinaryMatrix:
    """Apply a packed local matrix to the target columns of a copy of the given rows.

    Column ``positions[i]`` of a row becomes the parity of ``local[i]`` and the word whose bit ``j``
    is the row's entry at ``positions[j]``.
    """
    result = rows.copy()
    for row in range(rows.shape[0]):
        word = np.uint64(0)
        for index in range(len(positions)):
            word |= np.uint64(rows[row, positions[index]]) << np.uint64(index)
        if word:
            for index in range(len(positions)):
                result[row, positions[index]] = _popcount_word(local[index] & word) & 1
    return result


@njit(cache=True)
def _count_chunk(
    rows: _PackedMatrix,
    tags: _PackedMatrix,
    decoders: npt.NDArray[np.uint64],
    set_index: int,
    support: npt.NDArray[np.int64],
    depth: int,
    target: int,
    rank_rows: _PackedMatrix,
    complement: bool,
    max_exponent: int,
    histogram: npt.NDArray[np.int64],
    reject_lighter: bool,
    chunk_size: int,
) -> tuple[int, bool]:
    """Visit up to ``chunk_size`` combinations of ``len(support)`` rows, resuming at ``depth``.

    ``support`` holds the increasing row indices of the current combination; ``depth`` is the
    position being advanced, or -1 once every combination has been visited.  Each owned nontrivial
    word of weight ``target`` adds one to ``histogram`` (see ``_count_words``).  Returns the depth
    from which to resume, and whether the scan stopped at a nontrivial word lighter than ``target``,
    which is checked only with ``reject_lighter``.
    """
    num_rows, width = rows.shape
    layer = len(support)
    # Prefixes are recomputed on entry, so only support and depth carry state between calls.
    prefixes = np.zeros((layer, width), dtype=np.uint64)
    tag_prefixes = np.zeros((layer, tags.shape[1]), dtype=np.uint64)
    for position in range(1, depth + 1):
        row = support[position - 1]
        for index in range(width):
            prefixes[position, index] = prefixes[position - 1, index] ^ rows[row, index]
        for index in range(tags.shape[1]):
            tag_prefixes[position, index] = tag_prefixes[position - 1, index] ^ tags[row, index]
    word = np.empty(width, dtype=np.uint64)
    pivots = np.empty((len(rank_rows), width), dtype=np.uint64)
    visited = 0
    while depth >= 0 and visited < chunk_size:
        # Backtrack when no room remains for the rows after this position.
        if support[depth] > num_rows - layer + depth:
            depth -= 1
            if depth >= 0:
                support[depth] += 1
            continue
        if depth + 1 < layer:
            row = support[depth]
            for index in range(width):
                prefixes[depth + 1, index] = prefixes[depth, index] ^ rows[row, index]
            for index in range(tags.shape[1]):
                tag_prefixes[depth + 1, index] = tag_prefixes[depth, index] ^ tags[row, index]
            depth += 1
            support[depth] = row + 1
            continue

        # Visit the remaining combinations that differ only in their last row.
        stop = min(num_rows, support[depth] + chunk_size - visited)
        for row in range(support[depth], stop):
            weight = np.int64(0)
            for index in range(width):
                word[index] = prefixes[depth, index] ^ rows[row, index]
                weight += _popcount_word(word[index])
            if weight > target or (weight < target and not reject_lighter):
                continue
            # A nonzero quotient label marks a nontrivial logical rather than a stabilizer.
            nontrivial = False
            for index in range(tags.shape[1]):
                if tag_prefixes[depth, index] != tags[row, index]:
                    nontrivial = True
                    break
            if nontrivial and weight < target:
                return depth, True
            if nontrivial and _is_owned(word, decoders, set_index, layer):
                rank = _get_masked_rank(rank_rows, word, complement, pivots)
                histogram[max_exponent - rank] += 1
        visited += stop - support[depth]
        support[depth] = stop
    return depth, False


@njit(cache=True)
def _is_owned(
    word: npt.NDArray[np.uint64], decoders: npt.NDArray[np.uint64], set_index: int, layer: int
) -> bool:
    """Whether set ``set_index`` minimizes (layer, set index) among the sets that visit ``word``.

    Each set visits a word in the layer given by the weight of its coefficient vector in that set,
    so exactly one set owns each word, and it visits the word before any other set does.
    """
    for other in range(len(decoders)):
        if other == set_index:
            continue
        # A set that visits the word in an earlier layer, or in the same layer with a smaller index,
        # owns it.
        limit = layer if other < set_index else layer - 1
        size = 0
        for decoder in decoders[other]:
            parity = np.int64(0)
            for index in range(len(word)):
                parity += _popcount_word(word[index] & decoder[index])
            size += int(parity & 1)
            if size > limit:
                break
        if size <= limit:
            return False
    return True


@njit(cache=True)
def _get_masked_rank(
    rows: _PackedMatrix,
    mask: npt.NDArray[np.uint64],
    complement: bool,
    pivots: _PackedMatrix,
) -> int:
    """Rank of ``rows`` restricted to the support of ``mask``, or to its complement.

    ``pivots`` is scratch space with at least as many rows as ``rows``.
    """
    pivot_words = np.empty(len(rows), dtype=np.int64)
    pivot_bits = np.empty(len(rows), dtype=np.uint64)
    rank = 0
    for row in rows:
        current = row & ~mask if complement else row & mask
        # Pivot rows are reduced when stored, so eliminating them in order clears every pivot bit.
        for previous in range(rank):
            if current[pivot_words[previous]] & pivot_bits[previous]:
                current ^= pivots[previous]
        for index in range(len(current)):
            if current[index]:
                pivots[rank] = current
                pivot_words[rank] = index
                pivot_bits[rank] = current[index] & (~current[index] + np.uint64(1))
                rank += 1
                break
    return rank
