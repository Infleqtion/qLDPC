# SPDX-License-Identifier: Apache-2.0

"""Exact counts of the logical operators of a given weight in binary CSS stabilizer codes.

For a binary CSS stabilizer code with X checks ``H_X`` and Z checks ``H_Z``, this module counts, at
a weight ``w`` that defaults to the distance ``d``:

- ``count_x``: X-only words of weight ``w`` in ``ker(H_Z)`` that are not X stabilizers,
- ``count_z``: the same for Z, and
- optionally ``count_pauli``: physical Pauli operators of weight ``w`` that commute with every
  stabilizer but are not stabilizers, including Y and mixed operators.  This count is supported only
  for ``w <= d``, and it is zero for ``w < d``.

Counts are of physical operators ignoring phase, not of logical classes modulo stabilizers.  At the
default weight, ``count_x`` and ``count_z`` refer to the global minimum ``d``, so one of them may be
zero.  Subsystem, non-CSS and nonbinary codes are not supported, and counting is exponential in the
worst case.  Memory use is polynomial: no method stores one entry per logical word.

Each homogeneous (X-only or Z-only) sector is a basis of its normalizer whose first rows span the
stabilizers, with quotient labels that are nonzero exactly for nontrivial logical words (see Section
3.4 of ``brouwer_zimmermann.md``).  Every word of weight ``w`` in each sector is visited and counted
as it is found, by scanning coefficient combinations of increasing size ("layers") over one or more
information sets.  When no weight is given, the distance is first computed by the existing
Brouwer--Zimmermann (BZ) or brute-force distance search.

Counting with BZ needs a strict certificate.  BZ guarantees that every word not yet visited after
layer ``t`` has weight at least ``L(t)``, where ``L`` is computed from the ranks and the maximum
coordinate coverage of the information sets that are actually scanned.  A distance search may stop
at ``L(t) >= d``, but counting must visit every word of weight exactly ``w``, so it stops only at
``L(t) > w``.  That makes the count exact for any family of information sets; the family is chosen
to reach ``L(t) >= w + 1`` cheaply, which affects only the work.  The ``"brute_force"`` method
instead scans every layer of a single basis, which visits each word exactly once.

A word may be visited by several information sets.  Each set ``j`` is a full basis ``R_j`` of the
sector, so a word ``v`` has a unique coefficient vector ``c_j(v)`` with ``c_j(v) @ R_j = v``, and
set ``j`` visits ``v`` in layer ``|c_j(v)|``.  A word found by set ``i`` in layer ``t`` is counted
only if no set ``j`` has ``(|c_j(v)|, j) < (t, i)``.  The set that minimizes this pair visits ``v``
no later than set ``i`` does, so every word is counted exactly once without being stored.

Full Pauli operators of weight ``d`` are counted from the same visits.  Let ``(x, z)`` be a
nontrivial logical operator of weight ``d`` with X part ``x`` and Z part ``z``.  If ``x`` is not an
X stabilizer, it is a nontrivial X logical of weight at most ``d``, so its support ``T`` has weight
``d`` and contains ``z``.  Conversely, any ``z`` supported in ``T`` with ``H_X z = 0`` gives such an
operator, so ``T`` contributes ``2**(d - rank(H_X[:, T]))`` operators.  If ``x`` is an X stabilizer,
then ``z`` is a nontrivial Z logical with support ``T`` of weight ``d`` that contains ``x``, and
``T`` contributes ``2**(rank(H_X) - rank(H_X[:, ~T]))`` operators, one per X stabilizer supported in
``T``.  The two cases are disjoint, so ``count_pauli`` sums the first term over X-only words of
weight ``d`` and the second over Z-only words of weight ``d``, with one masked rank per word.

This argument fails above the distance, where the X part of a nontrivial operator can be a lighter
nontrivial logical whose support does not contain the Z part.  The operator's support then need not
be the support of any homogeneous logical word of the same weight.  For example, ``XYZI`` is a
nontrivial logical operator of weight three in the [[4, 2, 2]] code, which has no homogeneous
logical words of odd weight.  Given a weight ``w``, full Pauli counting does not search for the
distance first.  The scan at ``w`` stops only when every unvisited word is heavier than ``w``, so it
visits every lighter word, and it rejects ``w`` as soon as it finds a lighter nontrivial one.  If it
finds none, then ``w <= d``, and the counts are exact.

References:

- https://arxiv.org/abs/1603.06757 (Brouwer--Zimmermann information sets)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import numpy.typing as npt
from numba import njit

from ._binary import (
    _as_binary_matrix,
    _assert_binary,
    _get_independent_rows,
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

if TYPE_CHECKING:
    from ..common import CSSCode

LogicalCountMethod = Literal["brouwer_zimmermann", "brute_force"]

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
    """Exact counter of the logical operators of a given weight in a binary CSS stabilizer code.

    See the ``logical_counts`` module docstring for the conventions and algorithms.  Use
    :meth:`from_matrices` to start from parity check matrices without building a
    :class:`~qldpc.codes.CSSCode`.

    Args:
        code: A binary CSS stabilizer code.  Its parity checks may be redundant.

    Raises:
        TypeError: If the code is not a :class:`~qldpc.codes.CSSCode`.
        ValueError: If the code is not binary, or if its X and Z checks do not commute.  Subsystem
            codes, whose gauge generators need not commute, are not supported.
    """

    def __init__(self, code: CSSCode) -> None:
        from ..common import CSSCode

        if not isinstance(code, CSSCode):
            raise TypeError(f"Expected a CSSCode, not {type(code).__name__}")
        self._initialize(code.matrix_x, code.matrix_z)
        self._code: CSSCode | None = code

    @staticmethod
    def from_matrices(matrix_x: npt.ArrayLike, matrix_z: npt.ArrayLike) -> CSSLogicalCounter:
        """Build a counter from the parity check matrices of a binary CSS stabilizer code.

        Args:
            matrix_x: Binary X checks.  Redundant rows are allowed.
            matrix_z: Binary Z checks.  Redundant rows are allowed.

        Returns:
            A counter whose :attr:`code` is built only when it is first accessed.

        Raises:
            ValueError: If the checks are not binary, have different widths, or do not commute.
        """
        counter = object.__new__(CSSLogicalCounter)
        counter._initialize(matrix_x, matrix_z)
        counter._code = None
        return counter

    def _initialize(self, matrix_x: npt.ArrayLike, matrix_z: npt.ArrayLike) -> None:
        """Validate the checks and store an independent basis of each."""
        _assert_binary(matrix_x, "matrix_x")
        _assert_binary(matrix_z, "matrix_z")
        checks_x = _as_binary_matrix(matrix_x)
        checks_z = _as_binary_matrix(matrix_z, num_cols=checks_x.shape[1])
        if np.any((checks_x.astype(int) @ checks_z.T.astype(int)) % 2):
            raise ValueError("CSS stabilizer checks must commute; subsystem codes are unsupported")
        # Reduced row echelon form is unique, so equal row spaces give equal arrays.
        self._x = _get_independent_rows(checks_x)
        self._z = _get_independent_rows(checks_z)
        self.num_qubits = checks_x.shape[1]
        self.dimension = self.num_qubits - len(self._x) - len(self._z)

    @property
    def code(self) -> CSSCode:
        """The code whose logical operators are counted.

        For a counter built with :meth:`from_matrices` or a conjugation method, this is a new
        :class:`~qldpc.codes.CSSCode` with the counter's independent checks, built on first access.
        """
        if self._code is None:
            from ..common import CSSCode

            self._code = CSSCode(self._x.astype(np.int_), self._z.astype(np.int_))
        return self._code

    @property
    def checks_x(self) -> npt.NDArray[np.uint8]:
        """A copy of an independent basis of the X checks, in reduced row echelon form."""
        return self._x.copy()

    @property
    def checks_z(self) -> npt.NDArray[np.uint8]:
        """A copy of an independent basis of the Z checks, in reduced row echelon form."""
        return self._z.copy()

    @property
    def is_self_dual(self) -> bool:
        """Whether the X and Z checks span the same row space."""
        return np.array_equal(self._x, self._z)

    def count_logicals(
        self,
        weight: int | None = None,
        *,
        method: LogicalCountMethod = "brouwer_zimmermann",
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
        if method not in get_args(LogicalCountMethod):
            raise ValueError(f"Unknown logical-count method: {method!r}")
        if weight is not None and (
            not isinstance(weight, (int, np.integer))
            or isinstance(weight, bool)
            or not 0 <= weight <= self.num_qubits
        ):
            raise ValueError("The weight must be an integer between zero and the number of qubits")
        if not self.dimension:
            empty_pauli = 0 if full_pauli else None
            return CSSLogicalCounts(None if weight is None else int(weight), 0, 0, empty_pauli)

        sector_x = _get_nested_code_basis(_get_kernel(self._z), self._x)
        sector_z = _get_nested_code_basis(_get_kernel(self._x), self._z)
        target = int(weight) if weight is not None else _get_distance(sector_x, sector_z, method)

        # Full Pauli counts weigh each support against the X checks, as in the module docstring,
        # and the scans reject any nontrivial word lighter than the target, which would mean that
        # the target exceeds the distance.  Otherwise the X checks are omitted and every word adds
        # one to the zeroth exponent.
        lift = _pack_bits(self._x if full_pauli else self._x[:0])
        counts_x = _count_words(
            *sector_x,
            target,
            method,
            lift,
            complement=False,
            base=target if full_pauli else 0,
            reject_lighter=full_pauli,
        )
        counts_z = (
            None
            if counts_x is None
            else _count_words(
                *sector_z,
                target,
                method,
                lift,
                complement=True,
                base=len(self._x) if full_pauli else 0,
                reject_lighter=full_pauli,
            )
        )
        if counts_x is None or counts_z is None:
            raise ValueError(
                "Full Pauli operators can be counted only up to the distance, which is less than"
                f" weight {target}"
            )
        count_pauli = sum(int(count) << exponent for exponent, count in enumerate(counts_x))
        count_pauli += sum(int(count) << exponent for exponent, count in enumerate(counts_z))
        return CSSLogicalCounts(
            target,
            int(counts_x.sum()),
            int(counts_z.sum()),
            count_pauli if full_pauli else None,
        )

    def conjugated(
        self, targets: Sequence[int], matrix_x: npt.ArrayLike, matrix_z: npt.ArrayLike
    ) -> CSSLogicalCounter:
        """Return a counter for this code conjugated by a local CSS-preserving Clifford.

        Args:
            targets: Distinct qubit indices on which the Clifford acts, in order.
            matrix_x: Binary matrix that maps the X support on the targets, as a column vector, to
                its image.
            matrix_z: The same for Z supports.  It must satisfy ``matrix_x @ matrix_z.T = I``
                over GF(2), so that commutation is preserved.

        Returns:
            A new counter.  This counter is not modified.

        Raises:
            ValueError: If the targets or matrices are invalid.
        """
        positions = list(targets)
        size = len(positions)
        if (
            not size
            or any(
                not isinstance(index, (int, np.integer))
                or isinstance(index, bool)
                or not 0 <= index < self.num_qubits
                for index in positions
            )
            or len(set(positions)) != size
        ):
            raise ValueError("Clifford targets must be distinct, in-range qubit indices")
        _assert_binary(matrix_x, "matrix_x")
        _assert_binary(matrix_z, "matrix_z")
        local_x = np.asarray(matrix_x, dtype=int)
        local_z = np.asarray(matrix_z, dtype=int)
        if local_x.shape != (size, size) or local_z.shape != (size, size):
            raise ValueError("Local Clifford matrices must be square and match the target count")
        if not np.array_equal(local_x @ local_z.T % 2, np.eye(size, dtype=int)):
            raise ValueError("Local X/Z matrices must preserve the CSS commutation pairing")

        checks_x, checks_z = self._x.astype(int), self._z.astype(int)
        checks_x[:, positions] = checks_x[:, positions] @ local_x.T % 2
        checks_z[:, positions] = checks_z[:, positions] @ local_z.T % 2
        return CSSLogicalCounter.from_matrices(checks_x, checks_z)

    def conjugated_cnot(self, control: int, target: int) -> CSSLogicalCounter:
        """Return a counter for this code conjugated by a CNOT on zero-based qubit indices."""
        return self.conjugated([control, target], [[1, 0], [1, 1]], [[1, 1], [0, 1]])

    def conjugated_swel(self, targets: Sequence[int]) -> CSSLogicalCounter:
        """Return a counter for this code conjugated by a four-qubit SWEL-preserving Clifford.

        The Clifford applies the same binary matrix, whose rows have weight three, to X and Z
        supports on the four targets.  It therefore maps self-dual codes to self-dual codes, and
        preserves the self-dual-with-equivalent-logicals (SWEL) property of
        :meth:`qldpc.codes.CSSCode.is_swel`.  It is valid for any CSS code.

        Args:
            targets: Four distinct qubit indices, in order.

        Returns:
            A new counter.  This counter is not modified.

        Raises:
            ValueError: If there are not four valid targets.
        """
        if len(targets) != 4:
            raise ValueError("The SWEL-preserving Clifford requires four targets")
        matrix = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]
        return self.conjugated(targets, matrix, matrix)


def count_logicals_css(
    code: CSSCode,
    weight: int | None = None,
    *,
    method: LogicalCountMethod = "brouwer_zimmermann",
    full_pauli: bool = False,
) -> CSSLogicalCounts:
    """Count the nontrivial logical operators of a given weight in a binary CSS stabilizer code.

    This is shorthand for ``CSSLogicalCounter(code).count_logicals(...)``; see
    :meth:`CSSLogicalCounter.count_logicals` for the arguments and conventions.
    """
    return CSSLogicalCounter(code).count_logicals(weight, method=method, full_pauli=full_pauli)


####################################################################################################
# counting words of one weight


def _get_distance(sector_x: _Sector, sector_z: _Sector, method: LogicalCountMethod) -> int:
    """Minimum weight of a nontrivial word in either sector."""
    if method == "brute_force":
        return min(_get_brute_force_distance(sector) for sector in (sector_x, sector_z))
    return _get_distance_brouwer_zimmermann_many(
        [sector_x, sector_z], cutoff=0, block_size=_BLOCK_SIZE
    )


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
    method: LogicalCountMethod,
    lift: _PackedMatrix,
    *,
    complement: bool,
    base: int,
    reject_lighter: bool,
) -> npt.NDArray[np.int64] | None:
    """Visit each nontrivial word of one sector with the given weight exactly once.

    Returns a histogram over exponents ``base - rank(lift restricted to a mask)``, where the mask is
    the support of the word, or its complement.  The histogram sums to the number of words; see the
    module docstring for how the exponents count Pauli operators.  The scan also visits every
    lighter word, so with ``reject_lighter`` it returns None as soon as it finds a nontrivial one.
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
    histogram = np.zeros(base + 1, dtype=np.int64)
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
                    lift,
                    complement,
                    base,
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


def _get_kernel(matrix: _BinaryMatrix) -> _BinaryMatrix:
    """Basis of the null space of a binary matrix, from its reduced row echelon form."""
    reduced, _, pivots = _row_reduce_binary(matrix)
    free = np.setdiff1d(np.arange(matrix.shape[1]), pivots)
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    basis[:, free] = np.eye(len(free), dtype=np.uint8)
    basis[:, pivots] = reduced[: len(pivots), free].T
    return basis


####################################################################################################
# native kernels


@njit(cache=True)
def _count_chunk(
    rows: _PackedMatrix,
    tags: _PackedMatrix,
    decoders: npt.NDArray[np.uint64],
    set_index: int,
    support: npt.NDArray[np.int64],
    depth: int,
    target: int,
    lift: _PackedMatrix,
    complement: bool,
    base: int,
    histogram: npt.NDArray[np.int64],
    reject_lighter: bool,
    chunk_size: int,
) -> tuple[int, bool]:
    """Visit up to ``chunk_size`` combinations of ``len(support)`` rows, resuming at ``depth``.

    ``support`` holds the increasing row indices of the current combination; ``depth`` is the
    position being advanced, or -1 once every combination has been visited.  Each owned nontrivial
    word of weight ``target`` adds one to ``histogram`` (see ``_count_words``).  Returns the depth
    from which to resume, and whether the scan stopped at a nontrivial word lighter than
    ``target``, which is checked only with ``reject_lighter``.
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
    pivots = np.empty((len(lift), width), dtype=np.uint64)
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
                histogram[base - _get_masked_rank(lift, word, complement, pivots)] += 1
        visited += stop - support[depth]
        support[depth] = stop
    return depth, False


@njit(cache=True)
def _is_owned(
    word: npt.NDArray[np.uint64], decoders: npt.NDArray[np.uint64], set_index: int, layer: int
) -> bool:
    """Whether set ``set_index`` minimizes (layer, set index) among the sets that visit ``word``."""
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
