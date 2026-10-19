# SPDX-License-Identifier: Apache-2.0

"""Exact physical supports of minimum-weight logical words, and their full-Pauli count.

A sector is described by a basis of the normalizer space together with quotient labels: a label
is zero precisely for combinations that lie in the stabilizer space.  A support is reported when
its combination has the requested physical weight and a nonzero label.

Two enumerators share this contract:

- ``get_bz_supports`` scans coefficient supports over information sets.  It stops only when the
  Brouwer--Zimmermann unseen-word bound is strictly greater than the requested weight, because
  counting must also exhaust words whose weight equals the minimum.
- ``get_exhaustive_supports`` and ``get_exhaustive_minimum`` walk every combination of a small
  basis in Gray-code order.

(``collision`` is a third, opt-in way to find the same supports.)  All of them return complete
packed physical supports as rows of uint64 words, and none of them creates one Python object per
support.  Storage is checked before allocation, and a failure raises ``MemoryError`` or
``OverflowError`` rather than returning a partial result.

``count_minimum_paulis`` then lifts the X-only and Z-only supports of both sectors to the number
of physical Pauli operators of minimum weight.  A support shared by both sectors carries an X, a
Z, and a Y representative.  A support found in only one sector can also be dressed by the
opposite-type stabilizers that it contains, which is a single rank query per support; see
``_shortened_histogram``.

A physical coordinate j of a packed row is bit j % 64 of word j // 64, so coordinate zero is the
low bit.  This differs from the layout used by the distance searches in ``_binary``; do not mix
the two.

References:

- https://arxiv.org/abs/1603.06757 (BZ information-set coverage)
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from numba import njit

from .. import brouwer_zimmermann as bz
from .._binary import popcount_word, trailing_zeros

BinaryMatrix = npt.NDArray[np.uint8]
PackedMatrix = npt.NDArray[np.uint64]
_LabeledSet = tuple[BinaryMatrix, BinaryMatrix, int, npt.NDArray[np.int_]]

####################################################################################################
# storage budget


WORD_BYTES = 8

# Working-storage allowances.  Each is a conservative multiple of the bytes of the array it
# describes, covering the copies and temporaries that exist at the same time.  The budget bounds
# working arrays only, not the interpreter's or the JIT's resident memory.  A tracemalloc test in
# counter_test.py checks that measured peaks stay below the limit.

# A basis with its labels, plus the copies made by row reduction and by Clifford transport.
BASIS_COPIES = 4

# Full copies of a support array alive at once while supports are collected (a bytearray and its
# bytes copy), deduplicated (a sort), and lifted (intersection and difference temporaries).
SUPPORT_COPIES = 12

# A native output chunk is charged twice: the chunk itself and its bytes copy in the collector.
OUTPUT_COPIES = 2

# Small arrays and per-call bookkeeping.
SLACK_BYTES = 1024


class StorageBudget(NamedTuple):
    """Conservative working-storage limits for one counting call.

    ``basis_bytes`` is the storage already committed to logical bases.
    """

    limit_bytes: int
    basis_bytes: int
    batch_size: int

    def available(self, reserved_bytes: int = 0) -> int:
        """Bytes left after the bases, ``reserved_bytes`` and a small fixed allowance."""
        return self.limit_bytes - self.basis_bytes - reserved_bytes - SLACK_BYTES


def get_support_bytes(num_supports: int, width: int) -> int:
    """Bytes charged for retaining ``num_supports`` supports of ``width`` packed words."""
    return SUPPORT_COPIES * WORD_BYTES * width * num_supports


def get_output_row_bytes(width: int) -> int:
    """Bytes charged for one row of a native output chunk of ``width`` packed words."""
    return OUTPUT_COPIES * WORD_BYTES * width


####################################################################################################
# packed rows


def pack(matrix: BinaryMatrix) -> PackedMatrix:
    """Pack binary rows into little-endian uint64 words."""
    padded = np.pad(matrix, ((0, 0), (0, -matrix.shape[1] % 64)))
    return np.ascontiguousarray(np.packbits(padded, axis=1, bitorder="little").view("<u8"))


def unpack(matrix: PackedMatrix, width: int) -> BinaryMatrix:
    """Inverse of ``pack`` for rows with ``width`` physical coordinates."""
    return np.unpackbits(matrix.view(np.uint8), axis=1, bitorder="little")[:, :width].copy()


def get_support_keys(words: PackedMatrix) -> npt.NDArray[np.void]:
    """Compare complete packed rows, not individual words or collision-prone hashes."""
    return words.view(np.dtype((np.void, 8 * words.shape[1]))).reshape(-1)


####################################################################################################
# BZ support enumeration


def get_bz_supports(
    basis: BinaryMatrix, labels: BinaryMatrix, weight: int, budget: StorageBudget
) -> PackedMatrix:
    """All distinct physical supports of nontrivial words of exactly ``weight``.

    The caller must pass a ``weight`` that is no lighter than the true minimum, so that the strict
    stopping bound ``lower > weight`` certifies every word of that weight.  Supports found from
    different information sets are deduplicated by their complete physical words.
    """
    width = (basis.shape[1] + 63) // 64
    divisor = 2 if np.all(np.count_nonzero(basis, axis=1) % 2 == 0) else 1

    # Phase 1: choose the information sets whose coverage certifies the strict bound.
    sets, max_coverage = _choose_information_sets(basis, labels, weight, divisor, budget)
    ranks = [rank for _, _, rank, _ in sets]

    # Phase 2: reserve bounded native output space for the sets that remain.
    storage = sum(
        rows.nbytes + tags.nbytes + 8 * len(rows) * (width + (tags.shape[1] + 63) // 64)
        for rows, tags, _, _ in sets
    )
    output, max_words = _get_support_buffer(width, storage, budget)
    packed_sets = [(pack(rows), pack(tags)) for rows, tags, _, _ in sets]

    # Phase 3: scan coefficient layers of increasing weight until no unseen word can be lighter
    # than, or equal in weight to, the target.
    result = bytearray()
    for layer in range(1, len(basis) + 1):
        for packed, packed_tags in packed_sets:
            _scan_layer(packed, packed_tags, layer, weight, output, max_words, result)
        lower = bz._get_brouwer_zimmermann_lower_bound(
            len(basis),
            ranks,
            completed_weight=layer,
            weight_divisor=divisor,
            max_coverage=max_coverage,
        )
        # Equality can still hide more minimum-weight representatives.
        if lower > weight:
            break
    return _unique_supports(_support_array(result, width), len(sets))


def _choose_information_sets(
    basis: BinaryMatrix,
    labels: BinaryMatrix,
    weight: int,
    divisor: int,
    budget: StorageBudget,
) -> tuple[list[_LabeledSet], int]:
    """Select the information sets and their maximum coordinate coverage."""
    width = (basis.shape[1] + 63) // 64
    sets = bz._get_information_set_generators(basis, labels)
    # The distance selector uses >=, so target+1 requests the strict counting certificate.
    sets = bz._select_brouwer_zimmermann_information_sets(
        sets, dimension=len(basis), upper_bound=weight + 1, weight_divisor=divisor
    )
    set_storage = (
        basis.nbytes + labels.nbytes + 8 * len(basis) * (width + (labels.shape[1] + 63) // 64)
    )
    # Reserve both the retained family and transient basis-construction workspace.
    max_sets = min(8, budget.available() // max(1, 2 * set_storage))
    max_coverage = 1
    if max_sets > sum(item.rank == len(basis) for item in sets):
        # Overlapping sets can raise the certified bound, using the actual coordinate coverage.
        overlapping = bz._get_overlapping_information_sets(
            basis,
            labels,
            sets,
            upper_bound=weight + 1,
            weight_divisor=divisor,
            max_sets=max_sets,
        )
        if overlapping is not None:
            sets, max_coverage = overlapping
    checked = []
    for rows, tags, rank, pivots in sets:
        assert tags is not None
        checked.append((rows, tags, rank, pivots))
    return checked, max_coverage


def _scan_layer(
    rows: PackedMatrix,
    tags: PackedMatrix,
    layer: int,
    target: int,
    output: PackedMatrix,
    max_words: int,
    result: bytearray,
) -> None:
    """Append the supports of every combination of ``layer`` rows with physical weight ``target``.

    The scan resumes after each bounded native chunk, which gives the interpreter regular
    opportunities to interrupt it.
    """
    width = rows.shape[1]
    support = np.zeros(layer, dtype=np.int64)
    prefixes = np.zeros((layer + 1, width), dtype=np.uint64)
    tag_prefixes = np.zeros((layer + 1, tags.shape[1]), dtype=np.uint64)
    depth = 0
    while depth >= 0:
        depth, count = _scan_supports(
            rows, tags, support, prefixes, tag_prefixes, depth, target, len(output), output
        )
        _append_supports(result, output, count, max_words)


####################################################################################################
# exhaustive support enumeration


def get_exhaustive_supports(
    basis: BinaryMatrix, labels: BinaryMatrix, weight: int, budget: StorageBudget
) -> PackedMatrix:
    """All supports of nontrivial combinations of exactly ``weight``, by a Gray-code walk."""
    _check_exhaustive_rank(basis)
    width = (basis.shape[1] + 63) // 64
    storage = basis.nbytes + labels.nbytes + 8 * len(basis) * (width + (labels.shape[1] + 63) // 64)
    output, max_words = _get_support_buffer(
        width, storage, budget, max_candidates=(1 << len(basis)) - 1
    )
    packed = pack(basis)
    current = np.zeros(width, dtype=np.uint64)
    # The first rows of a standard-form basis span the stabilizers; these Gray-code bits are
    # ignored, so only words with a nonzero quotient label are reported.
    stabilizer_rank = len(basis) - labels.shape[1]
    result = bytearray()
    for start in range(1, 1 << len(basis), len(output)):
        count = _exhaustive_supports(
            packed,
            current,
            stabilizer_rank,
            start,
            min(start + len(output), 1 << len(basis)),
            weight,
            output,
        )
        _append_supports(result, output, count, max_words)
    return _support_array(result, width)


def get_exhaustive_minimum(
    basis: BinaryMatrix, labels: BinaryMatrix, budget: StorageBudget
) -> tuple[int, PackedMatrix]:
    """Minimum weight of a nontrivial combination, with all of its supports, in one walk."""
    _check_exhaustive_rank(basis)
    rows = pack(basis)
    output, max_words = _get_support_buffer(
        rows.shape[1],
        basis.nbytes + labels.nbytes + rows.nbytes,
        budget,
        max_candidates=(1 << len(basis)) - 1,
    )
    current = np.zeros(rows.shape[1], dtype=np.uint64)
    best = basis.shape[1] + 1
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
            result.clear()  # a lighter word invalidates every support kept so far
        _append_supports(result, output, count, max_words)
    return int(best), _support_array(result, rows.shape[1])


def _check_exhaustive_rank(basis: BinaryMatrix) -> None:
    if len(basis) > 62:
        raise OverflowError("Exhaustive coefficient ranks above 62 are unsupported; use BZ")


####################################################################################################
# full-Pauli lifting


def count_minimum_paulis(
    supports_x: PackedMatrix,
    supports_z: PackedMatrix,
    stabilizers_x: PackedMatrix,
    stabilizers_z: PackedMatrix,
    minimum: int,
    budget: StorageBudget,
) -> int:
    """Count physical Pauli representatives of weight ``minimum``, ignoring phase.

    ``supports_x`` and ``supports_z`` are the distinct supports of the minimum X-only and Z-only
    logical words.  Stabilizers are packed row spaces.  The result is an arbitrary-size integer.
    """
    width = stabilizers_x.shape[1]
    keys_x, keys_z = get_support_keys(supports_x), get_support_keys(supports_z)

    # A support shared by both sectors carries exactly X, Z and Y words, three in all.
    shared = np.intersect1d(keys_x, keys_z, assume_unique=True)
    union_size = len(supports_x) + len(supports_z) - len(shared)
    if get_support_bytes(union_size, width) + budget.basis_bytes > budget.limit_bytes:
        raise MemoryError("Full-Pauli support union exceeds memory_limit_bytes")
    total = 3 * len(shared)

    # An X-only support can be dressed by Z stabilizers contained inside it, and conversely.
    # This needs one rank query per support, not one per candidate word.
    for keys, stabilizers in (
        (np.setdiff1d(keys_x, keys_z, assume_unique=True), stabilizers_z),
        (np.setdiff1d(keys_z, keys_x, assume_unique=True), stabilizers_x),
    ):
        supports = keys.view(np.uint64).reshape(-1, width)
        for start in range(0, len(supports), budget.batch_size):
            words = supports[start : start + budget.batch_size]
            histogram = _shortened_histogram(stabilizers, words, minimum)
            total += sum(
                int(count) << exponent for exponent, count in enumerate(histogram) if count
            )
    return total


####################################################################################################
# helper methods


def _get_support_buffer(
    width: int, storage: int, budget: StorageBudget, *, max_candidates: int | None = None
) -> tuple[PackedMatrix, int]:
    """Reserve bounded native output and conservative bulk deduplication workspace.

    Returns the output chunk and the number of retained supports that fit in the budget.
    """
    allowance = budget.available(storage)
    # One extra word per row of slack for the chunk's counters and indices.
    chunk_size = min(budget.batch_size, allowance // get_output_row_bytes(width + 1))
    if max_candidates is not None:
        chunk_size = min(chunk_size, max_candidates)
    if chunk_size < 1:
        raise MemoryError("Support enumeration setup exceeds memory_limit_bytes")
    return np.empty((chunk_size, width), dtype=np.uint64), allowance // get_support_bytes(1, width)


def _append_supports(buffer: bytearray, output: PackedMatrix, count: int, max_words: int) -> None:
    """Copy matching rows in bulk, never creating one Python object per representative."""
    word_bytes = output.shape[1] * 8
    if len(buffer) + count * word_bytes > max_words * word_bytes:
        raise MemoryError("Minimum-support output exceeds memory_limit_bytes")
    if count:
        buffer.extend(output[:count].tobytes())


def _support_array(buffer: bytearray, width: int) -> PackedMatrix:
    return np.frombuffer(bytes(buffer), dtype=np.uint64).reshape(-1, width)


def _unique_supports(words: PackedMatrix, num_sets: int) -> PackedMatrix:
    """Deduplicate across information sets; a single set already visits each word once."""
    if num_sets == 1:
        return words
    if words.shape[1] == 1:
        return np.unique(words[:, 0]).reshape(-1, 1)
    return np.unique(get_support_keys(words)).view(np.uint64).reshape(-1, words.shape[1])


####################################################################################################
# native kernels


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
