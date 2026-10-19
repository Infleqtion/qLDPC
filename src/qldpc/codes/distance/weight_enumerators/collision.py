# SPDX-License-Identifier: Apache-2.0

"""Bounded physical half-shell joins for exact binary CSS support recovery.

This is an internal counting backend, not a hash-based approximation.
Every hash hit is checked against the complete multiword syndrome and logical label.
For weight w, split a support after its first floor(w/2) coordinates.
The two halves have equal syndromes precisely when their union is a normalizer word.
A nonzero XOR of their logical labels excludes stabilizers.
Requiring every left coordinate to precede every right coordinate gives each physical support
exactly one owner, for odd as well as even w, without storing a deduplication table.
Thus the join is valid at any weight, not only a previously certified minimum.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from numba import njit

from .supports import WORD_BYTES, get_output_row_bytes, get_support_bytes

PackedMatrix = npt.NDArray[np.uint64]
IndexArray = npt.NDArray[np.int64]
_MAX_INDEX = int(np.iinfo(np.intp).max)

# Fixed allowance for small native scratch arrays and bookkeeping.
_SETUP_SLACK_BYTES = 8192

# Four int64 words of index scratch per coordinate, weight position and shell-row word.
_INDEX_SCRATCH_BYTES = 32


def recover_supports(
    syndromes: PackedMatrix,
    labels: PackedMatrix,
    weight: int,
    *,
    memory_limit_bytes: int,
    reserved_bytes: int,
    batch_size: int,
) -> PackedMatrix:
    """Recover all nontrivial supports of one weight from packed per-coordinate maps.

    The caller supplies an ambient linear label map whose restriction to the normalizer has
    kernel equal to the stabilizer space.
    ``reserved_bytes`` includes caller-owned bases, maps and previously recovered supports.
    Both shell construction and chained queries return to Python in bounded work chunks, including
    zero-syndrome groups.
    Storage preflights include the retained shell, hash chains, output copies and lifting/dedup
    allowance; failures are explicit, with no fallback or partially successful result.
    Setup, complete chunk bytes and growing output must also fit native address capacity.
    The budget bounds working arrays, not the interpreter/JIT's resident memory.
    """
    _validate_arguments(syndromes, labels, weight, memory_limit_bytes, reserved_bytes, batch_size)
    plan = _plan_storage(
        syndromes, labels, int(weight), int(memory_limit_bytes), int(reserved_bytes), batch_size
    )
    shell = _build_shell(syndromes, labels, plan)
    return _query_shell(syndromes, labels, plan, shell)


class _Plan(NamedTuple):
    """Sizes of the half-shell and its output, all checked against the storage budget."""

    weight: int
    left_size: int
    right_size: int
    num_shell_rows: int
    table_size: int
    physical_width: int
    chunk: int
    max_supports: int
    max_address_supports: int


class _Shell(NamedTuple):
    """Left half-shell: signatures, labels and physical halves, indexed by hash chains."""

    signatures: PackedMatrix
    labels: PackedMatrix
    physical: PackedMatrix
    maxima: IndexArray
    links: IndexArray
    heads: IndexArray


def _validate_arguments(
    syndromes: PackedMatrix,
    labels: PackedMatrix,
    weight: int,
    memory_limit_bytes: int,
    reserved_bytes: int,
    batch_size: int,
) -> None:
    if syndromes.ndim != 2 or labels.ndim != 2:
        raise ValueError("Incompatible collision maps or physical weight")
    n = len(syndromes)
    if (
        labels.dtype != np.uint64
        or syndromes.dtype != np.uint64
        or len(labels) != n
        or not isinstance(weight, (int, np.integer))
        or isinstance(weight, bool)
        or not 1 <= weight <= n
    ):
        raise ValueError("Incompatible collision maps or physical weight")
    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < minimum
        for value, minimum in (
            (memory_limit_bytes, 1),
            (reserved_bytes, 0),
            (batch_size, 1),
        )
    ):
        raise ValueError("Collision memory and batch limits must be nonnegative integers")


def _plan_storage(
    syndromes: PackedMatrix,
    labels: PackedMatrix,
    weight: int,
    memory_limit_bytes: int,
    reserved_bytes: int,
    batch_size: int,
) -> _Plan:
    """Size every array in Python integers and reject over-budget work before allocating."""
    n = len(syndromes)
    left_size, right_size = weight // 2, (weight + 1) // 2
    size = math.comb(n, left_size)
    table_size = 1 << (2 * size - 1).bit_length()
    physical_width = (n + 63) // 64
    row_words = syndromes.shape[1] + labels.shape[1] + physical_width
    # All index/count arithmetic above is Python integer arithmetic, before native conversion.
    if max(size, table_size, size * row_words, n * row_words) > _MAX_INDEX // 8:
        raise OverflowError("Collision half-shell exceeds native index/address capacity")
    # Exact bytes of the shell: each row holds its words, a maximum coordinate and a chain link,
    # and the hash table holds one head per bucket.  Scratch index arrays are charged generously.
    setup = (
        reserved_bytes
        + _SETUP_SLACK_BYTES
        + WORD_BYTES * (size * (row_words + 2) + table_size)
        + _INDEX_SCRATCH_BYTES * (n + weight + row_words)
    )
    if setup > _MAX_INDEX:
        raise OverflowError("Collision half-shell setup exceeds native index/address capacity")
    allowance = memory_limit_bytes - setup
    chunk = min(int(batch_size), _MAX_INDEX, math.comb(n, weight))
    chunk = min(chunk, allowance // get_output_row_bytes(physical_width))
    if chunk < 1:
        raise MemoryError("Collision half-shell setup exceeds memory_limit_bytes")
    output_bytes = chunk * WORD_BYTES * physical_width
    if setup + output_bytes > _MAX_INDEX:
        raise OverflowError("Collision output chunk exceeds native index/address capacity")
    max_supports = (allowance - output_bytes) // get_support_bytes(1, physical_width)
    if max_supports < 1:
        raise MemoryError("Collision support storage exceeds memory_limit_bytes")
    max_address_supports = (_MAX_INDEX - setup - output_bytes) // get_support_bytes(
        1, physical_width
    )
    if max_address_supports < 1:
        raise OverflowError("Collision support storage exceeds native index/address capacity")
    return _Plan(
        weight,
        left_size,
        right_size,
        size,
        table_size,
        physical_width,
        chunk,
        max_supports,
        max_address_supports,
    )


def _build_shell(syndromes: PackedMatrix, labels: PackedMatrix, plan: _Plan) -> _Shell:
    """Enumerate every left half-support, in bounded chunks, into a hash-chained table."""
    size = plan.num_shell_rows
    shell = _Shell(
        np.empty((size, syndromes.shape[1]), dtype=np.uint64),
        np.empty((size, labels.shape[1]), dtype=np.uint64),
        np.empty((size, plan.physical_width), dtype=np.uint64),
        np.empty(size, dtype=np.int64),
        np.empty(size, dtype=np.int64),
        np.full(plan.table_size, -1, dtype=np.int64),
    )
    left = np.arange(plan.left_size, dtype=np.int64)
    for start in range(0, size, plan.chunk):
        _build_chunk(syndromes, labels, left, start, min(start + plan.chunk, size), *shell)
    return shell


def _query_shell(
    syndromes: PackedMatrix, labels: PackedMatrix, plan: _Plan, shell: _Shell
) -> PackedMatrix:
    """Join every right half-support against the shell, resuming in bounded work chunks."""
    right = np.arange(plan.right_size, dtype=np.int64)
    query_s = np.empty(syndromes.shape[1], dtype=np.uint64)
    query_l = np.empty(labels.shape[1], dtype=np.uint64)
    output = np.empty((plan.chunk, plan.physical_width), dtype=np.uint64)
    result = bytearray()
    cursor, done = -2, False
    while not done:
        cursor, done, count, _ = _query_chunk(
            syndromes, labels, right, query_s, query_l, *shell, cursor, plan.chunk, output
        )
        # Check the growing output before keeping it, so failures are explicit and never partial.
        num_supports = len(result) // (8 * plan.physical_width) + count
        if num_supports > plan.max_supports:
            raise MemoryError("Collision minimum-support output exceeds memory_limit_bytes")
        if num_supports > plan.max_address_supports:
            raise OverflowError(
                "Collision minimum-support output exceeds native index/address capacity"
            )
        if count:
            result.extend(output[:count].tobytes())
    return np.frombuffer(bytes(result), dtype=np.uint64).reshape(-1, plan.physical_width)


####################################################################################################
# helper methods


@njit(cache=True, inline="always")
def _next_support(support: IndexArray, n: int) -> bool:
    """Advance lexicographic combinations; False marks the exhausted (also empty) shell."""
    for index in range(len(support) - 1, -1, -1):
        if support[index] < n - len(support) + index:
            support[index] += 1
            for following in range(index + 1, len(support)):
                support[following] = support[following - 1] + 1
            return True
    return False


@njit(cache=True, inline="always")
def _bucket(signature: npt.NDArray[np.uint64], mask: int) -> int:
    """Hash only selects a chain; full signatures are always compared."""
    value = np.uint64(0)
    for word in signature:
        value ^= word
        value ^= value >> np.uint64(30)
        value ^= value << np.uint64(17)
        value ^= value >> np.uint64(23)
    return int(value & np.uint64(mask))


@njit(cache=True)
def _build_chunk(
    syndromes: PackedMatrix,
    labels: PackedMatrix,
    support: IndexArray,
    start: int,
    stop: int,
    shell_s: PackedMatrix,
    shell_l: PackedMatrix,
    shell_p: PackedMatrix,
    maxima: IndexArray,
    links: IndexArray,
    heads: IndexArray,
) -> None:
    """Generate signatures, physical halves and hash chains for one bounded shell chunk."""
    if (
        not 0 <= start <= stop <= len(shell_s)
        or shell_s.shape[1] != syndromes.shape[1]
        or shell_l.shape != (len(shell_s), labels.shape[1])
        or shell_p.shape != (len(shell_s), (len(syndromes) + 63) // 64)
        or len(labels) != len(syndromes)
        or len(maxima) != len(shell_s)
        or len(links) != len(shell_s)
        or not len(heads)
        or len(heads) & (len(heads) - 1)
    ):
        raise IndexError("Incompatible collision shell buffers")
    _validate_support(support, len(syndromes))
    for row in range(start, stop):
        for word in range(shell_s.shape[1]):
            value = np.uint64(0)
            for coordinate in support:
                value ^= syndromes[coordinate, word]
            shell_s[row, word] = value
        for word in range(shell_l.shape[1]):
            value = np.uint64(0)
            for coordinate in support:
                value ^= labels[coordinate, word]
            shell_l[row, word] = value
        shell_p[row, :] = 0
        for coordinate in support:
            shell_p[row, coordinate // 64] |= np.uint64(1) << np.uint64(coordinate % 64)
        maxima[row] = support[-1] if len(support) else -1
        bucket = _bucket(shell_s[row], len(heads) - 1)
        links[row] = heads[bucket]
        heads[bucket] = row
        _next_support(support, len(syndromes))


@njit(cache=True)
def _query_chunk(
    syndromes: PackedMatrix,
    labels: PackedMatrix,
    support: IndexArray,
    query_s: npt.NDArray[np.uint64],
    query_l: npt.NDArray[np.uint64],
    shell_s: PackedMatrix,
    shell_l: PackedMatrix,
    shell_p: PackedMatrix,
    maxima: IndexArray,
    links: IndexArray,
    heads: IndexArray,
    cursor: int,
    batch_size: int,
    output: PackedMatrix,
) -> tuple[int, bool, int, int]:
    """Resume after at most batch_size query preparations/chain visits, not just matches."""
    if (
        not len(support)
        or len(labels) != len(syndromes)
        or len(query_s) != syndromes.shape[1]
        or len(query_l) != labels.shape[1]
        or shell_s.shape[1] != len(query_s)
        or shell_l.shape != (len(shell_s), len(query_l))
        or shell_p.shape != (len(shell_s), (len(syndromes) + 63) // 64)
        or len(maxima) != len(shell_s)
        or len(links) != len(shell_s)
        or not len(heads)
        or len(heads) & (len(heads) - 1)
        or batch_size < 1
        or len(output) < batch_size
        or output.shape[1] != shell_p.shape[1]
        or not -2 <= cursor < len(shell_s)
    ):
        raise IndexError("Incompatible collision query buffers")
    _validate_support(support, len(syndromes))
    # Validation and lexicographic advancement keep coordinates nonnegative and in bounds.
    # Unsigned indexing avoids native negative-index adjustment on every map lookup.
    count, visited = 0, 0
    while visited < batch_size:
        if cursor == -1:
            if not _next_support(support, len(syndromes)):
                return cursor, True, count, visited
            cursor = -2
        if cursor == -2:
            for word in range(len(query_s)):
                value = np.uint64(0)
                for index in range(len(support)):
                    coordinate = np.uint64(support[index])
                    value ^= syndromes[coordinate, word]
                query_s[word] = value
            for word in range(len(query_l)):
                value = np.uint64(0)
                for index in range(len(support)):
                    coordinate = np.uint64(support[index])
                    value ^= labels[coordinate, word]
                query_l[word] = value
            cursor = heads[_bucket(query_s, len(heads) - 1)]
        else:
            if cursor >= len(shell_s):
                raise IndexError("Invalid collision chain index")
            equal = maxima[cursor] < support[0]
            for word in range(len(query_s)):
                equal = equal and shell_s[cursor, word] == query_s[word]
            nonzero = False
            for word in range(len(query_l)):
                nonzero = nonzero or shell_l[cursor, word] != query_l[word]
            if equal and nonzero:
                for word in range(shell_p.shape[1]):
                    output[count, word] = shell_p[cursor, word]
                for coordinate in support:
                    output[count, coordinate // 64] |= np.uint64(1) << np.uint64(coordinate % 64)
                count += 1
            following = links[cursor]
            if not -1 <= following < cursor:
                raise IndexError("Invalid collision chain index")
            cursor = following
        if cursor < -1:
            raise IndexError("Invalid collision chain index")
        visited += 1
    return cursor, False, count, visited


@njit(cache=True, inline="always")
def _validate_support(support: IndexArray, n: int) -> None:
    for index in range(len(support)):
        if not 0 <= support[index] < n or (index > 0 and support[index - 1] >= support[index]):
            raise IndexError("Invalid collision coordinate support")
