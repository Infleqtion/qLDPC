# SPDX-License-Identifier: Apache-2.0

"""Independent oracles for exact minimum-support enumeration in one CSS sector."""

from __future__ import annotations

import itertools
from unittest import mock

import numpy as np
import pytest

from qldpc import codes

from . import _binary as binary
from . import _weight_supports as supports
from ._weight_rows import StorageBudget
from .conftest import _get_sector_bases, _native

_CODE_422 = ([[1] * 4], [[1] * 4])
_WEIGHT_TWO_WORDS = {3, 5, 6, 9, 10, 12}


def _budget(limit: int = 256 << 20, batch_size: int = 1) -> StorageBudget:
    return StorageBudget(limit_bytes=limit, basis_bytes=72, batch_size=batch_size)


@pytest.mark.parametrize("interpreted", [False, True])
def test_scanner_validation_and_specializations(interpreted: bool) -> None:
    scan = _native(supports._scan_supports, interpreted)
    for width in (1, 2):
        rows = np.array([[1], [2], [4], [8]], dtype=np.uint64)
        rows = np.repeat(rows, width, axis=1)
        support = np.zeros(2, dtype=np.int64)
        prefixes = np.zeros((3, width), dtype=np.uint64)
        output = np.zeros((1, width), dtype=np.uint64)
        tags = prefixes.copy()
        depth = 0
        words: set[int] = set()
        while depth >= 0:
            if interpreted and width == 1:
                native = _native(supports._scan_supports_one, True)
            else:
                native = scan
            depth, count = native(rows, rows, support, prefixes, tags, depth, 2 * width, 1, output)
            words.update(map(int, output[:count, 0]))
        assert words == {3, 5, 6, 9, 10, 12}
        with pytest.raises(IndexError, match="buffers"):
            scan(rows, rows, support, prefixes[:1], tags, 0, 2 * width, 1, output)
        with pytest.raises(IndexError, match="support"):
            scan(rows, rows, np.array([-1, 0]), prefixes, tags, 0, 2 * width, 1, output)


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("width,label_width", [(1, 1), (1, 2), (2, 1), (3, 3)])
def test_batched_leaves_against_combinations(
    interpreted: bool, width: int, label_width: int
) -> None:
    """Sibling batching must emit exactly the eligible lexicographic combinations at seams."""
    rng = np.random.default_rng(621)
    rows = rng.integers(0, np.iinfo(np.uint64).max, size=(7, width), dtype=np.uint64)
    tags = rng.integers(0, 4, size=(7, label_width), dtype=np.uint64)
    tags[:, :-1] = 0
    tags[:3] = 0
    scan = _native(supports._scan_supports, interpreted)
    single = _native(supports._scan_supports_one, interpreted)
    with mock.patch.object(supports, "_scan_supports_one", single):
        for weight in (1, 2, 4, 7):
            candidates = []
            for indices in itertools.combinations(range(len(rows)), weight):
                value = np.bitwise_xor.reduce(rows[list(indices)], axis=0)
                label = np.bitwise_xor.reduce(tags[list(indices)], axis=0)
                candidates.append((value, bool(np.any(label))))
            target = sum(int(word).bit_count() for word in candidates[len(candidates) // 2][0])
            expected = [
                value.tolist()
                for value, eligible in candidates
                if eligible and sum(int(word).bit_count() for word in value) == target
            ]
            for batch_size in (1, 3, 8, 128):
                support = np.zeros(weight, dtype=np.int64)
                prefixes = np.zeros((weight + 1, width), dtype=np.uint64)
                tag_prefixes = np.zeros((weight + 1, label_width), dtype=np.uint64)
                output = np.empty((batch_size, width), dtype=np.uint64)
                depth = 0
                actual: list[list[int]] = []
                while depth >= 0:
                    depth, count = scan(
                        rows,
                        tags,
                        support,
                        prefixes,
                        tag_prefixes,
                        depth,
                        target,
                        batch_size,
                        output,
                    )
                    assert 0 <= count <= batch_size
                    actual.extend(output[:count].tolist())
                assert actual == expected
                np.testing.assert_array_equal(
                    prefixes[weight], np.bitwise_xor.reduce(rows[-weight:], axis=0)
                )
                np.testing.assert_array_equal(
                    tag_prefixes[weight], np.bitwise_xor.reduce(tags[-weight:], axis=0)
                )
                assert scan(
                    rows,
                    tags,
                    support,
                    prefixes,
                    tag_prefixes,
                    -1,
                    target,
                    batch_size,
                    output,
                ) == (-1, 0)


@pytest.mark.parametrize("interpreted", [False, True])
def test_native_exhaustive_corner_cases(interpreted: bool) -> None:
    exhaustive = _native(supports._exhaustive_supports, interpreted)
    minimum_exhaustive = _native(supports._scan_exhaustive_minimum, interpreted)
    rows = np.array([[0, 1], [1, 1]], dtype=np.uint64)
    matches = np.empty((4, 2), dtype=np.uint64)
    assert exhaustive(rows, np.zeros(2, dtype=np.uint64), 0, 1, 4, 1, matches) == 2
    np.testing.assert_array_equal(matches[:2], [[0, 1], [1, 0]])
    assert minimum_exhaustive(rows, np.zeros(2, dtype=np.uint64), 0, 1, 4, 129, matches) == (1, 2)

    # Single-word rows take a scalar fast path; the first row spans the stabilizer subspace.
    one = np.array([[1], [3], [4]], dtype=np.uint64)
    matches = np.empty((8, 1), dtype=np.uint64)
    count = exhaustive(one, np.zeros(1, dtype=np.uint64), 1, 1, 8, 2, matches)
    assert sorted(matches[:count, 0].tolist()) == [3, 5, 6]
    best, count = minimum_exhaustive(one, np.zeros(1, dtype=np.uint64), 1, 1, 8, 65, matches)
    assert (best, sorted(matches[:count, 0].tolist())) == (1, [2, 4])


@pytest.mark.parametrize("interpreted", [False, True])
def test_scan_collects_layer_supports(interpreted: bool) -> None:
    scan = _native(supports._scan_supports, interpreted)
    rows = np.array([[1], [2], [4], [8]], dtype=np.uint64)
    support = np.zeros(2, dtype=np.int64)
    prefixes = np.zeros((3, 1), dtype=np.uint64)
    tags = np.zeros_like(prefixes)
    output = np.zeros((1, 1), dtype=np.uint64)
    depth = 0
    words: set[int] = set()
    while depth >= 0:
        depth, count = scan(rows, rows, support, prefixes, tags, depth, 2, 1, output)
        words.update(map(int, output[:count, 0]))
    assert words == _WEIGHT_TWO_WORDS


def test_enumerators_agree_on_a_known_code() -> None:
    """The [[4,2,2]] code has six weight-two words in each sector."""
    basis, labels = _get_sector_bases(*_CODE_422)[0]
    for batch_size in (1, 4):
        budget = _budget(batch_size=batch_size)
        found = supports.get_bz_supports(basis, labels, 2, budget)
        assert set(map(int, found[:, 0])) == _WEIGHT_TWO_WORDS and len(found) == 6
        found = supports.get_exhaustive_supports(basis, labels, 2, budget)
        assert set(map(int, found[:, 0])) == _WEIGHT_TWO_WORDS and len(found) == 6
        distance, found = supports.get_exhaustive_minimum(basis, labels, budget)
        assert distance == 2 and set(map(int, found[:, 0])) == _WEIGHT_TWO_WORDS
        assert not len(supports.get_bz_supports(basis, labels, 1, budget))
        assert not len(supports.get_exhaustive_supports(basis, labels, 1, budget))


def test_information_sets_packed_once_and_budgeted() -> None:
    basis, labels = _get_sector_bases(*_CODE_422)[0]
    sets = supports.bz._get_information_set_generators(basis, labels)
    selected = supports.bz._select_brouwer_zimmermann_information_sets(
        sets, dimension=len(basis), upper_bound=3, weight_divisor=2
    )
    with mock.patch.object(supports, "pack", wraps=supports.pack) as pack:
        words = supports.get_bz_supports(basis, labels, 2, _budget())
        assert set(map(int, words[:, 0])) == _WEIGHT_TWO_WORDS
        assert pack.call_count == 2 * len(selected)
    with (
        mock.patch.object(supports, "pack", wraps=supports.pack) as pack,
        pytest.raises(MemoryError, match="setup"),
    ):
        supports.get_bz_supports(basis, labels, 2, _budget(limit=1))
    pack.assert_not_called()


@pytest.mark.parametrize("width", [15, 65])
def test_overlapping_supports_keep_strict_coverage_and_physical_ownership(width: int) -> None:
    """Compare overlapping, labeled bases with an independent physical weight-three oracle."""
    checks = np.asarray(codes.HammingCode(4).matrix, dtype=np.uint8)
    x, z = np.zeros((4, width), dtype=np.uint8), np.zeros((width - 11, width), dtype=np.uint8)
    x[:, :15], z[:4, :15] = checks, checks
    z[4:, 15:] = np.eye(width - 15, dtype=np.uint8)
    basis, labels = _get_sector_bases(x, z)[0]
    budget = StorageBudget(256 << 20, basis.nbytes + labels.nbytes, 7)
    orders = (
        np.arange(width),
        np.arange(width)[::-1],
        np.random.default_rng(42).permutation(width),
    )
    family = []
    for order in orders:
        rows, tags, pivots = binary._row_reduce_binary(basis, labels, columns=order)
        family.append(supports.bz._InformationSet(rows, tags, len(pivots), pivots))
    coverage = int(np.bincount(np.concatenate([item.pivots for item in family])).max())
    expected = {
        sum(1 << jj for jj in support)
        for support in itertools.combinations(range(15), 3)
        if not np.any(checks[:, support].sum(axis=1) % 2)
    }
    assert len(expected) == 35 and coverage > 1
    with (
        mock.patch.object(
            supports.bz, "_get_overlapping_information_sets", return_value=(family, coverage)
        ) as select,
        mock.patch.object(
            supports.bz,
            "_get_brouwer_zimmermann_lower_bound",
            wraps=supports.bz._get_brouwer_zimmermann_lower_bound,
        ) as bound,
    ):
        words = supports.get_bz_supports(basis, labels, 3, budget)
    actual = [sum(int(word) << (64 * jj) for jj, word in enumerate(row)) for row in words]
    assert set(actual) == expected and len(actual) == len(expected)
    assert select.call_args.kwargs["upper_bound"] == 4
    counting_calls = [call for call in bound.call_args_list if "max_coverage" in call.kwargs]
    assert counting_calls and all(
        call.kwargs["max_coverage"] == coverage for call in counting_calls
    )
    assert bound.call_args.kwargs["completed_weight"] >= 3

    # With no room for a family beyond its transient workspace, overlap is never attempted.
    tight = StorageBudget(budget.basis_bytes + 1024, budget.basis_bytes, 7)
    with (
        mock.patch.object(supports.bz, "_get_overlapping_information_sets") as select,
        pytest.raises(MemoryError, match="setup"),
    ):
        supports.get_bz_supports(basis, labels, 3, tight)
    select.assert_not_called()


def test_explicit_memory_and_rank_failures() -> None:
    basis, labels = _get_sector_bases(*_CODE_422)[0]
    small = StorageBudget(limit_bytes=1400, basis_bytes=72, batch_size=1 << 16)
    for collect in (supports.get_bz_supports, supports.get_exhaustive_supports):
        with pytest.raises(MemoryError, match="output"):
            collect(basis, labels, 2, small)
    with pytest.raises(MemoryError, match="output"):
        supports.get_exhaustive_minimum(basis, labels, small)
    with pytest.raises(MemoryError, match="setup"):
        supports._get_support_buffer(
            1, 0, StorageBudget(limit_bytes=1, basis_bytes=0, batch_size=1)
        )

    wide, wide_labels = _get_sector_bases(np.zeros((0, 63)), np.zeros((0, 63)))[0]
    assert len(wide) == 63
    with pytest.raises(OverflowError, match="62"):
        supports.get_exhaustive_supports(wide, wide_labels, 1, _budget())
    with pytest.raises(OverflowError, match="62"):
        supports.get_exhaustive_minimum(wide, wide_labels, _budget())
