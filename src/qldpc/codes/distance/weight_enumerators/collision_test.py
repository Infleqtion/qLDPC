# SPDX-License-Identifier: Apache-2.0

"""Independent physical-combination oracles for the internal collision join."""

from __future__ import annotations

import itertools
from typing import Any
from unittest import mock

import numpy as np
import pytest

from . import collision as c


def _words(rows: np.ndarray) -> list[int]:
    return [sum(int(word) << (64 * jj) for jj, word in enumerate(row)) for row in rows]


def _oracle(s: np.ndarray, labels: np.ndarray, weight: int) -> set[int]:
    result = set()
    for positions in itertools.combinations(range(len(s)), weight):
        syndrome = np.bitwise_xor.reduce(s[list(positions)], axis=0)
        label = np.bitwise_xor.reduce(labels[list(positions)], axis=0)
        if not np.any(syndrome) and np.any(label):
            result.add(sum(1 << position for position in positions))
    return result


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("batch", [1, 7, 65536])
def test_all_weights_and_degenerate_chains(interpreted: bool, batch: int) -> None:
    """No minimum assumption: odd/even joins and repeated syndromes have unique owners."""
    names = ("_build_chunk", "_query_chunk", "_next_support", "_bucket", "_validate_support")
    patches = (
        {name: getattr(getattr(c, name), "py_func", getattr(c, name)) for name in names}
        if interpreted
        else {}
    )
    with mock.patch.multiple(c, **patches) if patches else mock.patch.dict({}, {}):
        rng = np.random.default_rng(443)
        for n in (5, 8):
            for zero in (False, True):
                s = rng.integers(0, 4, (n, 2), dtype=np.uint64)
                s[:, 1] <<= np.uint64(63)
                if zero:
                    s[:] = 0
                labels = rng.integers(0, 4, (n, 3), dtype=np.uint64)
                labels[:, 2] <<= np.uint64(62)
                for weight in range(1, n + 1):
                    rows = c.recover_supports(
                        s,
                        labels,
                        weight,
                        memory_limit_bytes=1 << 20,
                        reserved_bytes=0,
                        batch_size=batch,
                    )
                    words = _words(rows)
                    assert len(words) == len(set(words))
                    assert set(words) == _oracle(s, labels, weight)


def test_multiword_physical_supports_and_empty_maps() -> None:
    """The high physical, syndrome and quotient bits are essential to this oracle."""
    n = 130
    s = np.zeros((n, 2), dtype=np.uint64)
    s[2:, 1] = np.uint64(1) << np.uint64(63)
    labels = np.zeros((n, 3), dtype=np.uint64)
    labels[129, 2] = np.uint64(1) << np.uint64(63)
    for weight in (1, 2, 3):
        rows = c.recover_supports(
            s,
            labels,
            weight,
            memory_limit_bytes=8 << 20,
            reserved_bytes=0,
            batch_size=901,
        )
        assert set(_words(rows)) == _oracle(s, labels, weight)
    empty = np.empty((3, 0), dtype=np.uint64)
    rows = c.recover_supports(
        empty,
        empty,
        2,
        memory_limit_bytes=1 << 20,
        reserved_bytes=0,
        batch_size=2,
    )
    assert rows.shape == (0, 1)


def test_preflights_before_allocations() -> None:
    s = np.zeros((100, 2), dtype=np.uint64)
    for weight, memory, message in ((50, 1 << 20, "capacity"), (10, 1 << 20, "setup")):
        with (
            mock.patch.object(c.np, "empty", side_effect=AssertionError("allocated")),
            pytest.raises((OverflowError, MemoryError), match=message),
        ):
            c.recover_supports(
                s,
                s,
                weight,
                memory_limit_bytes=memory,
                reserved_bytes=0,
                batch_size=2,
            )
    for kwargs in (
        {"weight": 0},
        {"weight": 101},
        {"weight": True},
        {"memory_limit_bytes": 0},
        {"reserved_bytes": -1},
        {"batch_size": 0},
        {"batch_size": 1.5},
    ):
        options: dict[str, Any] = {
            "weight": 2,
            "memory_limit_bytes": 1 << 20,
            "reserved_bytes": 0,
            "batch_size": 2,
        }
        options.update(kwargs)
        with pytest.raises(ValueError):
            c.recover_supports(s, s, **options)
    for labels in (s.astype(np.uint8), s[:1], s[:, 0]):
        invalid_labels: Any = labels
        with pytest.raises(ValueError):
            c.recover_supports(
                s, invalid_labels, 2, memory_limit_bytes=1 << 20, reserved_bytes=0, batch_size=1
            )
    with pytest.raises(ValueError):
        c.recover_supports(
            s[:, 0], s, 2, memory_limit_bytes=1 << 20, reserved_bytes=0, batch_size=1
        )
    # A huge Python batch limit is clamped before crossing the native boundary.
    assert (
        len(
            c.recover_supports(
                s[:2],
                np.ones((2, 1), dtype=np.uint64),
                1,
                memory_limit_bytes=1 << 20,
                reserved_bytes=0,
                batch_size=2**100,
            )
        )
        == 2
    )
    with pytest.raises(MemoryError, match="support storage"):
        c.recover_supports(s[:1], s[:1], 1, memory_limit_bytes=8512, reserved_bytes=0, batch_size=1)
    # Setup fits, but this degenerate shell has far more distinct output than the budget.
    with pytest.raises(MemoryError, match="output"):
        c.recover_supports(
            np.zeros((24, 1), dtype=np.uint64),
            np.eye(24, dtype=np.uint64),
            5,
            memory_limit_bytes=180_000,
            reserved_bytes=0,
            batch_size=1,
        )


def test_output_chunk_address_preflight() -> None:
    """Even a zero-width half-shell must not allocate before checking all output bytes."""
    empty_maps = np.empty((4096, 0), dtype=np.uint64)
    with mock.patch.multiple(
        c.np, empty=mock.DEFAULT, full=mock.DEFAULT, arange=mock.DEFAULT
    ) as allocs:
        for allocation in allocs.values():
            allocation.side_effect = AssertionError("allocation sentinel")
        with pytest.raises(OverflowError, match=r"output chunk.*address capacity"):
            c.recover_supports(
                empty_maps,
                empty_maps,
                7,
                memory_limit_bytes=10**30,
                reserved_bytes=0,
                batch_size=2**100,
            )
        for allocation in allocs.values():
            allocation.assert_not_called()


@pytest.mark.parametrize(
    ("capacity", "message"),
    (
        (8783, "half-shell setup"),
        (8784, "output chunk"),
        (8847, "output chunk"),
        (8848, "support storage"),
        (8943, "support storage"),
    ),
)
def test_aggregate_address_preflights(
    capacity: int, message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup is 8784 bytes, the output chunk is 64, and one support needs 96 more."""
    empty_maps = np.empty((6, 0), dtype=np.uint64)
    monkeypatch.setattr(c, "_MAX_INDEX", capacity)
    with mock.patch.multiple(
        c.np, empty=mock.DEFAULT, full=mock.DEFAULT, arange=mock.DEFAULT
    ) as allocs:
        for allocation in allocs.values():
            allocation.side_effect = AssertionError("allocation sentinel")
        with pytest.raises(OverflowError, match=rf"{message}.*address capacity"):
            c.recover_supports(
                empty_maps,
                empty_maps,
                3,
                memory_limit_bytes=10**30,
                reserved_bytes=0,
                batch_size=8,
            )
        for allocation in allocs.values():
            allocation.assert_not_called()


@pytest.mark.parametrize("capacity", [8944, 8945])
def test_aggregate_address_boundary(capacity: int, monkeypatch: pytest.MonkeyPatch) -> None:
    empty_maps = np.empty((6, 0), dtype=np.uint64)
    monkeypatch.setattr(c, "_MAX_INDEX", capacity)
    result = c.recover_supports(
        empty_maps,
        empty_maps,
        3,
        memory_limit_bytes=10**30,
        reserved_bytes=0,
        batch_size=8,
    )
    assert result.shape == (0, 1)


def test_reserved_address_preflight() -> None:
    empty_maps = np.empty((6, 0), dtype=np.uint64)
    reserved = c._MAX_INDEX - 8784 + 1
    with (
        mock.patch.object(c.np, "empty", side_effect=AssertionError("allocation sentinel")),
        mock.patch.object(c.np, "full", side_effect=AssertionError("allocation sentinel")),
        pytest.raises(OverflowError, match=r"half-shell setup.*address capacity"),
    ):
        c.recover_supports(
            empty_maps,
            empty_maps,
            3,
            memory_limit_bytes=10**30,
            reserved_bytes=reserved,
            batch_size=8,
        )


@pytest.mark.parametrize("fits", [False, True])
def test_output_growth_address_boundary(fits: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Four supports require exactly 8464 setup + 8 chunk + 4 * 96 output bytes."""
    syndromes = np.empty((4, 0), dtype=np.uint64)
    labels = np.ones((4, 1), dtype=np.uint64)
    monkeypatch.setattr(c, "_MAX_INDEX", 8856 - int(not fits))
    if not fits:
        with pytest.raises(OverflowError, match=r"minimum-support output.*address capacity"):
            c.recover_supports(
                syndromes, labels, 1, memory_limit_bytes=10**30, reserved_bytes=0, batch_size=1
            )
    else:
        result = c.recover_supports(
            syndromes, labels, 1, memory_limit_bytes=10**30, reserved_bytes=0, batch_size=1
        )
        assert _words(result) == [1, 2, 4, 8]


@pytest.mark.parametrize("memory", [1 << 20, 10**30])
def test_huge_python_limits_on_small_inputs(memory: int) -> None:
    result = c.recover_supports(
        np.empty((2, 0), dtype=np.uint64),
        np.ones((2, 1), dtype=np.uint64),
        1,
        memory_limit_bytes=memory,
        reserved_bytes=0,
        batch_size=2**100,
    )
    assert _words(result) == [1, 2]


@pytest.mark.parametrize("interpreted", [False, True])
def test_native_buffers_and_resumable_work(
    interpreted: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each call is bounded even in an all-zero-syndrome collision chain."""
    n = 8
    if interpreted:
        for name in (
            "_build_chunk",
            "_query_chunk",
            "_validate_support",
            "_bucket",
            "_next_support",
        ):
            function = getattr(c, name)
            monkeypatch.setattr(c, name, getattr(function, "py_func", function))
    s = np.zeros((n, 1), dtype=np.uint64)
    labels = np.eye(n, dtype=np.uint64)
    shell_s = np.empty((n, 1), dtype=np.uint64)
    shell_l = np.empty((n, n), dtype=np.uint64)
    shell_p = np.empty((n, 1), dtype=np.uint64)
    maxima, links = np.empty(n, dtype=np.int64), np.empty(n, dtype=np.int64)
    heads = np.full(16, -1, dtype=np.int64)
    support = np.array([0], dtype=np.int64)
    args: list[Any] = [s, labels, support, 0, n, shell_s, shell_l, shell_p, maxima, links, heads]
    c._build_chunk(*args)
    query = np.array([0], dtype=np.int64)
    qs, ql = np.empty(1, dtype=np.uint64), np.empty(n, dtype=np.uint64)
    output = np.empty((1, 1), dtype=np.uint64)
    query_args: list[Any] = [
        s,
        labels,
        query,
        qs,
        ql,
        shell_s,
        shell_l,
        shell_p,
        maxima,
        links,
        heads,
        -2,
        1,
        output,
    ]
    done, words, visits = False, [], 0
    while not done:
        cursor, done, count, visited = c._query_chunk(*query_args)
        assert visited <= 1
        visits += visited
        words.extend(_words(output[:count]))
        query_args[11] = cursor
    assert len(words) == len(set(words)) == 28
    assert visits == n * (n + 1)
    for index, value in ((3, -1), (4, n + 1), (10, np.empty(0, dtype=np.int64))):
        invalid = args.copy()
        invalid[index] = value
        with pytest.raises(IndexError, match="buffers"):
            c._build_chunk(*invalid)
    for bad_support in (np.array([-1]), np.array([n]), np.array([1, 1])):
        invalid = args.copy()
        invalid[2] = bad_support
        with pytest.raises(IndexError, match="coordinate"):
            c._build_chunk(*invalid)
    query_args[11] = -2
    for bad_support in (
        np.array([-1], dtype=np.int64),
        np.array([n], dtype=np.int64),
        np.array([1, 1], dtype=np.int64),
        np.array([2, 1], dtype=np.int64),
    ):
        invalid = query_args.copy()
        invalid[2] = bad_support
        output[:] = 123
        with pytest.raises(IndexError, match="coordinate"):
            c._query_chunk(*invalid)
        assert np.all(output == 123)
    for index, query_value in ((11, n), (12, 0), (13, np.empty((0, 1), dtype=np.uint64))):
        invalid = query_args.copy()
        invalid[index] = query_value
        with pytest.raises(IndexError, match="buffers"):
            c._query_chunk(*invalid)
    invalid_query = query_args.copy()
    invalid_query[-2] = 2
    invalid_query[-1] = np.empty((2, 1), dtype=np.uint64)
    for bad in (-3, n + 1):
        heads[:] = bad
        with pytest.raises(IndexError, match="chain"):
            c._query_chunk(*invalid_query)
    heads[:] = n - 1
    links[:] = n
    with pytest.raises(IndexError, match="chain"):
        c._query_chunk(*invalid_query)


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("batch", [1, 17])
def test_ordered_multiword_query_coordinates(
    interpreted: bool, batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Validated coordinate lookup preserves chain order across physical word boundaries."""
    if interpreted:
        for name in (
            "_build_chunk",
            "_query_chunk",
            "_validate_support",
            "_bucket",
            "_next_support",
        ):
            function = getattr(c, name)
            monkeypatch.setattr(c, name, getattr(function, "py_func", function))
    n = 130
    groups = np.arange(n, dtype=np.uint64) % np.uint64(7)
    syndromes = np.column_stack((groups, groups << np.uint64(61)))
    labels = np.column_stack((np.arange(1, n + 1, dtype=np.uint64), groups))
    actual = c.recover_supports(
        syndromes, labels, 2, memory_limit_bytes=1 << 20, reserved_bytes=0, batch_size=batch
    )
    expected = [
        (1 << left) | (1 << right)
        for right in range(n)
        for left in range(right - 1, -1, -1)
        if groups[left] == groups[right]
    ]
    assert _words(actual) == expected
