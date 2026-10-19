# SPDX-License-Identifier: Apache-2.0

"""Independent invariant tests for exact binary CSS weight enumeration."""

from __future__ import annotations

import functools
import itertools
import math
from typing import Any
from unittest import mock

import galois
import numpy as np
import pytest

from qldpc import codes
from qldpc.codes.distance import weight_enumerators as we


def _span(matrix: np.ndarray) -> set[int]:
    words = {0}
    for row in matrix:
        word = sum(int(bit) << jj for jj, bit in enumerate(row))
        words |= {old ^ word for old in words}
    return words


def _oracle(counter: we.CSSLogicalWeightCounter) -> tuple[list[int], list[int], int]:
    x, z, n = counter.checks_x, counter.checks_z, counter.num_qubits
    sx, sz = _span(x), _span(z)
    cx = {
        word for word in range(1 << n) if all((word & check).bit_count() % 2 == 0 for check in sz)
    }
    cz = {
        word for word in range(1 << n) if all((word & check).bit_count() % 2 == 0 for check in sx)
    }
    ax, az = [0] * (n + 1), [0] * (n + 1)
    for word in cx - sx:
        ax[word.bit_count()] += 1
    for word in cz - sz:
        az[word.bit_count()] += 1
    minimum = next((w for w in range(n + 1) if ax[w] or az[w]), None)
    full = sum(
        (xx | zz).bit_count() == minimum
        for xx, zz in itertools.product(cx, cz)
        if xx not in sx or zz not in sz
    )
    return ax, az, full


@pytest.mark.parametrize("interpreted", [False, True])
def test_random_css_oracles(interpreted: bool) -> None:
    """Compare every coefficient and the full normalizer, not just cached distances."""
    kernels = (
        "_histogram_chunk",
        "_minimum_native",
        "_scan_supports",
        "_scan_supports_one",
        "_exhaustive_supports",
        "_scan_exhaustive_minimum",
        "_rank_masked",
        "_rank_word",
        "_shortened_histogram",
        "_shortened_histogram_one",
        "_transform",
    )
    with (
        mock.patch.multiple(
            we,
            **{name: getattr(getattr(we, name), "py_func", getattr(we, name)) for name in kernels},
        )
        if interpreted
        else mock.patch.dict({}, {})
    ):
        rng = np.random.default_rng(810)
        for n in (4, 5, 6):
            for symmetric in (False, True):
                x = np.zeros((1, n), dtype=np.uint8)
                z = np.zeros((1, n), dtype=np.uint8)
                x[0, 0] = 1
                if symmetric:
                    x[0, 1] = 1
                    z = x.copy()
                else:
                    z[0, 1] = 1
                counter = we.CSSLogicalWeightCounter(x, z, batch_size=3)
                for _ in range(5):
                    targets = list(map(int, rng.choice(n, 4 if symmetric else 2, replace=False)))
                    counter = (
                        counter.conjugated_swel(targets)
                        if symmetric
                        else counter.conjugated_cnot(*targets)
                    )
                ax, az, full = _oracle(counter)
                spectrum = counter.get_spectrum()
                assert list(spectrum.counts_x) == ax
                assert list(spectrum.counts_z) == az
                assert spectrum.max_weight == n
                minimum = next(w for w in range(n + 1) if ax[w] or az[w])
                for method in ("auto", "dual", "brouwer_zimmermann", "brute_force", "collision"):
                    summary = counter.get_minimum(method=method)
                    assert summary == we.CSSMinimumLogicalWeights(
                        minimum, ax[minimum], az[minimum], full
                    )
                    assert counter.get_minimum(method=method, full_pauli=False).count_pauli is None
                    assert summary.count_homogeneous == ax[minimum] + az[minimum]


def test_known_codes_and_strict_bz_boundary() -> None:
    """The unseen bound equals two after layer one, which misses one of six minimum words."""
    counter = we.CSSLogicalWeightCounter([[1, 1, 1, 1]], [[1, 1, 1, 1]], batch_size=1)
    assert counter.get_minimum(method="brouwer_zimmermann") == we.CSSMinimumLogicalWeights(
        2, 6, 6, 18
    )
    steane = codes.SteaneCode()
    assert codes.get_minimum_logical_weights_css(steane.matrix_x, steane.matrix_z) == (
        we.CSSMinimumLogicalWeights(3, 7, 7, 21)
    )
    assert we.CSSLogicalWeightCounter(steane.matrix_x, steane.matrix_z).get_minimum(
        method="brouwer_zimmermann"
    ) == we.CSSMinimumLogicalWeights(3, 7, 7, 21)
    counter = we.CSSLogicalWeightCounter([[1, 1, 0]], [[0, 0, 1]])
    assert counter.get_minimum() == we.CSSMinimumLogicalWeights(1, 2, 0, 2)


def test_empty_codes_redundancy_and_big_integers() -> None:
    """Rank normalization and exact coefficients beyond uint64."""
    empty = np.empty((0, 100), dtype=np.uint8)
    counter = we.CSSLogicalWeightCounter(empty, empty)
    spectrum = counter.get_spectrum(50)
    assert spectrum.counts_x[50] == spectrum.counts_z[50] == math.comb(100, 50) > 2**64
    assert counter.get_minimum() == we.CSSMinimumLogicalWeights(1, 100, 100, 300)
    assert counter.get_spectrum(0).counts_x == (0,)
    large = we.CSSLogicalWeightCounter(np.empty((0, 1000)), np.empty((0, 1000)))
    assert large.get_spectrum(1).counts_x == (0, 1000)
    assert we._safe_coefficients(1000, 0).shape[0] < 10
    trivial = we.CSSLogicalWeightCounter(np.eye(4, dtype=np.uint8), np.empty((0, 4)))
    assert trivial.get_minimum() == we.CSSMinimumLogicalWeights(None, 0, 0, 0)
    assert trivial.get_minimum(full_pauli=False).count_pauli is None
    assert not any(trivial.get_spectrum().counts_x)
    large_trivial = we.CSSLogicalWeightCounter(np.eye(65, dtype=int), np.empty((0, 65)))
    assert large_trivial.get_spectrum().counts_x == (0,) * 66
    code = codes.SteaneCode()
    redundant = np.vstack([code.matrix_x, code.matrix_x, np.zeros((1, 7), dtype=int)])
    assert we.CSSLogicalWeightCounter(redundant, redundant).get_minimum().count_pauli == 21
    histogram = np.zeros(81, dtype=np.int64)
    histogram[80] = 1
    counter = we.CSSLogicalWeightCounter(np.empty((0, 80)), np.empty((0, 80)))
    with mock.patch.object(we, "_shortened_histogram", return_value=histogram):
        support = np.array([[1, 0]], dtype=np.uint64)
        assert counter._lift(support, np.empty((0, 2), dtype=np.uint64), 80) == 2**80


@pytest.mark.parametrize("width", [63, 64, 65, 129])
def test_multiword_vectors_and_labels(width: int) -> None:
    """Physical words and quotient signatures independently cross uint64 boundaries."""
    empty = np.empty((0, width), dtype=np.uint8)
    counter = we.CSSLogicalWeightCounter(empty, empty)
    assert counter.get_minimum(method="brouwer_zimmermann") == we.CSSMinimumLogicalWeights(
        1, width, width, 3 * width
    )
    checks = np.zeros((1, width), dtype=np.uint8)
    checks[0, :2] = 1
    counter = we.CSSLogicalWeightCounter(checks, empty)
    assert counter.get_minimum(method="brouwer_zimmermann") == we.CSSMinimumLogicalWeights(
        1, width, width - 2, 3 * width - 4
    )
    changed = counter.conjugated_cnot(0, width - 1)
    assert changed.get_minimum(method="brouwer_zimmermann") == we.CSSMinimumLogicalWeights(
        1, width, width - 3, 3 * width - 6
    )
    assert np.array_equal(changed.conjugated_cnot(0, width - 1).checks_x, checks)
    assert changed.get_minimum(method="collision") == changed.get_minimum(
        method="brouwer_zimmermann"
    )
    if width == 129:
        counter = we.CSSLogicalWeightCounter(
            np.pad(np.eye(65, dtype=np.uint8), ((0, 0), (0, 64))),
            np.empty((0, width), dtype=np.uint8),
        )
        assert counter.get_minimum(method="collision") == we.CSSMinimumLogicalWeights(
            1, 64, 64, 192
        )


def test_collision_sector_maps_and_boundaries() -> None:
    """Dependent checks, unequal distances and dressing are independent of family promises."""
    examples = (
        (np.empty((0, 5)), np.empty((0, 5))),
        ([[1, 1, 0]], [[0, 0, 1]]),
        ([[1, 1, 1, 1]], [[1, 1, 1, 1]]),
        (np.eye(4), np.empty((0, 4))),
        ([[1, 1, 0, 0], [1, 1, 0, 0]], [[0, 0, 1, 1], [0, 0, 0, 0]]),
    )
    for x, z in examples:
        counter = we.CSSLogicalWeightCounter(x, z, batch_size=1)
        assert counter.get_minimum(method="collision") == counter.get_minimum(method="dual")
        assert counter.get_minimum(method="collision", full_pauli=False).count_pauli is None
    code = codes.SteaneCode()
    assert codes.get_minimum_logical_weights_css(
        code.matrix_x, code.matrix_z, method="collision"
    ) == we.CSSMinimumLogicalWeights(3, 7, 7, 21)
    counter = we.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z, memory_limit_bytes=100)
    with pytest.raises(MemoryError, match="basis"):
        counter.get_minimum(method="collision")
    counter = we.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z, memory_limit_bytes=1000)
    with pytest.raises(MemoryError, match="map setup"):
        counter.get_minimum(method="collision")
    counter.memory_limit_bytes = 1 << 20
    with (
        mock.patch.object(we, "recover_supports", return_value=np.empty((0, 1), dtype=np.uint64)),
        pytest.raises(ArithmeticError, match="no nontrivial"),
    ):
        counter.get_minimum(method="collision")


def test_collision_connected_dense_eighty_qubits() -> None:
    """Exhaust physical supports independently; do not compare family-supplied distances."""
    n, k = 80, 32
    rng = np.random.default_rng(47)
    x = np.zeros((24, n), dtype=np.uint8)
    z = x.copy()
    x[:, k : k + 24] = np.eye(24, dtype=np.uint8)
    z[:, k + 24 :] = np.eye(24, dtype=np.uint8)
    for _ in range(30 * n):
        control, target = rng.choice(n, 2, replace=False)
        x[:, target] ^= x[:, control]
        z[:, control] ^= z[:, target]
    assert np.linalg.matrix_rank(galois.GF(2)(x)) == 24
    assert np.linalg.matrix_rank(galois.GF(2)(z)) == 24
    assert not np.any((x @ z.T) % 2)
    reached = {0}
    for _ in range(n):
        for row in np.vstack([x, z]):
            connected_positions = set(map(int, np.flatnonzero(row)))
            if connected_positions & reached:
                reached |= connected_positions
    assert len(reached) == n and np.mean(np.vstack([x, z])) > 0.4
    counts = []
    for stabilizers, checks in ((x, z), (z, x)):
        pivots: dict[int, int] = {}
        for row in stabilizers:
            word = sum(int(bit) << jj for jj, bit in enumerate(row))
            while word:
                pivot = word.bit_length() - 1
                if pivot not in pivots:
                    pivots[pivot] = word
                    break
                word ^= pivots[pivot]
        columns = [sum(int(bit) << jj for jj, bit in enumerate(col)) for col in checks.T]
        found = set()
        for weight in range(1, 5):
            for positions in itertools.combinations(range(n), weight):
                syndrome = functools.reduce(int.__xor__, (columns[jj] for jj in positions))
                if syndrome:
                    continue
                word = sum(1 << jj for jj in positions)
                reduced = word
                for pivot in sorted(pivots, reverse=True):
                    if reduced >> pivot & 1:
                        reduced ^= pivots[pivot]
                if reduced:
                    assert weight == 4
                    found.add(word)
        counts.append(len(found))
    assert counts == [1, 0]
    counter = we.CSSLogicalWeightCounter(x, z)
    expected = we.CSSMinimumLogicalWeights(4, 1, 0, 1)
    assert counter.get_minimum(method="collision") == expected
    assert counter.get_minimum(method="auto") == expected


def test_conjugation_ownership_and_matrix_validation() -> None:
    """Conjugation is transactional and preserves commutation with ordered targets."""
    code = codes.SteaneCode()
    counter = we.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z)
    before = counter.checks_x
    changed = counter.conjugated_cnot(1, 6)
    assert np.array_equal(counter.checks_x, before)
    expected = before.copy()
    expected[:, 6] ^= expected[:, 1]
    assert np.array_equal(changed.checks_x, expected)
    assert counter.is_self_dual
    assert not changed.is_self_dual
    copy = counter.checks_x
    copy[:] = 0
    assert np.any(counter.checks_x)
    for targets in ([0, 0], [-1, 2], [0, 7], [0.5, 2], []):
        invalid_targets: Any = targets
        with pytest.raises(ValueError, match="targets"):
            counter.conjugated(invalid_targets, np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="four"):
        counter.conjugated_swel([0, 1])
    with pytest.raises(ValueError, match="square"):
        counter.conjugated([0, 1], [[1]], [[1]])
    with pytest.raises(ValueError, match="pairing"):
        counter.conjugated([0, 1], np.zeros((2, 2)), np.eye(2))


def test_invalid_inputs_and_resource_errors() -> None:
    for checks in ([[2, 0]], [[np.nan, 0]], galois.GF(3)([[1, 0]])):
        with pytest.raises(ValueError, match="binary"):
            we.CSSLogicalWeightCounter(checks, [[0, 0]])
    with pytest.raises(ValueError, match="commute"):
        we.CSSLogicalWeightCounter([[1, 0]], [[1, 0]])
    for kwargs in ({"memory_limit_bytes": 0}, {"batch_size": 0}):
        with pytest.raises(ValueError, match="positive"):
            we.CSSLogicalWeightCounter([[0, 0]], [[0, 0]], **kwargs)
    counter = we.CSSLogicalWeightCounter([[0, 0]], [[0, 0]])
    invalid_method: Any = "unknown"
    with pytest.raises(ValueError, match="Unknown"):
        counter.get_minimum(method=invalid_method)
    for weight in (-1, 3):
        with pytest.raises(ValueError, match="max_weight"):
            counter.get_spectrum(weight)
    counter = we.CSSLogicalWeightCounter([[1, 1, 0]], [[0, 0, 1]], memory_limit_bytes=1)
    with pytest.raises(MemoryError, match="setup"):
        counter.get_minimum()
    wide = we.CSSLogicalWeightCounter(np.eye(64, 65, dtype=np.uint8), np.empty((0, 65)))
    with pytest.raises(OverflowError, match="62"):
        wide.get_minimum(method="dual")
    with pytest.raises(OverflowError, match="62"):
        wide.get_minimum(method="brute_force")
    assert wide.get_minimum() == we.CSSMinimumLogicalWeights(1, 1, 1, 3)


def test_multiword_deduplication_with_frozen_qubits() -> None:
    """Tensoring with fixed |0> ancillas preserves all Steane minimum representatives."""
    code = codes.SteaneCode()
    x = np.zeros((3, 130), dtype=np.uint8)
    z = np.zeros((126, 130), dtype=np.uint8)
    x[:, :7], z[:3, :7] = code.matrix_x, code.matrix_z
    z[3:, 7:] = np.eye(123, dtype=np.uint8)
    counter = we.CSSLogicalWeightCounter(x, z)
    assert counter.get_minimum(method="brouwer_zimmermann") == we.CSSMinimumLogicalWeights(
        3, 7, 7, 21
    )


def test_cached_bases_and_sparse_auto_dispatch() -> None:
    """Transport cached quotient bases without reusing a stale score or weight histogram."""
    code = codes.CSSCode.stack([codes.SteaneCode()] * 7)
    counter = we.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z)
    assert counter._choose_method() == "brouwer_zimmermann"
    assert counter.get_minimum() == we.CSSMinimumLogicalWeights(3, 49, 49, 147)
    changed = counter.conjugated_cnot(0, 7)
    assert changed._basis_cache is not None
    fresh = we.CSSLogicalWeightCounter(changed.checks_x, changed.checks_z)
    assert changed.get_minimum() == fresh.get_minimum()
    assert counter.get_minimum().count_pauli == 147
    constrained = we.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z, memory_limit_bytes=1)
    assert constrained._choose_method() == "dual"
    assert constrained.get_minimum(full_pauli=False).count_homogeneous == 98
    asymmetric = we.CSSLogicalWeightCounter(
        np.hstack([np.eye(24, dtype=np.uint8), np.ones((24, 40), dtype=np.uint8)]),
        np.empty((0, 64), dtype=np.uint8),
    )
    assert asymmetric._choose_method() == "brouwer_zimmermann"
    assert asymmetric.get_minimum() == we.CSSMinimumLogicalWeights(1, 64, 0, 64)
    # Runtime NumPy sizes must not wrap when the Python budget subtraction goes negative.
    numpy_size: Any = np.uint64(1)
    small = we.CSSLogicalWeightCounter(
        [[1, 1, 0]], [[0, 0, 1]], memory_limit_bytes=numpy_size, batch_size=numpy_size
    )
    with pytest.raises(MemoryError, match="setup"):
        small.get_minimum()


def test_exact_transform_guards_and_fallback() -> None:
    assert we._safe_coefficients(100, 40).shape[0] < 50
    for counts, coefficients in ((np.array([1]), (1,)), (np.array([2]), (-1,))):
        with pytest.raises(ArithmeticError, match="MacWilliams"):
            we._dual_coefficient(counts, coefficients, 1)
    code = codes.SteaneCode()
    counter = we.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z)
    with mock.patch.object(we, "_minimum_native", return_value=(-1, 0, 0)):
        assert counter.get_minimum().count_pauli == 21
    with (
        mock.patch.object(we, "_minimum_native", return_value=(-1, 0, 0)),
        mock.patch.object(we, "_dual_coefficient", return_value=0),
        pytest.raises(ArithmeticError),
    ):
        counter.get_minimum(full_pauli=False)


def test_explicit_arithmetic_and_memory_failures() -> None:
    code = codes.SteaneCode()
    counter = we.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z)
    with (
        mock.patch.object(we, "_dual_coefficient", return_value=0),
        pytest.raises(ArithmeticError, match="Negative"),
    ):
        counter.get_spectrum()
    with (
        mock.patch.object(we, "_minimum_native", return_value=(-1, 0, 0)),
        mock.patch.object(counter, "_histograms", return_value=(np.zeros(8), np.zeros(8))),
        pytest.raises(ArithmeticError, match="positive-dimensional"),
    ):
        counter.get_minimum(full_pauli=False)
    for exhaustive in (False, True):
        small = we.CSSLogicalWeightCounter([[1] * 4], [[1] * 4], memory_limit_bytes=1400)
        basis, labels = small._sector_bases()[0]
        with pytest.raises(MemoryError, match="output"):
            small._supports(basis, labels, 2, exhaustive=exhaustive)
    with pytest.raises(MemoryError, match="output"):
        small.get_minimum(method="brute_force")
    tiny = we.CSSLogicalWeightCounter(np.zeros((0, 1)), np.zeros((0, 1)), memory_limit_bytes=1)
    with pytest.raises(MemoryError, match="setup"):
        tiny._support_buffer(0)
    with pytest.raises(MemoryError, match="union"):
        tiny._lift(np.array([[1]], dtype=np.uint64), np.empty((0, 1), dtype=np.uint64), 1)
    wide = we.CSSLogicalWeightCounter(np.zeros((0, 63)), np.zeros((0, 63)))
    with pytest.raises(OverflowError, match="62"):
        wide._supports(*wide._sector_bases()[0], 1, exhaustive=True)


@pytest.mark.parametrize("interpreted", [False, True])
def test_rank_word_integer_oracle(interpreted: bool) -> None:
    """Exhaust restricted three-row spaces, reusing poisoned scratch across every query."""
    rank = getattr(we._rank_word, "py_func", we._rank_word) if interpreted else we._rank_word
    scratch = np.full(64, 2**64 - 1, dtype=np.uint64)
    coordinates = (0, 31, 63)
    words = [
        sum(((bits >> jj) & 1) << pos for jj, pos in enumerate(coordinates)) for bits in range(8)
    ]
    for triple in itertools.product(words, repeat=3):
        backing = np.zeros(6, dtype=np.uint64)
        backing[::2] = triple
        rows = backing[::2]
        for mask in words:
            span = {0}
            for value in triple:
                span |= {old ^ (value & mask) for old in span}
            assert rank(rows, np.uint64(mask), scratch) == len(span).bit_length() - 1
        np.testing.assert_array_equal(rows, triple)
    assert rank(np.empty(0, dtype=np.uint64), np.uint64(2**64 - 1), scratch) == 0


@pytest.mark.parametrize("interpreted", [False, True])
def test_shortened_word_span_oracle(interpreted: bool) -> None:
    """Count contained words independently, including bit 63 and a reused rank buffer."""
    shortened = (
        getattr(we._shortened_histogram_one, "py_func", we._shortened_histogram_one)
        if interpreted
        else we._shortened_histogram_one
    )
    rows = np.array([5, 6, 8, 2**63], dtype=np.uint64)
    span = {0}
    for value in rows:
        span |= {old ^ int(value) for old in span}
    supports = np.array(
        [
            sum(((bits >> jj) & 1) << pos for jj, pos in enumerate((0, 1, 2, 3, 63)))
            for bits in range(32)
        ],
        dtype=np.uint64,
    )
    expected = np.zeros(65, dtype=np.int64)
    for support in supports:
        contained = sum(value & ~int(support) == 0 for value in span)
        expected[contained.bit_length() - 1] += 1
    np.testing.assert_array_equal(shortened(rows, supports, 64), expected)


@pytest.mark.parametrize("interpreted", [False, True])
def test_native_corner_cases(interpreted: bool) -> None:
    minimum = (
        getattr(we._minimum_native, "py_func", we._minimum_native)
        if interpreted
        else we._minimum_native
    )
    with pytest.raises(ArithmeticError, match="Nonintegral"):
        minimum(np.array([1]), np.array([1]), 1, 1, np.array([[1]]))
    with pytest.raises(ArithmeticError, match="Negative"):
        minimum(np.array([2]), np.array([0]), 0, 0, np.array([[1]]))
    assert minimum(np.array([0]), np.array([0]), 0, 0, np.array([[1]])) == (-1, 0, 0)
    histogram = (
        getattr(we._histogram_chunk, "py_func", we._histogram_chunk)
        if interpreted
        else we._histogram_chunk
    )
    rows = np.array([[0, 1], [1, 1]], dtype=np.uint64)
    counts = np.zeros(129, dtype=np.int64)
    counts[0] = 1
    histogram(rows, np.zeros(2, dtype=np.uint64), counts, 1, 4)
    assert counts[:3].tolist() == [1, 2, 1]
    exhaustive = (
        getattr(we._exhaustive_supports, "py_func", we._exhaustive_supports)
        if interpreted
        else we._exhaustive_supports
    )
    minimum_exhaustive = (
        getattr(we._scan_exhaustive_minimum, "py_func", we._scan_exhaustive_minimum)
        if interpreted
        else we._scan_exhaustive_minimum
    )
    matches = np.empty((4, 2), dtype=np.uint64)
    assert exhaustive(rows, np.zeros(2, dtype=np.uint64), 0, 1, 4, 1, matches) == 2
    np.testing.assert_array_equal(matches[:2], [[0, 1], [1, 0]])
    assert minimum_exhaustive(rows, np.zeros(2, dtype=np.uint64), 0, 1, 4, 129, matches) == (1, 2)
    shortened = (
        getattr(we._shortened_histogram, "py_func", we._shortened_histogram)
        if interpreted
        else we._shortened_histogram
    )
    np.testing.assert_array_equal(
        shortened(rows, np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint64), 2),
        [0, 2, 1],
    )
    rank = getattr(we._rank_masked, "py_func", we._rank_masked) if interpreted else we._rank_masked
    assert rank(np.array([[1], [2], [3]], dtype=np.uint64), np.array([3], dtype=np.uint64)) == 2
    assert (
        rank(
            np.array([[0, 1], [0, 2], [0, 3], [1, 0]], dtype=np.uint64),
            np.array([2**64 - 1, 2**64 - 1], dtype=np.uint64),
        )
        == 3
    )
    rank_word = getattr(we._rank_word, "py_func", we._rank_word) if interpreted else we._rank_word
    scratch = np.empty(64, dtype=np.uint64)
    assert rank_word(np.array([], dtype=np.uint64), np.uint64(3), scratch) == 0
    assert rank_word(np.array([1], dtype=np.uint64), np.uint64(0), scratch) == 0
    assert rank_word(np.array([1, 2, 3], dtype=np.uint64), np.uint64(3), scratch) == 2
    scan = (
        getattr(we._scan_supports, "py_func", we._scan_supports)
        if interpreted
        else we._scan_supports
    )
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
    assert words == {3, 5, 6, 9, 10, 12}


@pytest.mark.parametrize("interpreted", [False, True])
def test_scanner_validation_and_specializations(interpreted: bool) -> None:
    scan = (
        getattr(we._scan_supports, "py_func", we._scan_supports)
        if interpreted
        else we._scan_supports
    )
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
                native = getattr(we._scan_supports_one, "py_func", we._scan_supports_one)
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
    scan = (
        getattr(we._scan_supports, "py_func", we._scan_supports)
        if interpreted
        else we._scan_supports
    )
    single = (
        getattr(we._scan_supports_one, "py_func", we._scan_supports_one)
        if interpreted
        else we._scan_supports_one
    )
    with mock.patch.object(we, "_scan_supports_one", single):
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


def test_information_sets_packed_once_and_budgeted() -> None:
    counter = we.CSSLogicalWeightCounter([[1] * 4], [[1] * 4], batch_size=1)
    basis, labels = counter._sector_bases()[0]
    sets = we.bz._get_information_set_generators(basis, labels)
    selected = we.bz._select_brouwer_zimmermann_information_sets(
        sets, dimension=len(basis), upper_bound=3, weight_divisor=2
    )
    with mock.patch.object(we, "_pack", wraps=we._pack) as pack:
        words = counter._supports(basis, labels, 2, exhaustive=False)
        assert set(map(int, words[:, 0])) == {3, 5, 6, 9, 10, 12}
        assert pack.call_count == 2 * len(selected)
    counter.memory_limit_bytes = 1
    with (
        mock.patch.object(we, "_pack", wraps=we._pack) as pack,
        pytest.raises(MemoryError, match="setup"),
    ):
        counter._supports(basis, labels, 2, exhaustive=False)
    pack.assert_not_called()


@pytest.mark.parametrize("width", [15, 65])
def test_overlapping_supports_keep_strict_coverage_and_physical_ownership(width: int) -> None:
    """Compare overlapping, labeled bases with an independent physical weight-three oracle."""
    checks = np.asarray(codes.HammingCode(4).matrix, dtype=np.uint8)
    x, z = np.zeros((4, width), dtype=np.uint8), np.zeros((width - 11, width), dtype=np.uint8)
    x[:, :15], z[:4, :15] = checks, checks
    z[4:, 15:] = np.eye(width - 15, dtype=np.uint8)
    counter = we.CSSLogicalWeightCounter(x, z, batch_size=7)
    basis, labels = counter._sector_bases()[0]
    orders = (
        np.arange(width),
        np.arange(width)[::-1],
        np.random.default_rng(42).permutation(width),
    )
    family = []
    for order in orders:
        rows, tags, pivots = we.binary._row_reduce_binary(basis, labels, columns=order)
        family.append(we.bz._InformationSet(rows, tags, len(pivots), pivots))
    coverage = int(np.bincount(np.concatenate([item.pivots for item in family])).max())
    expected = {
        sum(1 << jj for jj in support)
        for support in itertools.combinations(range(15), 3)
        if not np.any(checks[:, support].sum(axis=1) % 2)
    }
    assert len(expected) == 35 and coverage > 1
    with (
        mock.patch.object(
            we.bz, "_get_overlapping_information_sets", return_value=(family, coverage)
        ) as select,
        mock.patch.object(
            we.bz,
            "_get_brouwer_zimmermann_lower_bound",
            wraps=we.bz._get_brouwer_zimmermann_lower_bound,
        ) as bound,
    ):
        words = counter._supports(basis, labels, 3, exhaustive=False)
    actual = [sum(int(word) << (64 * jj) for jj, word in enumerate(row)) for row in words]
    assert set(actual) == expected and len(actual) == len(expected)
    assert select.call_args.kwargs["upper_bound"] == 4
    counting_calls = [call for call in bound.call_args_list if "max_coverage" in call.kwargs]
    assert counting_calls and all(
        call.kwargs["max_coverage"] == coverage for call in counting_calls
    )
    assert bound.call_args.kwargs["completed_weight"] >= 3
    counter.memory_limit_bytes = counter._basis_storage() + 1024
    with (
        mock.patch.object(we.bz, "_get_overlapping_information_sets") as select,
        pytest.raises(MemoryError, match="setup"),
    ):
        counter._supports(basis, labels, 3, exhaustive=False)
    select.assert_not_called()


def test_connected_multiword_surface() -> None:
    """Check rank, connectivity and full counts independently of family parameter caches."""
    code = codes.SurfaceCode(9)
    x, z = np.asarray(code.matrix_x), np.asarray(code.matrix_z)
    assert not np.any((x @ z.T) % 2)
    assert (
        len(code) - np.linalg.matrix_rank(galois.GF2(x)) - np.linalg.matrix_rank(galois.GF2(z)) == 1
    )
    neighbors = (np.vstack([x, z]).T @ np.vstack([x, z])) != 0
    reached = {0}
    while True:
        extended = reached | set(map(int, np.flatnonzero(np.any(neighbors[list(reached)], axis=0))))
        if extended == reached:
            break
        reached = extended
    assert len(reached) == 81
    counter = we.CSSLogicalWeightCounter(x, z)
    assert counter.get_minimum() == we.CSSMinimumLogicalWeights(9, 1556, 1556, 9 * 2**9)
    changed = counter.conjugated_cnot(0, 1)
    assert changed.get_minimum() == we.CSSMinimumLogicalWeights(8, 70, 0, 2**7)
