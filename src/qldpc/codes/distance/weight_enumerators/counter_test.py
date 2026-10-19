# SPDX-License-Identifier: Apache-2.0

"""Independent invariant tests for exact binary CSS weight enumeration.

Backend-specific behavior is tested beside each backend; these tests cover the counter's
dispatch, conjugation, resource policy and agreement with explicit oracles.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import math
import tracemalloc
from typing import Any
from unittest import mock

import galois
import numpy as np
import pytest

from qldpc import codes

from . import counter as we
from . import dual, supports
from .conftest import _native


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
    # Interpreted mode runs the Python implementation of every native kernel the counter uses.
    kernels = {
        dual: ("_histogram_chunk", "_minimum_native"),
        supports: (
            "_scan_supports",
            "_scan_supports_one",
            "_exhaustive_supports",
            "_scan_exhaustive_minimum",
            "_rank_masked",
            "_rank_word",
            "_shortened_histogram",
            "_shortened_histogram_one",
        ),
        we: ("transform_rows",),
    }
    with contextlib.ExitStack() as stack:
        if interpreted:
            for module, names in kernels.items():
                stack.enter_context(
                    mock.patch.multiple(
                        module, **{name: _native(getattr(module, name), True) for name in names}
                    )
                )
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
    trivial = we.CSSLogicalWeightCounter(np.eye(4, dtype=np.uint8), np.empty((0, 4)))
    assert trivial.get_minimum() == we.CSSMinimumLogicalWeights(None, 0, 0, 0)
    assert trivial.get_minimum(full_pauli=False).count_pauli is None
    assert not any(trivial.get_spectrum().counts_x)
    large_trivial = we.CSSLogicalWeightCounter(np.eye(65, dtype=int), np.empty((0, 65)))
    assert large_trivial.get_spectrum().counts_x == (0,) * 66
    code = codes.SteaneCode()
    redundant = np.vstack([code.matrix_x, code.matrix_x, np.zeros((1, 7), dtype=int)])
    assert we.CSSLogicalWeightCounter(redundant, redundant).get_minimum().count_pauli == 21


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


def test_dual_fallback_and_arithmetic_failures() -> None:
    code = codes.SteaneCode()
    counter = we.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z)
    with mock.patch.object(dual, "_minimum_native", return_value=(-1, 0, 0)):
        assert counter.get_minimum().count_pauli == 21
    with (
        mock.patch.object(dual, "_dual_coefficient", return_value=0),
        pytest.raises(ArithmeticError, match="Negative"),
    ):
        counter.get_spectrum()
    with (
        mock.patch.object(dual, "_minimum_native", return_value=(-1, 0, 0)),
        mock.patch.object(counter, "_get_histograms", return_value=(np.zeros(8), np.zeros(8))),
        pytest.raises(ArithmeticError, match="positive-dimensional"),
    ):
        counter.get_minimum(full_pauli=False)


def test_explicit_memory_failures() -> None:
    small = we.CSSLogicalWeightCounter([[1] * 4], [[1] * 4], memory_limit_bytes=1400)
    with pytest.raises(MemoryError, match="output"):
        small.get_minimum(method="brute_force")
    with pytest.raises(MemoryError, match="output"):
        small.get_minimum(method="brouwer_zimmermann")
    wide = we.CSSLogicalWeightCounter(np.zeros((0, 63)), np.zeros((0, 63)))
    with pytest.raises(OverflowError, match="62"):
        wide.get_minimum(method="brute_force")


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


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("width", [5, 130])
def test_transform_rows_against_bit_oracle(interpreted: bool, width: int) -> None:
    """Targets may straddle word boundaries; untouched coordinates must be preserved."""
    transform = _native(we.transform_rows, interpreted)
    rng = np.random.default_rng(width)
    matrix = rng.integers(0, 2, size=(4, width), dtype=np.uint8)
    targets = [0, width - 1, width // 2] if width > 64 else [0, 4]
    local = rng.integers(0, 2, size=(len(targets), len(targets)), dtype=np.uint8)
    expected = matrix.copy()
    expected[:, targets] = (matrix[:, targets] @ local.T) % 2
    packed = supports.pack(matrix)
    before = packed.copy()
    result = transform(
        packed, np.asarray(targets, dtype=np.int64), supports.pack(local)[:, 0].astype(np.uint64)
    )
    assert np.array_equal(supports.unpack(result, width), expected)
    assert np.array_equal(packed, before)


def _measure_peak(call: Any) -> tuple[bool, int]:
    """Whether ``call`` fit its budget, and the peak bytes it allocated (NumPy and Python)."""
    tracemalloc.start()
    try:
        call()
        fits = True
    except MemoryError:
        fits = False
    finally:
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return fits, peak


@pytest.mark.parametrize("num_qubits", [40, 70])
def test_storage_budget_bounds_measured_allocations(num_qubits: int) -> None:
    """Whenever a budgeted phase succeeds, its measured peak allocation fits within the limit.

    All-ones checks give the minimum-weight supports of every pair of qubits, so the output is
    large enough to exhaust a small budget.  The budget covers support enumeration and the
    collision tables, not the distance search that precedes BZ enumeration, so the BZ route is
    measured through its support collector alone.
    """
    checks = np.ones((1, num_qubits), dtype=np.uint8)
    reference = we.CSSLogicalWeightCounter(checks, checks)
    (basis, labels), _ = reference._get_sector_bases()
    basis_bytes = reference._get_basis_storage()

    def collect_bz(limit: int) -> Any:
        budget = supports.StorageBudget(limit, basis_bytes, 1 << 12)
        return lambda: supports.get_bz_supports(basis, labels, 2, budget)

    def count_by_collision(limit: int) -> Any:
        counter = we.CSSLogicalWeightCounter(
            checks, checks, memory_limit_bytes=limit, batch_size=1 << 12
        )
        return lambda: counter.get_minimum(method="collision")

    for make_call in (collect_bz, count_by_collision):
        # Exclude compilation and one-time setup from the measurement.
        make_call(1 << 30)()
        outcomes = {}
        for limit in (1 << power for power in range(16, 24)):
            fits, peak = _measure_peak(make_call(limit))
            outcomes[limit] = fits
            if fits:
                assert peak <= limit, (make_call.__name__, limit, peak)
        # The sweep must cross the threshold, or it would not test the budget at all.
        assert not outcomes[1 << 16] and outcomes[1 << 23], (make_call.__name__, outcomes)


def _random_css_code(num_qubits: int, num_logicals: int, seed: int) -> tuple[Any, Any]:
    """A random CSS stabilizer code: a trivial code conjugated by a random invertible matrix."""
    rng = np.random.default_rng(seed)
    rank_x = (num_qubits - num_logicals) // 2
    rank_z = num_qubits - num_logicals - rank_x
    checks_x = np.zeros((rank_x, num_qubits), dtype=np.uint8)
    checks_z = np.zeros((rank_z, num_qubits), dtype=np.uint8)
    for row in range(rank_x):
        checks_x[row, num_logicals + row] = 1
    for row in range(rank_z):
        checks_z[row, num_logicals + rank_x + row] = 1
    while True:
        matrix = rng.integers(0, 2, size=(num_qubits, num_qubits), dtype=np.uint8)
        if np.linalg.matrix_rank(galois.GF2(matrix)) == num_qubits:
            break
    inverse = np.array(np.linalg.inv(galois.GF2(matrix)), dtype=np.uint8)
    return (checks_x @ matrix.T) % 2, (checks_z @ inverse) % 2


@pytest.mark.parametrize(
    "num_qubits,num_logicals,seed",
    [(8, 2, 1), (9, 1, 2), (10, 3, 3), (11, 2, 4), (12, 4, 5), (12, 1, 6)],
)
def test_all_methods_match_explicit_enumeration_on_random_codes(
    num_qubits: int, num_logicals: int, seed: int
) -> None:
    """Five independent algorithms must agree with each other and with explicit enumeration.

    Random codes have unequal sector ranks and unequal X and Z minimum weights, unlike the
    structured examples elsewhere.  Small batches force the chunked native paths.
    """
    checks_x, checks_z = _random_css_code(num_qubits, num_logicals, seed)
    counter = we.CSSLogicalWeightCounter(checks_x, checks_z, batch_size=7)
    assert counter.dimension == num_logicals
    counts_x, counts_z, full = _oracle(counter)
    minimum = next(
        weight for weight in range(num_qubits + 1) if counts_x[weight] or counts_z[weight]
    )
    expected = we.CSSMinimumLogicalWeights(minimum, counts_x[minimum], counts_z[minimum], full)
    spectrum = counter.get_spectrum()
    assert list(spectrum.counts_x) == counts_x and list(spectrum.counts_z) == counts_z
    for method in ("auto", "dual", "brouwer_zimmermann", "brute_force", "collision"):
        assert counter.get_minimum(method=method) == expected, method


def _sparse_css_counter(
    num_blocks: int, num_frozen: int, seed: int, num_gates: int
) -> we.CSSLogicalWeightCounter:
    """Steane blocks and frozen qubits, scrambled by a few CNOTs.

    Unlike dense random codes, these have low-weight stabilizers that fit inside minimum-weight
    logical supports, which is what makes the opposite-type dressing term of full-Pauli counts
    nonzero.
    """
    code = codes.CSSCode.stack([codes.SteaneCode()] * num_blocks)
    num_qubits = len(code) + num_frozen
    checks_x = np.zeros((len(code.matrix_x), num_qubits), dtype=np.uint8)
    checks_z = np.zeros((len(code.matrix_z) + num_frozen, num_qubits), dtype=np.uint8)
    checks_x[:, : len(code)] = code.matrix_x
    checks_z[: len(code.matrix_z), : len(code)] = code.matrix_z
    checks_z[len(code.matrix_z) :, len(code) :] = np.eye(num_frozen, dtype=np.uint8)
    counter = we.CSSLogicalWeightCounter(checks_x, checks_z, batch_size=5)
    rng = np.random.default_rng(seed)
    for _ in range(num_gates):
        control, target = map(int, rng.choice(num_qubits, 2, replace=False))
        counter = counter.conjugated_cnot(control, target)
    return counter


@pytest.mark.parametrize(
    "num_blocks,num_frozen,seed,num_gates",
    [(1, 2, 1, 4), (1, 3, 2, 8), (1, 1, 3, 6), (2, 0, 4, 5), (2, 0, 5, 12)],
)
def test_all_methods_match_explicit_enumeration_on_sparse_codes(
    num_blocks: int, num_frozen: int, seed: int, num_gates: int
) -> None:
    """Sparse, partially scrambled codes reach the full-Pauli dressing term and ties."""
    counter = _sparse_css_counter(num_blocks, num_frozen, seed, num_gates)
    counts_x, counts_z, full = _oracle(counter)
    minimum = next(
        weight for weight in range(counter.num_qubits + 1) if counts_x[weight] or counts_z[weight]
    )
    expected = we.CSSMinimumLogicalWeights(minimum, counts_x[minimum], counts_z[minimum], full)
    for method in ("auto", "dual", "brouwer_zimmermann", "brute_force", "collision"):
        assert counter.get_minimum(method=method) == expected, method
    # Full Paulis can exceed the X-only and Z-only counts only through mixed representatives.
    assert expected.count_pauli is not None and expected.count_pauli >= expected.count_x


def _collect_scan(scan: Any, rows: Any, tags: Any, weight: int, target: int, batch: int) -> Any:
    """All matches of a resumable coefficient scan, run to completion."""
    support = np.zeros(weight, dtype=np.int64)
    prefixes = np.zeros((weight + 1, rows.shape[1]), dtype=np.uint64)
    tag_prefixes = np.zeros((weight + 1, tags.shape[1]), dtype=np.uint64)
    output = np.empty((batch, rows.shape[1]), dtype=np.uint64)
    found: list[tuple[int, ...]] = []
    depth = 0
    while depth >= 0:
        depth, count = scan(
            rows, tags, support, prefixes, tag_prefixes, depth, target, batch, output
        )
        found.extend(map(tuple, output[:count].tolist()))
    return found


@pytest.mark.parametrize("interpreted", [False, True])
def test_single_word_kernels_match_padded_multiword_kernels(interpreted: bool) -> None:
    """Each single-word fast path must agree with its general path on identical data.

    Appending a zero word changes the layout, and so the code path, but not any weight.
    """
    rng = np.random.default_rng(5150)
    rows = rng.integers(0, np.iinfo(np.uint64).max, size=(9, 1), dtype=np.uint64)
    padded = np.hstack([rows, np.zeros_like(rows)])
    tags = rng.integers(0, 4, size=(9, 1), dtype=np.uint64)
    tags[:3] = 0

    # Gray-code histogram of a row space.
    histogram = _native(dual._histogram_chunk, interpreted)
    counts = [np.zeros(129, dtype=np.int64) for _ in range(2)]
    for counted, matrix in zip(counts, (rows, padded)):
        counted[0] = 1
        histogram(matrix, np.zeros(matrix.shape[1], dtype=np.uint64), counted, 1, 1 << 9)
    np.testing.assert_array_equal(counts[0], counts[1])

    # Exhaustive supports and minimum, with a stabilizer subspace spanned by the first rows.
    # Identity rows give many ties at the minimum, and the target equals the weight of a
    # stabilizer-only combination, so mishandling either is visible.
    exhaustive = _native(supports._exhaustive_supports, interpreted)
    minimum = _native(supports._scan_exhaustive_minimum, interpreted)
    sparse = np.array([[1 << bit] for bit in range(6)], dtype=np.uint64)
    for data, target in ((rows, int(rows[0, 0]).bit_count()), (sparse, 2)):
        for stabilizer_rank in (0, 3):
            results = []
            padded_data = np.hstack([data, np.zeros_like(data)])
            for matrix in (data, padded_data):
                width = matrix.shape[1]
                output = np.empty((1 << len(data), width), dtype=np.uint64)
                current = np.zeros(width, dtype=np.uint64)
                count = exhaustive(
                    matrix, current, stabilizer_rank, 1, 1 << len(data), target, output
                )
                words = output[:count, 0].tolist()
                best, tied = minimum(
                    matrix,
                    np.zeros(width, dtype=np.uint64),
                    stabilizer_rank,
                    1,
                    1 << len(data),
                    65,
                    output,
                )
                results.append((words, best, tied, output[:tied, 0].tolist()))
            assert results[0] == results[1]
            assert results[0][0] and results[0][2]

    # Coefficient-layer scan over sibling leaves, at several batch sizes.
    scan = _native(supports._scan_supports, interpreted)
    single = _native(supports._scan_supports_one, interpreted)
    with mock.patch.object(supports, "_scan_supports_one", single):
        matched = 0
        for weight, target, batch in ((2, 32, 1), (3, 32, 4), (4, 31, 100)):
            narrow = _collect_scan(scan, rows, tags, weight, target, batch)
            wide = _collect_scan(scan, padded, tags, weight, target, batch)
            assert narrow == [(word,) for word, _ in wide]
            assert all(second == 0 for _, second in wide)
            matched += len(narrow)
        assert matched

    # Opposite-type stabilizer counts: small stabilizers, so that many masks contain some of them.
    shortened = _native(supports._shortened_histogram, interpreted)
    stabilizers = np.array([[0b0011], [0b0110], [0b1100]], dtype=np.uint64)
    masks = rng.integers(0, 16, size=(40, 1), dtype=np.uint64)
    narrow_histogram = shortened(stabilizers, masks, 4)
    wide_histogram = shortened(
        np.hstack([stabilizers, np.zeros_like(stabilizers)]),
        np.hstack([masks, np.zeros_like(masks)]),
        4,
    )
    np.testing.assert_array_equal(narrow_histogram, wide_histogram)
    assert narrow_histogram.sum() == len(masks) and narrow_histogram[1:].sum() > 0
