# SPDX-License-Identifier: Apache-2.0

"""Tests for exact counts of the logical operators of a given weight in binary CSS codes."""

from __future__ import annotations

import contextlib
import functools
import itertools
from collections.abc import Iterator
from typing import Any
from unittest import mock

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes
from qldpc.objects import Pauli

from . import logical_counts as lc
from ._binary import (
    _get_independent_rows,
    _get_kernel,
    _get_nested_code_basis,
    _pack_bits,
    _row_reduce_binary,
)
from .brouwer_zimmermann import _InformationSet
from .core import DistanceMethod

Counts = lc.CSSLogicalCounts
METHODS: tuple[DistanceMethod, ...] = ("brouwer_zimmermann", "brute_force")
KERNELS = (
    "_count_chunk",
    "_is_owned",
    "_get_masked_rank",
    "_apply_local",
    "_is_css_pairing",
)


@contextlib.contextmanager
def _maybe_interpreted(interpreted: bool) -> Iterator[None]:
    """Run the Python implementations of the native kernels, so that coverage can trace them."""
    if not interpreted:
        yield
        return
    with mock.patch.multiple(lc, **{name: getattr(lc, name).py_func for name in KERNELS}):
        yield


def _counter(checks_x: npt.ArrayLike, checks_z: npt.ArrayLike) -> lc.CSSLogicalCounter:
    """A counter for the CSS code with the given binary checks."""
    code = codes.CSSCode(np.asarray(checks_x, dtype=int), np.asarray(checks_z, dtype=int))
    return lc.CSSLogicalCounter(code)


def _bits(values: list[int], width: int) -> npt.NDArray[np.uint8]:
    """Binary rows whose bit j is bit j of each integer."""
    return np.array([[value >> jj & 1 for jj in range(width)] for value in values], dtype=np.uint8)


def _random_code(
    rng: np.random.Generator, num_qubits: int, rank_x: int, rank_z: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Random commuting checks: Z checks are random combinations of the kernel of the X checks."""
    checks_x = rng.integers(0, 2, (rank_x, num_qubits), dtype=np.uint8)
    kernel = _get_kernel(checks_x)
    checks_z = (rng.integers(0, 2, (rank_z, len(kernel))) @ kernel % 2).astype(np.uint8)
    return checks_x, checks_z


def _random_css_code(
    rng: np.random.Generator, num_qubits: int, rank_x: int, rank_z: int
) -> codes.CSSCode:
    """A random binary CSS stabilizer code, as in _random_code."""
    checks_x, checks_z = _random_code(rng, num_qubits, rank_x, rank_z)
    return codes.CSSCode(checks_x.astype(int), checks_z.astype(int))


def _frozen(
    checks_x: npt.ArrayLike, checks_z: npt.ArrayLike, width: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Append qubits fixed in |0> by a Z check each; no logical of minimum weight touches them."""
    x, z = np.asarray(checks_x, dtype=np.uint8), np.asarray(checks_z, dtype=np.uint8)
    extra = width - x.shape[1]
    padded_x = np.pad(x, ((0, 0), (0, extra)))
    padded_z = np.vstack(
        [np.pad(z, ((0, 0), (0, extra))), np.hstack([np.zeros((extra, z.shape[1])), np.eye(extra)])]
    ).astype(np.uint8)
    return padded_x, padded_z


# X checks {0,1,2,3,4} and {2,4}, and Z checks {1,2,3,4} and {0,1}, on five qubits.  By hand, the
# distance is two, the X-only logicals of weight two are {2,3} and {3,4}, and the Z-only ones are
# {2,4}, {0,3} and {1,3}.  The Z word {2,4} also supports the X stabilizer {2,4}, so in total there
# are 2 + 3 + 1 = 6 logical Pauli operators of weight two.
_HAND_X, _HAND_Z = _bits([31, 20], 5), _bits([30, 3], 5)


####################################################################################################
# results


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_known_codes(method: DistanceMethod, interpreted: bool) -> None:
    steane = codes.SteaneCode()
    examples = (
        (steane.matrix_x, steane.matrix_z, Counts(3, 7, 7, 21)),
        # After layer one the BZ bound equals two, which would miss one of six words of weight two.
        ([[1, 1, 1, 1]], [[1, 1, 1, 1]], Counts(2, 6, 6, 18)),
        (_HAND_X, _HAND_Z, Counts(2, 2, 3, 6)),
        # The Z sector is heavier, so it contributes nothing at the global minimum.
        ([[1, 1, 0]], [[0, 0, 1]], Counts(1, 2, 0, 2)),
        (np.empty((0, 3)), np.empty((0, 3)), Counts(1, 3, 3, 9)),
        (np.eye(4), np.empty((0, 4)), Counts(None, 0, 0, 0)),
    )
    with _maybe_interpreted(interpreted):
        for checks_x, checks_z, expected in examples:
            counter = _counter(checks_x, checks_z)
            assert counter.count_logicals(full_pauli=True, method=method) == expected
            homogeneous = counter.count_logicals(method=method)
            assert homogeneous == Counts(expected.weight, expected.count_x, expected.count_z, None)
    assert lc.count_logicals_css(steane) == Counts(3, 7, 7, None)


def test_surface_codes() -> None:
    # Unrotated surface codes have d minimum-weight logicals of each type: straight lines.
    for distance in (3, 5, 7):
        counter = lc.CSSLogicalCounter(codes.SurfaceCode(distance, rotated=False))
        expected = Counts(distance, distance, distance, 2 * distance)
        assert counter.count_logicals(full_pauli=True) == expected
    for distance in (3, 4):
        code = codes.SurfaceCode(distance)
        counts_x, counts_z, _, full = _enumerate(code)
        expected = Counts(distance, counts_x[distance], counts_z[distance], full)
        assert lc.CSSLogicalCounter(code).count_logicals(full_pauli=True) == expected
    # A regression check on a larger rotated surface code, which the methods above cannot afford.
    counter = lc.CSSLogicalCounter(codes.SurfaceCode(9))
    assert counter.count_logicals(full_pauli=True) == Counts(9, 1556, 1556, 4608)


@pytest.mark.parametrize("width", [5, 63, 64, 65, 129])
def test_frozen_qubits_preserve_counts(width: int) -> None:
    """Frozen qubits change the physical width, not the counts at the distance."""
    # Each frozen qubit adds a Z stabilizer to the normalizer basis, so brute force is unaffordable.
    methods = METHODS if width < 20 else ("brouwer_zimmermann",)
    for method in methods:
        counter = _counter(*_frozen(_HAND_X, _HAND_Z, width))
        assert counter.count_logicals(full_pauli=True, method=method) == Counts(2, 2, 3, 6)
    steane = codes.SteaneCode()
    counter = _counter(*_frozen(steane.matrix_x, steane.matrix_z, width + 2))
    assert counter.count_logicals(full_pauli=True) == Counts(3, 7, 7, 21)


@pytest.mark.parametrize(
    "num_qubits, rank_x, rank_z",
    [(10, 3, 3), (14, 4, 5), (16, 10, 2), (24, 14, 4), (32, 14, 12), (40, 22, 18)],
)
def test_methods_agree_on_random_codes(num_qubits: int, rank_x: int, rank_z: int) -> None:
    """Brute force visits every word once, so it checks BZ's set selection and ownership."""
    rng = np.random.default_rng(num_qubits)
    for _ in range(3):
        counter = _counter(*_random_code(rng, num_qubits, rank_x, rank_z))
        assert counter.count_logicals(
            full_pauli=True, method="brouwer_zimmermann"
        ) == counter.count_logicals(full_pauli=True, method="brute_force")


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_fixed_weights_of_known_codes(method: DistanceMethod, interpreted: bool) -> None:
    """The Steane X normalizer is the [7, 4] Hamming code; its weight-four words are stabilizers."""
    steane = codes.SteaneCode()
    counter = lc.CSSLogicalCounter(steane)
    with _maybe_interpreted(interpreted):
        for weight, count in enumerate([0, 0, 0, 7, 0, 0, 0, 1]):
            expected = Counts(weight, count, count, None)
            assert counter.count_logicals(weight, method=method) == expected
        assert counter.count_logicals(3, method=method, full_pauli=True) == Counts(3, 7, 7, 21)
        for weight in (0, 2):
            assert counter.count_logicals(weight, full_pauli=True) == Counts(weight, 0, 0, 0)
        with pytest.raises(ValueError, match="less than weight 4"):
            counter.count_logicals(4, method=method, full_pauli=True)

        # XYZI is a logical of weight three in the [[4, 2, 2]] code, but no homogeneous logical is
        # odd.  A frozen qubit adds a trivial Z stabilizer of weight one, which is lighter than the
        # distance, and which turns each of the six Z logicals of weight two into one of weight
        # three.
        counter = _counter(*_frozen([[1, 1, 1, 1]], [[1, 1, 1, 1]], 5))
        assert counter.count_logicals(3, method=method) == Counts(3, 0, 6, None)
        assert counter.count_logicals(2, method=method, full_pauli=True) == Counts(2, 6, 6, 18)
        with pytest.raises(ValueError, match="less than weight 3"):
            counter.count_logicals(3, method=method, full_pauli=True)
        trivial = _counter(np.eye(4), np.empty((0, 4)))
        assert trivial.count_logicals(2, method=method, full_pauli=True) == Counts(2, 0, 0, 0)


def test_explicit_weights_skip_the_distance_search() -> None:
    steane = codes.SteaneCode()
    counter = lc.CSSLogicalCounter(steane)
    with mock.patch.object(lc, "_get_distance", side_effect=AssertionError("distance searched")):
        assert counter.count_logicals(3, full_pauli=True) == Counts(3, 7, 7, 21)
        assert counter.count_logicals(7) == Counts(7, 1, 1, None)


@pytest.mark.parametrize("num_qubits, rank_x, rank_z", [(10, 3, 3), (12, 5, 4), (14, 8, 2)])
def test_fixed_weights_partition_the_logical_words(
    num_qubits: int, rank_x: int, rank_z: int
) -> None:
    """Every nontrivial X word has one weight, so the counts sum to |ker(H_Z)| - |row(H_X)|."""
    rng = np.random.default_rng(num_qubits)
    for _ in range(3):
        counter = _counter(*_random_code(rng, num_qubits, rank_x, rank_z))
        results = [counter.count_logicals(weight) for weight in range(num_qubits + 1)]
        for weight, result in enumerate(results):
            assert result == counter.count_logicals(weight, method="brute_force")
        rx, rz = len(counter.checks_x), len(counter.checks_z)
        assert sum(result.count_x for result in results) == 2 ** (num_qubits - rz) - 2**rx
        assert sum(result.count_z for result in results) == 2 ** (num_qubits - rx) - 2**rz
        default = counter.count_logicals()
        assert default.weight is not None and results[default.weight] == default
        assert not any(result.count_x or result.count_z for result in results[: default.weight])


def test_methods_agree_on_conjugated_sparse_codes() -> None:
    rng = np.random.default_rng(3)
    code = codes.CSSCode.stack([codes.SteaneCode()] * 3)
    counter = lc.CSSLogicalCounter(code)
    for _ in range(6):
        counter = counter.conjugated_cnot(*map(int, rng.choice(counter.num_qubits, 2, False)))
        assert counter.count_logicals(
            full_pauli=True, method="brouwer_zimmermann"
        ) == counter.count_logicals(full_pauli=True, method="brute_force")


def test_dense_eighty_qubits_against_physical_enumeration() -> None:
    """Enumerate physical supports of weight at most four, independently of the counter."""
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
    assert np.mean(np.vstack([x, z])) > 0.4
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
        found = 0
        for weight in range(1, 5):
            for positions in itertools.combinations(range(n), weight):
                if functools.reduce(int.__xor__, (columns[jj] for jj in positions)):
                    continue
                reduced = sum(1 << jj for jj in positions)
                for pivot in sorted(pivots, reverse=True):
                    if reduced >> pivot & 1:
                        reduced ^= pivots[pivot]
                if reduced:
                    assert weight == 4
                    found += 1
        counts.append(found)
    assert counts == [1, 0]
    assert _counter(x, z).count_logicals(full_pauli=True) == Counts(4, 1, 0, 1)


####################################################################################################
# information sets and ownership


@pytest.mark.parametrize("width", [15, 65])
def test_overlapping_information_sets(width: int) -> None:
    """Overlapping sets visit words repeatedly; ownership and coverage keep the count exact."""
    checks = np.asarray(codes.HammingCode(4).matrix, dtype=np.uint8)
    checks_x, checks_z = _frozen(checks, checks, width)
    counter = _counter(checks_x, checks_z)
    basis, labels = _get_nested_code_basis(_get_kernel(counter.checks_z), counter.checks_x)
    family = []
    for order in (
        np.arange(width),
        np.arange(width)[::-1],
        np.random.default_rng(4).permutation(width),
    ):
        rows, tags, pivots = _row_reduce_binary(basis, labels, columns=order)
        family.append(_InformationSet(rows, tags, len(pivots), pivots))
    coverage = int(np.bincount(np.concatenate([item.pivots for item in family])).max())
    assert coverage > 1

    rank_rows = _pack_bits(counter.checks_x)
    with (
        mock.patch.object(lc, "_get_overlapping_information_sets", return_value=(family, coverage)),
        mock.patch.object(
            lc,
            "_get_brouwer_zimmermann_lower_bound",
            wraps=lc._get_brouwer_zimmermann_lower_bound,
        ) as bound,
    ):
        histogram = lc._count_words(
            basis,
            labels,
            3,
            "brouwer_zimmermann",
            rank_rows,
            complement=False,
            max_exponent=3,
            reject_lighter=True,
        )
    # The X-only logical words of weight three are the 35 weight-three Hamming codewords.
    assert histogram is not None and histogram.sum() == 35
    assert all(call.kwargs["max_coverage"] == coverage for call in bound.call_args_list)
    reference = lc._count_words(
        basis,
        labels,
        3,
        "brute_force",
        rank_rows,
        complement=False,
        max_exponent=3,
        reject_lighter=True,
    )
    assert reference is not None and np.array_equal(histogram, reference)


@pytest.mark.parametrize("interpreted", [False, True])
def test_ownership_breaks_ties_by_layer_then_index(interpreted: bool) -> None:
    generators = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    other = np.array([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=np.uint8)
    decoders = np.stack([lc._get_decoder(generators), lc._get_decoder(other)])
    word = _pack_bits(np.array([[1, 1, 1, 1]], dtype=np.uint8))[0]
    with _maybe_interpreted(interpreted):
        is_owned = lc._is_owned
        # The first set visits 1111 in layer two, and the second set in layer one.
        assert not is_owned(word, decoders, 0, 2)
        assert is_owned(word, decoders, 1, 1)
        # Two copies of one set visit every word in the same layer; the first copy owns it.
        same = np.stack([decoders[0], decoders[0]])
        assert is_owned(word, same, 0, 2)
        assert not is_owned(word, same, 1, 2)


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_resumable_chunks(chunk_size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lc, "_CHUNK_SIZE", chunk_size)
    code = codes.CSSCode.stack([codes.SteaneCode()] * 2)
    counter = lc.CSSLogicalCounter(code)
    with _maybe_interpreted(True):
        for method in METHODS:
            assert counter.count_logicals(full_pauli=True, method=method) == Counts(3, 14, 14, 42)


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("width", [7, 70])
def test_masked_rank(interpreted: bool, width: int) -> None:
    rng = np.random.default_rng(width)
    field = galois.GF(2)
    with _maybe_interpreted(interpreted):
        for _ in range(20):
            rows = rng.integers(0, 2, (int(rng.integers(0, 6)), width), dtype=np.uint8)
            mask = rng.integers(0, 2, width, dtype=np.uint8).astype(bool)
            packed_rows = _pack_bits(rows)
            packed_mask = _pack_bits(mask[None].astype(np.uint8))[0]
            scratch = np.empty_like(packed_rows)
            for complement, columns in ((False, mask), (True, ~mask)):
                expected = (
                    np.linalg.matrix_rank(field(rows[:, columns]))
                    if len(rows) and columns.any()
                    else 0
                )
                assert (
                    lc._get_masked_rank(packed_rows, packed_mask, complement, scratch) == expected
                )


####################################################################################################
# conjugation and validation


@pytest.mark.parametrize("interpreted", [False, True])
def test_conjugation(interpreted: bool) -> None:
    code = codes.SteaneCode()
    counter = lc.CSSLogicalCounter(code)
    before = counter.checks_x
    with _maybe_interpreted(interpreted):
        changed = counter.conjugated_cnot(1, 6)
    assert np.array_equal(counter.checks_x, before)
    # Conjugated checks are put in reduced row echelon form only when they are requested.
    assert changed._canonical is None
    expected = before.copy()
    expected[:, 6] ^= expected[:, 1]
    assert np.array_equal(changed.checks_x, _get_independent_rows(expected))
    assert changed._canonical is not None
    assert counter.is_self_dual and not changed.is_self_dual
    # A second CNOT restores a self-dual code, which is recognized but no longer shares its data.
    restored = changed.conjugated_cnot(1, 6)
    assert restored.is_self_dual and not restored._self_dual
    with _maybe_interpreted(interpreted):
        general = counter.conjugated([1, 6], [[1, 0], [1, 1]], [[1, 1], [0, 1]])
    assert np.array_equal(general.checks_x, changed.checks_x)
    assert np.array_equal(general.checks_z, changed.checks_z)
    assert np.array_equal(changed.conjugated_cnot(1, 6).checks_x, before)
    assert changed.count_logicals(full_pauli=True) == changed.count_logicals(
        full_pauli=True, method="brute_force"
    )

    # The SWEL-preserving Clifford keeps the code self-dual.
    swel = counter.conjugated_swel(0, 2, 3, 5)
    assert swel.is_self_dual
    assert swel.count_logicals(full_pauli=True) == swel.count_logicals(
        full_pauli=True, method="brute_force"
    )

    wide = _counter(np.empty((0, 65)), np.empty((0, 65)))
    for targets in ([0, 0], [-1, 2], [0, 7], [0.5, 2], [True, 2], [], list(range(65))):
        invalid_targets: Any = targets
        with pytest.raises(ValueError, match="1 to 64 distinct"):
            (wide if len(targets) == 65 else counter).conjugated(invalid_targets, [[1]], [[1]])
    with pytest.raises(ValueError, match="1 to 64 distinct"):
        counter.conjugated_swel(0, 1, 1, 2)
    with pytest.raises(ValueError, match="square"):
        counter.conjugated([0, 1], [[1]], [[1]])
    with _maybe_interpreted(interpreted):
        with pytest.raises(ValueError, match="pairing"):
            counter.conjugated([0, 1], np.zeros((2, 2)), np.eye(2))
        for invalid in (2 * np.eye(2), np.full((2, 2), np.nan), galois.GF(3)(np.eye(2, dtype=int))):
            with pytest.raises(ValueError, match="binary"):
                counter.conjugated([0, 1], invalid, np.eye(2))


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("size", [1, 2, 3, 5, 64])
def test_local_matrices_match_matrix_products(interpreted: bool, size: int) -> None:
    """Applying a packed local matrix equals multiplying the target columns by it over GF(2)."""
    rng = np.random.default_rng(size)
    field = galois.GF(2)
    while True:
        local = rng.integers(0, 2, (size, size))
        if np.linalg.matrix_rank(field(local)) == size:
            break
    inverse_transpose = np.asarray(np.linalg.inv(field(local)).T, dtype=int)
    rows = rng.integers(0, 2, (9, 70), dtype=np.uint8)
    positions = np.sort(rng.choice(70, size, replace=False))
    with _maybe_interpreted(interpreted):
        packed_x, packed_z = lc._pack_local_matrices(local, inverse_transpose, size)
        result = lc._apply_local(rows, positions, packed_x)
    expected = rows.astype(int)
    expected[:, positions] = expected[:, positions] @ local.T % 2
    assert np.array_equal(result, expected)
    assert len(packed_z) == size


def test_gate_constants() -> None:
    """The prepacked CNOT and SWEL-preserving matrices match their unpacked forms."""
    cnot = lc._pack_local_matrices([[1, 0], [1, 1]], [[1, 1], [0, 1]], 2)
    swel = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]
    for packed, expected in zip(
        (*lc._CNOT, *lc._SWEL), (*cnot, *lc._pack_local_matrices(swel, swel, 4)), strict=True
    ):
        assert packed.dtype == np.uint64 and np.array_equal(packed, expected)
    # Conjugation checks identity, not equality, to recognize gates that act identically on X and Z.
    assert lc._SWEL[0] is lc._SWEL[1]


def test_code_input_and_lazily_built_code() -> None:
    steane = codes.SteaneCode()
    counter = lc.CSSLogicalCounter(steane)
    assert counter.code is steane
    assert counter.count_logicals(full_pauli=True) == Counts(3, 7, 7, 21)

    # Conjugated counters build a code with the same checks on demand.
    derived = counter.conjugated_cnot(0, 1)
    assert derived._code is None
    code = derived.code
    assert derived.code is code and isinstance(code, codes.CSSCode)
    assert np.array_equal(_get_independent_rows(code.matrix_x.view(np.ndarray)), derived.checks_x)
    assert np.array_equal(_get_independent_rows(code.matrix_z.view(np.ndarray)), derived.checks_z)

    not_css: Any = codes.FiveQubitCode()
    with pytest.raises(TypeError, match="Expected a CSSCode"):
        lc.CSSLogicalCounter(not_css)
    with pytest.raises(ValueError, match="binary"):
        lc.CSSLogicalCounter(codes.CSSCode([[1, 1]], [[1, 1]], field=3))


####################################################################################################
# subsystem codes and an independent enumeration


def _span(rows: npt.ArrayLike) -> set[int]:
    """All combinations of the given binary rows, as integers whose bit j is column j."""
    words = {0}
    for row in np.asarray(rows, dtype=int):
        value = sum(int(bit) << jj for jj, bit in enumerate(row))
        words |= {word ^ value for word in words}
    return words


def _enumerate(code: codes.CSSCode) -> tuple[list[int], list[int], int, int]:
    """Count logical operators by enumerating the normalizers, using CSSCode's stabilizers.

    Returns the X-only and Z-only counts at every weight, the distance, and the number of nontrivial
    logical operators of any type at the distance.
    """
    num_qubits = len(code)
    normalizers = []
    for stabilizers in (code.get_stabilizer_ops(Pauli.Z), code.get_stabilizer_ops(Pauli.X)):
        field = galois.GF(2)
        kernel = field(stabilizers).null_space() if len(stabilizers) else field.Identity(num_qubits)
        normalizers.append(_span(kernel))
    normal_x, normal_z = normalizers
    gauge_x, gauge_z = _span(code.matrix_x), _span(code.matrix_z)
    counts_x, counts_z = [0] * (num_qubits + 1), [0] * (num_qubits + 1)
    for word in normal_x - gauge_x:
        counts_x[word.bit_count()] += 1
    for word in normal_z - gauge_z:
        counts_z[word.bit_count()] += 1
    distance = next(
        weight for weight in range(1, num_qubits + 1) if counts_x[weight] or counts_z[weight]
    )
    full = sum(
        (word_x | word_z).bit_count() == distance
        for word_x, word_z in itertools.product(normal_x, normal_z)
        if word_x not in gauge_x or word_z not in gauge_z
    )
    return counts_x, counts_z, distance, full


def _random_subsystem_code(
    rng: np.random.Generator, num_qubits: int, *, self_dual: bool = False
) -> codes.CSSCode:
    """A random binary CSS subsystem code, optionally with identical X and Z checks."""
    while True:
        checks_x = rng.integers(0, 2, (3, num_qubits))
        checks_z = checks_x if self_dual else rng.integers(0, 2, (3, num_qubits))
        code = codes.CSSCode(checks_x, checks_z)
        if code.is_subsystem_code and code.dimension > 0:
            return code


@pytest.mark.parametrize("method", METHODS)
def test_counts_match_enumeration(method: DistanceMethod) -> None:
    """Compare every weight, and full Pauli counts, with an enumeration of both normalizers."""
    rng = np.random.default_rng(17)
    examples = [
        codes.BaconShorCode(3),
        codes.BaconShorCode(2, 3),
        codes.SteaneCode(),
        *(_random_subsystem_code(rng, num_qubits) for num_qubits in (7, 8, 9, 10)),
        _random_subsystem_code(rng, 9, self_dual=True),
        *(_random_css_code(rng, 9, 3, 3) for _ in range(2)),
    ]
    for code in examples:
        counts_x, counts_z, distance, full = _enumerate(code)
        counter = lc.CSSLogicalCounter(code)
        assert counter.dimension == code.dimension
        for weight in range(len(code) + 1):
            expected = Counts(weight, counts_x[weight], counts_z[weight], None)
            assert counter.count_logicals(weight, method=method) == expected
        expected_full = Counts(distance, counts_x[distance], counts_z[distance], full)
        assert counter.count_logicals(method=method, full_pauli=True) == expected_full
        assert counter.count_logicals(distance, method=method, full_pauli=True) == expected_full
        assert distance == code.get_distance_exact()
        with pytest.raises(ValueError, match=f"less than weight {distance + 1}"):
            counter.count_logicals(distance + 1, method=method, full_pauli=True)


def test_conjugation_carries_sector_bases() -> None:
    """Conjugated counters reuse transformed sectors, and agree with counters built afresh."""
    rng = np.random.default_rng(29)
    for code in (codes.BaconShorCode(3), _random_css_code(rng, 10, 3, 3)):
        counter = lc.CSSLogicalCounter(code)
        assert counter.conjugated_cnot(0, 1)._sectors is None
        counter.count_logicals()
        for _ in range(6):
            control, target = map(int, rng.choice(counter.num_qubits, 2, replace=False))
            counter = counter.conjugated_cnot(control, target)
            assert counter._sectors is not None
            fresh = _counter(counter.checks_x, counter.checks_z)
            with mock.patch.object(lc, "_get_nested_code_basis", side_effect=AssertionError):
                results = [counter.count_logicals(w) for w in range(counter.num_qubits + 1)]
                carried = counter.count_logicals(full_pauli=True)
            assert results == [fresh.count_logicals(w) for w in range(fresh.num_qubits + 1)]
            assert carried == fresh.count_logicals(full_pauli=True)


def test_self_dual_codes_share_x_and_z_data() -> None:
    """Gates acting identically on X and Z keep the shared data; a CNOT splits it."""
    rng = np.random.default_rng(31)
    swel = np.array([[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]])
    counter = lc.CSSLogicalCounter(codes.CSSCode.stack([codes.SteaneCode()] * 2))
    assert counter._self_dual and counter._gauge_z is counter._gauge_x
    assert counter.conjugated_swel(0, 1, 2, 3)._sectors is None
    counter.count_logicals()
    for step in range(6):
        targets = list(map(int, rng.choice(counter.num_qubits, 4, replace=False)))
        if step == 1:
            counter = counter.conjugated(targets, swel, swel)
        elif step == 3:
            counter = counter.conjugated_cnot(*targets[:2])
        else:
            counter = counter.conjugated_swel(*targets)
        assert counter._sectors is not None
        shared = counter._sectors[1] is counter._sectors[0]
        assert shared == counter._self_dual == (step < 3)
        assert (counter._gauge_z is counter._gauge_x) == shared
        counts_x, counts_z, distance, full = _enumerate(counter.code)
        expected = Counts(distance, counts_x[distance], counts_z[distance], full)
        assert counter.count_logicals(full_pauli=True) == expected
        assert counter.count_logicals(full_pauli=True, method="brute_force") == expected

    # A self-dual subsystem code shares its stabilizers, which differ from its gauge generators.
    counter = lc.CSSLogicalCounter(_random_subsystem_code(rng, 9, self_dual=True))
    counter.count_logicals()
    for _ in range(3):
        counter = counter.conjugated_swel(*map(int, rng.choice(9, 4, replace=False)))
        assert counter._stabilizers_z is counter._stabilizers_x is not counter._gauge_x
        counts_x, counts_z, distance, full = _enumerate(counter.code)
        expected = Counts(distance, counts_x[distance], counts_z[distance], full)
        assert counter.count_logicals(full_pauli=True) == expected


def test_checks_are_copies_in_canonical_form() -> None:
    code = codes.SteaneCode()
    redundant = np.vstack([code.matrix_x, code.matrix_x, np.zeros((1, 7), dtype=int)])
    counter = _counter(redundant, code.matrix_z)
    assert len(counter.checks_x) == 3 and counter.dimension == 1 and counter.is_self_dual
    copy = counter.checks_x
    copy[:] = 0
    assert np.any(counter.checks_x)
    assert counter.count_logicals(full_pauli=True) == Counts(3, 7, 7, 21)


def test_invalid_inputs() -> None:
    counter = _counter([[0, 0]], [[0, 0]])
    invalid_method: Any = "unknown"
    with pytest.raises(ValueError, match="Unknown distance method"):
        counter.count_logicals(method=invalid_method)
    for weight in (-1, 3, 1.5, True, "1"):
        invalid_weight: Any = weight
        with pytest.raises(ValueError, match="weight must be an integer"):
            counter.count_logicals(invalid_weight)
    numpy_weight: Any = np.int64(2)
    assert counter.count_logicals(numpy_weight) == Counts(2, 1, 1, None)
