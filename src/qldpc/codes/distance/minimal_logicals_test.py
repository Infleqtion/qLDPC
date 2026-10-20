# SPDX-License-Identifier: Apache-2.0

"""Tests for exact counts of the minimal logical operators of binary CSS codes."""

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

from . import minimal_logicals as ml
from ._binary import (
    _get_independent_rows,
    _get_nested_code_basis,
    _row_reduce_binary,
)
from .brouwer_zimmermann import _InformationSet

Weights = ml.CSSMinimumLogicalWeights
METHODS: tuple[ml.MinimalLogicalMethod, ...] = ("brouwer_zimmermann", "brute_force")
KERNELS = ("_count_chunk", "_is_owned", "_get_masked_rank")


@contextlib.contextmanager
def _maybe_interpreted(interpreted: bool) -> Iterator[None]:
    """Run the Python implementations of the native kernels, so that coverage can trace them."""
    if not interpreted:
        yield
        return
    with mock.patch.multiple(ml, **{name: getattr(ml, name).py_func for name in KERNELS}):
        yield


def _bits(values: list[int], width: int) -> npt.NDArray[np.uint8]:
    """Binary rows whose bit j is bit j of each integer."""
    return np.array([[value >> jj & 1 for jj in range(width)] for value in values], dtype=np.uint8)


def _random_code(
    rng: np.random.Generator, num_qubits: int, rank_x: int, rank_z: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Random commuting checks: Z checks are random combinations of the kernel of the X checks."""
    checks_x = rng.integers(0, 2, (rank_x, num_qubits), dtype=np.uint8)
    kernel = ml._get_kernel(checks_x)
    checks_z = (rng.integers(0, 2, (rank_z, len(kernel))) @ kernel % 2).astype(np.uint8)
    return checks_x, checks_z


def _frozen(
    checks_x: npt.ArrayLike, checks_z: npt.ArrayLike, width: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Append qubits fixed in |0> by a Z check each; no minimal logical can touch them."""
    x, z = np.asarray(checks_x, dtype=np.uint8), np.asarray(checks_z, dtype=np.uint8)
    extra = width - x.shape[1]
    padded_x = np.pad(x, ((0, 0), (0, extra)))
    padded_z = np.vstack(
        [np.pad(z, ((0, 0), (0, extra))), np.hstack([np.zeros((extra, z.shape[1])), np.eye(extra)])]
    ).astype(np.uint8)
    return padded_x, padded_z


# X checks {0,1,2,3,4} and {2,4}, and Z checks {1,2,3,4} and {0,1}, on five qubits.  By hand, the
# X-minimal words are {2,3} and {3,4}, and the Z-minimal words are {2,4}, {0,3} and {1,3}.  The Z
# word {2,4} also supports the X stabilizer {2,4}, so there are 2 + 3 + 1 = 6 minimal Paulis.
_HAND_X, _HAND_Z = _bits([31, 20], 5), _bits([30, 3], 5)


####################################################################################################
# results


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("method", METHODS)
def test_known_codes(method: ml.MinimalLogicalMethod, interpreted: bool) -> None:
    steane = codes.SteaneCode()
    examples = (
        (steane.matrix_x, steane.matrix_z, Weights(3, 7, 7, 21)),
        # After layer one the BZ bound equals two, which would miss one of six minimal words.
        ([[1, 1, 1, 1]], [[1, 1, 1, 1]], Weights(2, 6, 6, 18)),
        (_HAND_X, _HAND_Z, Weights(2, 2, 3, 6)),
        # The Z sector is heavier, so it contributes nothing at the global minimum.
        ([[1, 1, 0]], [[0, 0, 1]], Weights(1, 2, 0, 2)),
        (np.empty((0, 3)), np.empty((0, 3)), Weights(1, 3, 3, 9)),
        (np.eye(4), np.empty((0, 4)), Weights(None, 0, 0, 0)),
    )
    with _maybe_interpreted(interpreted):
        for checks_x, checks_z, expected in examples:
            counter = ml.CSSLogicalWeightCounter(checks_x, checks_z)
            assert counter.get_minimum(method=method) == expected
            homogeneous = counter.get_minimum(method=method, full_pauli=False)
            assert homogeneous == Weights(
                expected.distance, expected.count_x, expected.count_z, None
            )
    assert ml.get_minimum_logical_weights_css(steane.matrix_x, steane.matrix_z) == examples[0][2]


def test_surface_code() -> None:
    code = codes.SurfaceCode(9)
    counter = ml.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z)
    assert counter.get_minimum() == Weights(9, 1556, 1556, 4608)


@pytest.mark.parametrize("width", [5, 63, 64, 65, 129])
def test_frozen_qubits_preserve_counts(width: int) -> None:
    """Frozen qubits change the physical width, not the minimal logical operators."""
    # Each frozen qubit adds a Z stabilizer to the normalizer basis, so brute force is unaffordable.
    methods = METHODS if width < 20 else ("brouwer_zimmermann",)
    for method in methods:
        counter = ml.CSSLogicalWeightCounter(*_frozen(_HAND_X, _HAND_Z, width))
        assert counter.get_minimum(method=method) == Weights(2, 2, 3, 6)
    steane = codes.SteaneCode()
    counter = ml.CSSLogicalWeightCounter(*_frozen(steane.matrix_x, steane.matrix_z, width + 2))
    assert counter.get_minimum() == Weights(3, 7, 7, 21)


@pytest.mark.parametrize(
    "num_qubits, rank_x, rank_z",
    [(10, 3, 3), (14, 4, 5), (16, 10, 2), (24, 14, 4), (32, 14, 12), (40, 22, 18)],
)
def test_methods_agree_on_random_codes(num_qubits: int, rank_x: int, rank_z: int) -> None:
    """Brute force visits every word once, so it checks BZ's set selection and ownership."""
    rng = np.random.default_rng(num_qubits)
    for _ in range(3):
        counter = ml.CSSLogicalWeightCounter(*_random_code(rng, num_qubits, rank_x, rank_z))
        assert counter.get_minimum(method="brouwer_zimmermann") == counter.get_minimum(
            method="brute_force"
        )


def test_methods_agree_on_conjugated_sparse_codes() -> None:
    rng = np.random.default_rng(3)
    code = codes.CSSCode.stack([codes.SteaneCode()] * 3)
    counter = ml.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z)
    for _ in range(6):
        counter = counter.conjugated_cnot(*map(int, rng.choice(counter.num_qubits, 2, False)))
        assert counter.get_minimum(method="brouwer_zimmermann") == counter.get_minimum(
            method="brute_force"
        )


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
    assert ml.CSSLogicalWeightCounter(x, z).get_minimum() == Weights(4, 1, 0, 1)


####################################################################################################
# information sets and ownership


@pytest.mark.parametrize("width", [15, 65])
def test_overlapping_information_sets(width: int) -> None:
    """Overlapping sets visit words repeatedly; ownership and coverage keep the count exact."""
    checks = np.asarray(codes.HammingCode(4).matrix, dtype=np.uint8)
    checks_x, checks_z = _frozen(checks, checks, width)
    counter = ml.CSSLogicalWeightCounter(checks_x, checks_z)
    basis, labels = _get_nested_code_basis(ml._get_kernel(counter.checks_z), counter.checks_x)
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

    lift = ml._pack(counter.checks_x)
    with (
        mock.patch.object(ml, "_get_overlapping_information_sets", return_value=(family, coverage)),
        mock.patch.object(
            ml,
            "_get_brouwer_zimmermann_lower_bound",
            wraps=ml._get_brouwer_zimmermann_lower_bound,
        ) as bound,
    ):
        histogram = ml._count_minimal_words(
            basis, labels, 3, "brouwer_zimmermann", lift, complement=False, base=3
        )
    # The 35 weight-three words of the [15, 11, 3] Hamming code are its X-minimal words.
    assert histogram.sum() == 35
    assert all(call.kwargs["max_coverage"] == coverage for call in bound.call_args_list)
    reference = ml._count_minimal_words(
        basis, labels, 3, "brute_force", lift, complement=False, base=3
    )
    assert np.array_equal(histogram, reference)


@pytest.mark.parametrize("interpreted", [False, True])
def test_ownership_breaks_ties_by_layer_then_index(interpreted: bool) -> None:
    generators = np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.uint8)
    other = np.array([[1, 0, 1, 0], [1, 1, 1, 1]], dtype=np.uint8)
    decoders = np.stack([ml._get_decoder(generators), ml._get_decoder(other)])
    word = ml._pack(np.array([[1, 1, 1, 1]], dtype=np.uint8))[0]
    with _maybe_interpreted(interpreted):
        is_owned = ml._is_owned
        # The first set visits 1111 in layer two, and the second set in layer one.
        assert not is_owned(word, decoders, 0, 2)
        assert is_owned(word, decoders, 1, 1)
        # Two copies of one set visit every word in the same layer; the first copy owns it.
        same = np.stack([decoders[0], decoders[0]])
        assert is_owned(word, same, 0, 2)
        assert not is_owned(word, same, 1, 2)


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_resumable_chunks(chunk_size: int, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ml, "_CHUNK_SIZE", chunk_size)
    code = codes.CSSCode.stack([codes.SteaneCode()] * 2)
    counter = ml.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z)
    with _maybe_interpreted(True):
        for method in METHODS:
            assert counter.get_minimum(method=method) == Weights(3, 14, 14, 42)


@pytest.mark.parametrize("interpreted", [False, True])
@pytest.mark.parametrize("width", [7, 70])
def test_masked_rank(interpreted: bool, width: int) -> None:
    rng = np.random.default_rng(width)
    field = galois.GF(2)
    with _maybe_interpreted(interpreted):
        for _ in range(20):
            rows = rng.integers(0, 2, (int(rng.integers(0, 6)), width), dtype=np.uint8)
            mask = rng.integers(0, 2, width, dtype=np.uint8).astype(bool)
            packed_rows = ml._pack(rows)
            packed_mask = ml._pack(mask[None].astype(np.uint8))[0]
            scratch = np.empty_like(packed_rows)
            for complement, columns in ((False, mask), (True, ~mask)):
                expected = (
                    np.linalg.matrix_rank(field(rows[:, columns]))
                    if len(rows) and columns.any()
                    else 0
                )
                assert (
                    ml._get_masked_rank(packed_rows, packed_mask, complement, scratch) == expected
                )


####################################################################################################
# conjugation and validation


def test_conjugation() -> None:
    code = codes.SteaneCode()
    counter = ml.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z)
    before = counter.checks_x
    changed = counter.conjugated_cnot(1, 6)
    assert np.array_equal(counter.checks_x, before)
    expected = before.copy()
    expected[:, 6] ^= expected[:, 1]
    assert np.array_equal(changed.checks_x, _get_independent_rows(expected))
    assert counter.is_self_dual and not changed.is_self_dual
    assert np.array_equal(changed.conjugated_cnot(1, 6).checks_x, before)
    assert changed.get_minimum() == changed.get_minimum(method="brute_force")

    # The SWEL-preserving Clifford keeps the code self-dual.
    swel = counter.conjugated_swel([0, 2, 3, 5])
    assert swel.is_self_dual
    assert swel.get_minimum() == swel.get_minimum(method="brute_force")

    for targets in ([0, 0], [-1, 2], [0, 7], [0.5, 2], [True, 2], []):
        invalid_targets: Any = targets
        with pytest.raises(ValueError, match="targets"):
            counter.conjugated(invalid_targets, np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="four"):
        counter.conjugated_swel([0, 1])
    with pytest.raises(ValueError, match="square"):
        counter.conjugated([0, 1], [[1]], [[1]])
    with pytest.raises(ValueError, match="pairing"):
        counter.conjugated([0, 1], np.zeros((2, 2)), np.eye(2))
    with pytest.raises(ValueError, match="binary"):
        counter.conjugated([0, 1], 2 * np.eye(2), np.eye(2))


def test_checks_are_copies_in_canonical_form() -> None:
    code = codes.SteaneCode()
    redundant = np.vstack([code.matrix_x, code.matrix_x, np.zeros((1, 7), dtype=int)])
    counter = ml.CSSLogicalWeightCounter(redundant, code.matrix_z)
    assert len(counter.checks_x) == 3 and counter.dimension == 1 and counter.is_self_dual
    copy = counter.checks_x
    copy[:] = 0
    assert np.any(counter.checks_x)
    assert counter.get_minimum() == Weights(3, 7, 7, 21)


def test_invalid_inputs() -> None:
    for checks in ([[2, 0]], [[np.nan, 0]], galois.GF(3)([[1, 0]])):
        with pytest.raises(ValueError, match="binary"):
            ml.CSSLogicalWeightCounter(checks, [[0, 0]])
    with pytest.raises(ValueError, match="widths"):
        ml.CSSLogicalWeightCounter([[1, 0]], [[1, 0, 0]])
    with pytest.raises(ValueError, match="commute"):
        ml.CSSLogicalWeightCounter([[1, 0]], [[1, 0]])
    invalid_method: Any = "unknown"
    with pytest.raises(ValueError, match="Unknown"):
        ml.CSSLogicalWeightCounter([[0, 0]], [[0, 0]]).get_minimum(method=invalid_method)
