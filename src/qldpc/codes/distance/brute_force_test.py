# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact exhaustive binary distance enumeration."""

from __future__ import annotations

import itertools
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

import qldpc

from . import _binary as binary
from .conftest import _bitwise_count


@pytest.mark.parametrize("block_size", [1, 3, 13])
def test_get_distance_classical(block_size: int) -> None:
    """get_distance_classical visits every nontrivial XOR combination of generators exactly once."""
    generators = np.random.randint(2, size=(9, 137))

    # The mocked Hamming-weight function records every packed nonzero generator combination once.
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_hamming_weight(
        arr: npt.NDArray[np.uint64],
        buf: npt.NDArray[np.uint64] | None = None,
        out: npt.NDArray[np.uint64] | None = None,
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return binary._hamming_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance.brute_force._get_hamming_weight_fn",
        return_value=(_mock_hamming_weight, 0),
    ):
        distance = qldpc.codes.distance.get_distance_classical(
            np.vstack([generators, generators[0], generators[0] ^ generators[1]]),
            block_size=block_size,
            method="brute_force",
        )

    int_generators = binary._rows_to_ints(generators)
    expected_bitstrings = [
        tuple(np.bitwise_xor.reduce(np.vstack(gens)).tolist())
        for n in range(1, len(generators) + 1)
        for gens in itertools.combinations(int_generators, n)
    ]

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(expected_bitstrings)

    observed_array = np.array(observed_bitstrings, dtype=np.uint64)
    expected_distance = _bitwise_count(observed_array).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("block_size", [1, 3, 13])
def test_get_distance_quantum(block_size: int) -> None:
    """get_distance_quantum visits every stabilizer + logical-op combination exactly once.

    Uses Hamming (homogeneous) weight mode.
    """
    stabilizers = np.random.randint(2, size=(8, 97))
    logical_ops = np.random.randint(2, size=(5, 97))

    # The mocked Hamming-weight function records every nonzero logical combination combined
    # with every stabilizer combination, including the empty one.
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_hamming_weight(
        arr: npt.NDArray[np.uint64],
        buf: npt.NDArray[np.uint64] | None = None,
        out: npt.NDArray[np.uint64] | None = None,
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return binary._hamming_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance.brute_force._get_hamming_weight_fn",
        return_value=(_mock_hamming_weight, 0),
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            np.vstack([logical_ops, logical_ops[0] ^ stabilizers[0]]),
            np.vstack([stabilizers, stabilizers[0]]),
            homogeneous=True,
            block_size=block_size,
            method="brute_force",
        )

    int_stabilizers = binary._rows_to_ints(stabilizers)
    int_logical_ops = binary._rows_to_ints(logical_ops)
    expected_bitstrings = [
        tuple(np.bitwise_xor.reduce(np.vstack(stabs + ops)).tolist())
        for ns in range(len(stabilizers) + 1)
        for stabs in itertools.combinations(int_stabilizers, ns)
        for nl in range(1, len(logical_ops) + 1)
        for ops in itertools.combinations(int_logical_ops, nl)
    ]

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(expected_bitstrings)

    observed_array = np.array(observed_bitstrings, dtype=np.uint64)
    expected_distance = _bitwise_count(observed_array).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("block_size", [1, 3, 13])
def test_get_distance_quantum_symplectic(block_size: int) -> None:
    """get_distance_quantum visits every stabilizer + logical-op combination exactly once.

    Uses symplectic weight mode.
    """
    stabilizers = np.random.randint(2, size=(3, 98))
    logical_ops = np.random.randint(2, size=(5, 98))

    # The mocked symplectic-weight function records every nonzero logical combination combined
    # with every stabilizer combination, including the empty one.
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_symplectic_weight(
        arr: npt.NDArray[np.uint64],
        buf: npt.NDArray[np.uint64] | None = None,
        out: npt.NDArray[np.uint64] | None = None,
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return binary._symplectic_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance.brute_force._get_symplectic_weight_fn",
        return_value=(_mock_symplectic_weight, 0),
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            np.vstack([logical_ops, logical_ops[0] ^ stabilizers[0]]),
            np.vstack([stabilizers, stabilizers[0]]),
            homogeneous=False,
            block_size=block_size,
            method="brute_force",
        )

    int_stabilizers = binary._rows_to_ints(binary._riffle(stabilizers))
    int_logical_ops = binary._rows_to_ints(binary._riffle(logical_ops))
    expected_bitstrings = np.array(
        [
            np.bitwise_xor.reduce(np.vstack(stabs + ops))
            for ns in range(len(stabilizers) + 1)
            for stabs in itertools.combinations(int_stabilizers, ns)
            for nl in range(1, len(logical_ops) + 1)
            for ops in itertools.combinations(int_logical_ops, nl)
        ]
    )

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(map(tuple, expected_bitstrings))

    vals = (expected_bitstrings | (expected_bitstrings >> 1)) & 0x5555555555555555
    expected_distance = _bitwise_count(vals).sum(-1).min()
    assert distance == expected_distance


def test_get_distance_empty_stabilizers_symplectic() -> None:
    """A symplectic-weight distance with no stabilizers returns the min logical operator weight.

    With an empty ``stabilizers`` argument there is nothing to interleave, and the X/Z riffling
    still handles the empty input.
    """
    # The symplectic vector [1, 0 | 0, 0] is an X operator on qubit 0 with symplectic weight 1.
    distance = qldpc.codes.distance.get_distance_quantum(
        [[1, 0, 0, 0]], [], homogeneous=False, method="brute_force"
    )
    assert distance == 1


def test_get_distance_small_block_size() -> None:
    """A block_size smaller than the packed word count still yields the correct distance.

    When the columns span more than ``64 * (block_size + 1)`` bits, no operators are vectorized and
    the calculation proceeds by the sequential sweep alone.
    """
    generators = np.zeros((2, 200), dtype=np.uint64)
    generators[0] = 1  # First generator has Hamming weight 200.
    generators[1, :100] = 1  # Second generator has Hamming weight 100.
    assert qldpc.codes.distance.get_distance_classical(generators, block_size=1) == 100


def test_cutoff_early_exit() -> None:
    """cutoff makes get_distance_* return early once an operator of weight <= cutoff is found.

    Exercises all three early-exit return paths: the pre-loop vectorized block, the logical-op
    sweep, and the nested stabilizer sweep.
    """
    # The pre-loop vectorized block contains a codeword with Hamming weight 1.
    assert (
        qldpc.codes.distance.get_distance_classical([[1, 0, 0, 0]], cutoff=1, method="brute_force")
        == 1
    )

    # The logical-operator sweep with block_size=1 finds a generator with Hamming weight 1.
    generators = [[1, 0, 1], [0, 1, 0]]
    assert (
        qldpc.codes.distance.get_distance_classical(
            generators, block_size=1, cutoff=1, method="brute_force"
        )
        == 1
    )

    # The stabilizer sweep finds a Hamming-weight-1 operator only after XORing a stabilizer row.
    logical_ops = [[0, 0, 1, 1, 1]]
    stabilizers = [[1, 0, 0, 1, 0], [0, 1, 1, 1, 1]]
    distance = qldpc.codes.distance.get_distance_quantum(
        logical_ops,
        stabilizers,
        block_size=1,
        cutoff=1,
        homogeneous=True,
        method="brute_force",
    )
    assert distance == 1
