# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact-distance helpers in qldpc.codes.distance."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import qldpc

from . import _binary as binary
from .conftest import _bitwise_count, _get_random_full_rank_matrix


def test_css_brouwer_zimmermann_special_inputs() -> None:
    """The joint CSS helper handles empty, trivial, and nontrivial quotient sectors."""
    empty = np.empty((0, 4), dtype=np.uint8)
    assert (
        qldpc.codes.get_distance_css_brouwer_zimmermann(
            [(empty, empty)],
            cutoff=0,
        )
        == 4
    )
    assert (
        qldpc.codes.get_distance_css_brouwer_zimmermann(
            [([[1, 0]], [[1, 0]])],
            cutoff=0,
        )
        == 0
    )

    with mock.patch(
        "qldpc.codes.distance.core._get_distance_brouwer_zimmermann_many",
        return_value=2,
    ) as mock_search:
        distance = qldpc.codes.get_distance_css_brouwer_zimmermann(
            [([[1, 1]], [])],
            cutoff=0,
            upper_bound=3,
        )
    assert distance == 2
    assert mock_search.call_args.kwargs["upper_bound"] == 3


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_information_set_search() -> None:
    """Information-set search excludes stabilizers, finds logical products, and honors cutoffs."""
    # The Hamming-weight-one stabilizer is lighter than every logical operator.
    # With block_size=0, some Brouwer-Zimmermann batches contain only that stabilizer.
    distance = qldpc.codes.get_distance_quantum(
        [[1, 1, 1, 1]], [[1, 0, 0, 0]], homogeneous=True, cutoff=0, block_size=0
    )
    assert distance == 3

    # The lightest logical operator is the product of both logical generators.
    distance = qldpc.codes.get_distance_quantum(
        [[1, 1, 1, 0], [1, 1, 0, 1]], [], homogeneous=True, cutoff=0
    )
    assert distance == 2

    # Every generator in every information set has Hamming weight at least three.
    # Certifying the weight-two codeword requires searching two-generator combinations.
    generators = [
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    ]
    assert qldpc.codes.get_distance_classical(generators, cutoff=0) == 2

    # Both reduced generators have Hamming weight three.
    # Their weight-two sum reaches the distance cutoff.
    assert qldpc.codes.get_distance_classical([[1, 0, 1, 1], [0, 1, 1, 1]], cutoff=2) == 2

    # Row reduction exposes a Hamming-weight-one generator before any search runs.
    assert qldpc.codes.get_distance_classical([[1, 1, 0], [0, 1, 1], [1, 1, 1]], cutoff=1) == 1


def test_brouwer_zimmermann_special_inputs() -> None:
    """Brouwer-Zimmermann validates matrix shapes and preserves quotient edge cases."""
    assert qldpc.codes.get_distance_classical(np.empty((0, 4), dtype=np.uint8), cutoff=0) == 4
    assert (
        qldpc.codes.get_distance_quantum(
            np.empty((0, 4), dtype=np.uint8),
            np.empty((0, 4), dtype=np.uint8),
            homogeneous=True,
        )
        == 4
    )
    assert qldpc.codes.get_distance_quantum([[1, 0]], [[1, 0]], homogeneous=True) == 0
    methods: tuple[qldpc.codes.DistanceMethod, ...] = ("brouwer_zimmermann", "brute_force")
    for method in methods:
        for stabilizers in ([[1, 0, 0, 0]], [[1, 0, 0, 0]] * 2):
            assert (
                qldpc.codes.get_distance_quantum(
                    np.empty((0, 4), dtype=np.uint8), stabilizers, method=method
                )
                == 2
            )

    with pytest.raises(ValueError, match="incompatible widths"):
        qldpc.codes.get_distance_quantum([[1, 0, 0]], [[1, 0]], homogeneous=True)
    with pytest.raises(ValueError, match="even number of columns"):
        qldpc.codes.get_distance_quantum([[1, 0, 0]], [], homogeneous=False)
    with pytest.raises(ValueError, match="even number of columns"):
        qldpc.codes.get_distance_quantum(
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint8),
            homogeneous=False,
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        qldpc.codes.get_distance_classical(np.zeros((1, 1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="infer the width"):
        qldpc.codes.get_distance_classical([])


def test_get_distance_classical_methods() -> None:
    """get_distance_classical dispatches to NumPy bitcount or the fallback as available."""
    generators = np.random.randint(2, size=(6, 56), dtype=np.uint64)
    distance_default = qldpc.codes.distance.get_distance_classical(generators, block_size=3)

    # The mocked NumPy backend uses a per-word popcount reference when NumPy lacks bitwise_count.
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # This branch verifies the NumPy popcount backend for packed Hamming words.
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._binary._hamming_weight",
            wraps=binary._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_classical(generators, block_size=3)
        bitcount.assert_called()
        fallback.assert_not_called()
        assert distance == distance_default

    # This branch verifies the fallback popcount backend for packed Hamming words.
    with (
        mock.patch("numpy.bitwise_count", None, create=True),
        mock.patch(
            "qldpc.codes.distance._binary._hamming_weight",
            wraps=binary._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_classical(generators, block_size=3)
        fallback.assert_called()
        assert distance == distance_default


def test_distance_backend_validation() -> None:
    """Distance backend and exact-method selectors are restricted to documented values."""
    for backend in ["auto", "decoder", "gap", "sqetch"]:
        qldpc.codes.validate_distance_backend(backend)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown distance backend"):
        qldpc.codes.validate_distance_backend("other")  # type: ignore[arg-type]

    for method in ["brouwer_zimmermann", "brute_force"]:
        qldpc.codes.validate_distance_method(method)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown distance method"):
        qldpc.codes.validate_distance_method("other")  # type: ignore[arg-type]


def test_get_distance_quantum_methods() -> None:
    """Homogeneous get_distance_quantum dispatches to NumPy bitcount or the fallback."""
    stabilizers = np.random.randint(2, size=(4, 56), dtype=np.uint64)
    logical_ops = np.random.randint(2, size=(3, 56), dtype=np.uint64)
    distance_default = qldpc.codes.distance.get_distance_quantum(
        logical_ops, stabilizers, block_size=3, homogeneous=True
    )

    # The mocked NumPy backend uses a per-word popcount reference when NumPy lacks bitwise_count.
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # This branch verifies the NumPy popcount backend for homogeneous quantum packed words.
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._binary._hamming_weight",
            wraps=binary._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, block_size=3, homogeneous=True
        )
        bitcount.assert_called()
        fallback.assert_not_called()
        assert distance == distance_default

    # This branch verifies the fallback popcount backend for homogeneous quantum packed words.
    with (
        mock.patch("numpy.bitwise_count", None, create=True),
        mock.patch(
            "qldpc.codes.distance._binary._hamming_weight",
            wraps=binary._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, block_size=3, homogeneous=True
        )
        fallback.assert_called()
        assert distance == distance_default


def test_get_distance_quantum_methods_symplectic() -> None:
    """Symplectic get_distance_quantum dispatches to NumPy bitcount or the fallback."""
    stabilizers = np.random.randint(2, size=(4, 56), dtype=np.uint64)
    logical_ops = np.random.randint(2, size=(3, 56), dtype=np.uint64)
    distance_default = qldpc.codes.distance.get_distance_quantum(
        logical_ops,
        stabilizers,
        block_size=3,
        homogeneous=False,
        method="brute_force",
    )

    # The mocked NumPy backend uses a per-word popcount reference when NumPy lacks bitwise_count.
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # This branch verifies the NumPy popcount backend for symplectic quantum packed words.
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._binary._symplectic_weight",
            wraps=binary._symplectic_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops,
            stabilizers,
            block_size=3,
            homogeneous=False,
            method="brute_force",
        )
        bitcount.assert_called()
        fallback.assert_not_called()
        assert distance == distance_default

    # This branch verifies the fallback popcount backend for symplectic quantum packed words.
    with (
        mock.patch("numpy.bitwise_count", None, create=True),
        mock.patch(
            "qldpc.codes.distance._binary._symplectic_weight",
            wraps=binary._symplectic_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops,
            stabilizers,
            block_size=3,
            homogeneous=False,
            method="brute_force",
        )
        fallback.assert_called()
        assert distance == distance_default


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.classical.HammingCode(4), 3),
        (qldpc.codes.classical.RepetitionCode(3), 3),
        (qldpc.codes.classical.RepetitionCode(8), 8),
        (qldpc.codes.classical.RingCode(8), 8),
    ],
)
def test_get_distance_classical_known_codes(
    code: qldpc.codes.ClassicalCode, expected_distance: int
) -> None:
    """Classical distance matches known values for Hamming, repetition, and ring codes."""
    distance = qldpc.codes.distance.get_distance_classical(code.generator)
    assert distance == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.quantum.C4Code(), 2),
        (qldpc.codes.quantum.C6Code(), 2),
        (qldpc.codes.quantum.SteaneCode(), 3),
        (qldpc.codes.quantum.SurfaceCode(3), 3),
        (qldpc.codes.quantum.SurfaceCode(4), 4),
        (qldpc.codes.quantum.ToricCode(4), 4),
    ],
)
def test_get_distance_quantum_css_codes(code: qldpc.codes.CSSCode, expected_distance: int) -> None:
    """Quantum distance matches known values for C4, C6, Steane, and small surface/toric codes."""
    distance_x = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(qldpc.objects.Pauli.X),
        code.get_stabilizer_ops(qldpc.objects.Pauli.X),
        homogeneous=True,
    )
    distance_z = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(qldpc.objects.Pauli.Z),
        code.get_stabilizer_ops(qldpc.objects.Pauli.Z),
        homogeneous=True,
    )
    distance_all = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=True,
    )
    assert min(distance_x, distance_z, distance_all) == expected_distance

    # CSSCode.forget_distance clears the X-sector and Z-sector distance caches before recomputation.
    # Assigning only CSSCode._distance would leave cached sector distances available to the method.
    code.forget_distance()
    assert code.get_distance_exact() == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.quantum.FiveQubitCode(), 3),
    ],
)
def test_get_distance_quantum_noncss_codes(
    code: qldpc.codes.QuditCode, expected_distance: int
) -> None:
    """Quantum distance matches known values for small non-CSS codes."""
    distance = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=False,
    )
    assert distance == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_random_cross_checks() -> None:
    """Information-set search agrees with exhaustive enumeration on random nested codes."""
    rng = np.random.default_rng(118)
    for length in range(5, 10):
        basis = _get_random_full_rank_matrix(rng, min(5, length - 1), length)
        expected = qldpc.codes.get_distance_classical(basis, cutoff=0, method="brute_force")
        assert qldpc.codes.get_distance_classical(basis, cutoff=0) == expected

        stabilizers, logical_ops = basis[:2], basis[2:]
        expected = qldpc.codes.get_distance_quantum(
            logical_ops, stabilizers, cutoff=0, homogeneous=True, method="brute_force"
        )
        assert (
            qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=0, homogeneous=True)
            == expected
        )

    for num_qubits in range(3, 7):
        basis = _get_random_full_rank_matrix(rng, min(6, 2 * num_qubits - 1), 2 * num_qubits)
        stabilizers, logical_ops = basis[:2], basis[2:]
        expected = qldpc.codes.get_distance_quantum(
            logical_ops, stabilizers, cutoff=0, method="brute_force"
        )
        assert qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=0) == expected
