"""Unit tests for common.py

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import itertools
import subprocess
import unittest.mock

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli

####################################################################################################
# classical code tests


def test_constructions_classical(pytestconfig: pytest.Config) -> None:
    """Classical code constructions."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.ClassicalCode.random(5, 3, field=2, seed=np.random.randint(2**32))
    assert len(code) == code.num_bits == 5
    assert "ClassicalCode" in str(code)
    assert code.get_random_word() in code

    code = codes.ClassicalCode.random(5, 3, field=3, seed=np.random.randint(2**32))
    assert "GF(3)" in str(code)

    num_bits = 2
    code = codes.RepetitionCode(num_bits, field=3)
    assert code.num_bits == num_bits
    assert code.dimension == 1
    assert code.get_weight() == 2

    # invalid classical code construction
    with pytest.raises(ValueError, match="inconsistent"):
        codes.ClassicalCode(codes.ClassicalCode.random(2, 2, field=2), field=3)

    # construct a code from its generator matrix
    code = codes.ClassicalCode.random(5, 3)
    assert codes.ClassicalCode.equiv(code, codes.ClassicalCode.from_generator(code.generator))

    # puncture a code
    assert codes.ClassicalCode.from_generator(code.generator[:, 1:]) == code.puncture(0)

    # shortening a repetition code yields a trivial code
    num_bits = 3
    code = codes.RepetitionCode(num_bits)
    words = [[0] * (num_bits - 1)]
    assert np.array_equal(code.shorten(0).words(), words)

    # stack two codes
    code_a = codes.ClassicalCode.random(5, 3, field=3, seed=np.random.randint(2**32))
    code_b = codes.ClassicalCode.random(5, 3, field=3, seed=np.random.randint(2**32))
    code = codes.ClassicalCode.stack(code_a, code_b)
    assert len(code) == len(code_a) + len(code_b)
    assert code.dimension == code_a.dimension + code_b.dimension

    # stacking codes over different fields is not supported
    with pytest.raises(ValueError, match="different fields"):
        code_b = codes.RepetitionCode(2, field=2)
        code = codes.ClassicalCode.stack(code_a, code_b)


def test_named_codes(order: int = 2) -> None:
    """Named codes from the GAP computer algebra system."""
    code = codes.RepetitionCode(order)
    checks = [list(row) for row in code.matrix.view(np.ndarray)]

    with unittest.mock.patch("qldpc.external.codes.get_code", return_value=(checks, None)):
        assert codes.ClassicalCode.from_name(f"RepetitionCode({order})") == code


def test_dual_code(bits: int = 5, checks: int = 3, field: int = 3) -> None:
    """Dual code construction."""
    code = codes.ClassicalCode.random(bits, checks, field)
    assert all(word_a @ word_b == 0 for word_a in code.words() for word_b in (~code).words())


def test_tensor_product(
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (4, 2),
) -> None:
    """Tensor product of classical codes."""
    code_a = codes.ClassicalCode.random(*bits_checks_a)
    code_b = codes.ClassicalCode.random(*bits_checks_b)
    code_ab = codes.ClassicalCode.tensor_product(code_a, code_b)
    basis = np.reshape(code_ab.generator, (-1, code_a.num_bits, code_b.num_bits))
    assert all(not (code_a.matrix @ word @ code_b.matrix.T).any() for word in basis)

    n_a, k_a, d_a = code_a.get_code_params()
    n_b, k_b, d_b = code_b.get_code_params()
    n_ab, k_ab, d_ab = code_ab.get_code_params()
    assert (n_ab, k_ab, d_ab) == (n_a * n_b, k_a * k_b, d_a * d_b)

    with pytest.raises(ValueError, match="Cannot take tensor product"):
        code_b = codes.ClassicalCode.random(*bits_checks_b, field=code_a.field.order**2)
        codes.ClassicalCode.tensor_product(code_a, code_b)


def test_distance_classical(bits: int = 3) -> None:
    """Distance of a vector from a classical code."""
    rep_code = codes.RepetitionCode(bits, field=2)

    # "forget" the exact code distance
    rep_code._exact_distance = None

    assert rep_code.get_distance(bound=True) == bits
    assert rep_code.get_distance() == bits
    for vector in itertools.product(rep_code.field.elements, repeat=bits):
        weight = np.count_nonzero(vector)
        dist_bound = rep_code.get_distance_bound(vector=vector)
        dist_exact = rep_code.get_distance_exact(vector=vector)
        assert dist_exact == min(weight, bits - weight)
        assert dist_exact <= dist_bound

    trivial_code = codes.ClassicalCode([[1, 0], [1, 1]])
    random_vector = np.random.randint(2, size=trivial_code.num_bits)
    assert trivial_code.dimension == 0
    assert trivial_code.get_distance_exact() is np.nan
    assert trivial_code.get_distance_bound() is np.nan
    assert (
        np.count_nonzero(random_vector)
        == trivial_code.get_distance_exact(vector=random_vector)
        == trivial_code.get_distance_bound(vector=random_vector)
        == trivial_code.get_one_distance_bound(vector=random_vector)
    )


def test_conversions_classical(bits: int = 5, checks: int = 3) -> None:
    """Conversions between matrix and graph representations of a classical code."""
    code = codes.ClassicalCode.random(bits, checks)
    assert np.array_equal(code.matrix, codes.ClassicalCode.graph_to_matrix(code.graph))


def get_mock_process(stdout: str) -> subprocess.CompletedProcess[str]:
    """Fake process with the given stdout."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def test_automorphism() -> None:
    """Compute automorphism group of the smallest nontrivial trinary Hamming code."""
    code = codes.HammingCode(2, field=3)
    automorphisms = "\n()\n(2,4,3)\n(2,3,4)\n"

    # raise an error when GAP is not installed
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
        pytest.raises(ValueError, match="Cannot build GAP group"),
    ):
        code.get_automorphism_group()

    # otherwise, check that automorphisms do indeed preserve the code space
    with (
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=True),
        unittest.mock.patch(
            "qldpc.external.gap.get_result", return_value=get_mock_process(automorphisms)
        ),
    ):
        group = code.get_automorphism_group()
        for member in group.generate():
            assert not np.any(code.matrix @ group.lift(member) @ code.generator.T)


####################################################################################################
# quantum code tests


def test_code_string() -> None:
    """Human-readable representation of a code."""
    code = codes.QuditCode([[0, 1]], field=2)
    assert "qubits" in str(code)

    code = codes.QuditCode([[0, 1]], field=3)
    assert "GF(3)" in str(code)

    code = codes.HGPCode(codes.RepetitionCode(2, field=2))
    assert "qubits" in str(code)

    code = codes.HGPCode(codes.RepetitionCode(2, field=3))
    assert "GF(3)" in str(code)


def get_random_qudit_code(qudits: int, checks: int, field: int = 2) -> codes.QuditCode:
    """Construct a random (but probably trivial or invalid) QuditCode."""
    return codes.QuditCode(
        codes.ClassicalCode.random(2 * qudits, checks, field).matrix,
        skip_validation=True,
    )


def test_qubit_code(num_qubits: int = 5, num_checks: int = 3) -> None:
    """Random qubit code."""
    assert get_random_qudit_code(num_qubits, num_checks, field=2).num_qubits == num_qubits
    with pytest.raises(ValueError, match="3-dimensional qudits"):
        assert get_random_qudit_code(num_qubits, num_checks, field=3).num_qubits


def test_qudit_code() -> None:
    """Miscellaneous qudit code tests and coverage."""
    code = codes.FiveQubitCode()
    assert code.dimension == 1
    assert code.get_weight() == 4
    assert code.get_logical_ops(Pauli.X).shape == code.get_logical_ops(Pauli.Z).shape

    # equivlence to code with redundant stabilizers
    redundant_code = codes.QuditCode(np.vstack([code.matrix, code.matrix]))
    assert codes.QuditCode.equiv(code, redundant_code)

    # cover calls to the known code exact distance
    assert code.get_code_params() == (5, 1, 3)
    assert code.get_distance(bound=True) == 3

    # "forget" the code distance and recompute
    code._exact_distance = None
    assert code.get_distance_exact() == 3

    code._exact_distance = None
    with pytest.raises(NotImplementedError, match="not implemented"):
        code.get_distance(bound=True)

    # stacking two codes
    two_codes = codes.QuditCode.stack(code, code)
    assert len(two_codes) == len(code) * 2
    assert two_codes.dimension == code.dimension * 2

    # stacking codes over different fields is not supported
    with pytest.raises(ValueError, match="different fields"):
        second_code = codes.SurfaceCode(2, field=3)
        codes.QuditCode.stack(code, second_code)


def test_undefined_distance() -> None:
    """The distance of dimension-0 codes is undefined."""
    assert codes.QuditCode([[0, 1]]).get_distance() is np.nan


@pytest.mark.parametrize("field", [2, 3])
def test_conversions_quantum(field: int, bits: int = 5, checks: int = 3) -> None:
    """Conversions between matrix and graph representations of a code."""
    code = get_random_qudit_code(bits, checks, field)
    graph = codes.QuditCode.matrix_to_graph(code.matrix)
    assert np.array_equal(code.matrix, codes.QuditCode.graph_to_matrix(graph))


@pytest.mark.parametrize("field", [2, 3])
def test_qudit_stabilizers(field: int, bits: int = 5, checks: int = 3) -> None:
    """Stabilizers of a QuditCode."""
    code_a = get_random_qudit_code(bits, checks, field)
    stabilizers = code_a.get_stabilizers()
    code_b = codes.QuditCode.from_stabilizers(*stabilizers, field=field, skip_validation=True)
    assert code_a == code_b
    assert stabilizers == code_b.get_stabilizers()

    with pytest.raises(ValueError, match="different lengths"):
        codes.QuditCode.from_stabilizers("I", "I I", field=field)


def test_trivial_deformations(num_qudits: int = 5, num_checks: int = 3, field: int = 3) -> None:
    """Trivial local Clifford deformations do not modify a code."""
    code = get_random_qudit_code(num_qudits, num_checks, field)
    assert code == code.conjugated(skip_validation=True)


def test_qudit_ops() -> None:
    """Logical operator construction for Galois qudit codes."""
    code: codes.QuditCode

    code = codes.FiveQubitCode()
    logical_ops = code.get_logical_ops()
    assert logical_ops.shape == (2, code.dimension, 2 * code.num_qudits)
    assert np.array_equal(logical_ops[0], [[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    assert np.array_equal(logical_ops[1], [[0, 1, 1, 0, 0, 0, 0, 0, 0, 1]])
    assert code.get_logical_ops() is code._logical_ops

    code = codes.QuditCode.from_stabilizers(*code.get_stabilizers(), "I I I I I", field=2)
    assert np.array_equal(logical_ops, code.get_logical_ops())


####################################################################################################
# CSS code tests


def test_css_code() -> None:
    """Miscellaneous CSS code tests and coverage."""
    code_x = codes.ClassicalCode.random(3, 2)

    code_z = ~code_x
    code = codes.CSSCode(code_x, code_z)
    assert code.get_weight() == max(code_x.get_weight(), code_z.get_weight())
    assert code.num_checks_x == code_x.num_checks
    assert code.num_checks_z == code_z.num_checks
    assert code.num_checks == code.num_checks_x + code.num_checks_z
    assert code == codes.CSSCode(code.code_x, code.code_z)

    code_z = codes.ClassicalCode.random(4, 2)
    with pytest.raises(ValueError, match="incompatible"):
        codes.CSSCode(code_x, code_z)

    with pytest.raises(ValueError, match="different fields"):
        code_z = codes.ClassicalCode.random(3, 2, field=code_x.field.order**2)
        codes.CSSCode(code_x, code_z)


def test_css_ops() -> None:
    """Logical operator construction for CSS codes."""
    code: codes.CSSCode

    code = codes.HGPCode(codes.ClassicalCode.random(4, 2, field=3))
    code.get_random_logical_op(Pauli.X, ensure_nontrivial=False)
    code.get_random_logical_op(Pauli.X, ensure_nontrivial=True)

    # test that logical operators have trivial syndromes
    logicals_x = code.get_logical_ops(Pauli.X)
    logicals_z = code.get_logical_ops(Pauli.Z)
    assert not np.any(logicals_x[:, len(code) :])
    assert not np.any(logicals_z[:, : len(code)])
    assert not np.any(code.matrix @ logicals_x.T)
    assert not np.any(code.matrix @ logicals_z.T)
    assert code.get_logical_ops() is code._logical_ops

    # test that logical operators are dual to each other
    logicals_x = logicals_x[:, : len(code)]
    logicals_z = logicals_z[:, len(code) :]
    assert np.array_equal(logicals_x @ logicals_z.T, np.eye(code.dimension, dtype=int))

    # successfullly construct and reduce logical operators in a code with "over-complete" checks
    dist = 4
    code = codes.ToricCode(dist, rotated=True, field=2)
    code.reduce_logical_ops()
    assert code.get_code_params() == (dist**2, 2, dist)
    assert not any(np.count_nonzero(op) < dist for op in code.get_logical_ops(Pauli.X))
    assert not any(np.count_nonzero(op) < dist for op in code.get_logical_ops(Pauli.Z))

    # reducing logical operator weight only supported for prime number fields
    code = codes.HGPCode(codes.ClassicalCode.random(4, 2, field=4))
    with pytest.raises(ValueError, match="prime number fields"):
        code.reduce_logical_op(Pauli.X, 0)

    # the 2x2 toric code has redundant stabilizers
    code = codes.ToricCode(2)
    assert code.num_checks == 4
    assert codes.CSSCode.equiv(code, codes.CSSCode([[1, 1, 1, 1]], [[1, 1, 1, 1]]))


def test_distance_css() -> None:
    """Distance calculations for CSS codes."""
    code = codes.HGPCode(codes.RepetitionCode(2, field=3))
    assert code.get_distance(bound=True) == 2
    assert code.get_distance(bound=False) == 2

    # an empty quantum code has distance infinity
    trivial_code = codes.ClassicalCode([[1, 0], [1, 1]])
    code = codes.HGPCode(trivial_code)
    assert code.dimension == 0
    assert code.get_distance(bound=True) is np.nan
    assert code.get_distance(bound=False) is np.nan


def test_stacking_css_codes() -> None:
    """Stack two CSS codes."""
    steane_code = codes.SteaneCode()
    code = codes.CSSCode.stack(steane_code, steane_code)
    assert len(code) == len(steane_code) * 2
    assert code.dimension == steane_code.dimension * 2

    # stacking codes over different fields is not supported
    with pytest.raises(ValueError, match="different fields"):
        qudit_code = codes.SurfaceCode(2, field=3)
        code = codes.CSSCode.stack(steane_code, qudit_code)

    # stacking a CSSCode with a QuditCode yields a QuditCode
    code = codes.CSSCode.stack(steane_code, codes.FiveQubitCode())
    assert not isinstance(code, codes.CSSCode)
