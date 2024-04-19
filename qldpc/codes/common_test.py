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

import itertools
import unittest.mock

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli, QuditOperator

####################################################################################################
# classical code tests


def test_constructions_classical() -> None:
    """Classical code constructions."""
    code = codes.ClassicalCode.random(5, 3, seed=0)
    assert code.num_bits == 5
    assert "ClassicalCode" in str(code)
    assert code.get_random_word() in code

    code = codes.ClassicalCode.random(5, 3, field=3, seed=0)
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
    assert rep_code.get_distance(bound=True) == bits
    assert rep_code.get_distance(bound=False) == bits
    for vector in itertools.product(rep_code.field.elements, repeat=bits):
        weight = np.count_nonzero(vector)
        dist_bound = rep_code.get_distance_bound(vector=vector)
        dist_exact = rep_code.get_distance_exact(vector=vector)
        assert dist_exact == min(weight, bits - weight)
        assert dist_exact <= dist_bound

    trivial_code = codes.ClassicalCode([[1, 0], [1, 1]])
    random_vector = np.random.randint(2, size=trivial_code.num_bits)
    assert trivial_code.dimension == 0
    assert trivial_code.get_distance_bound() is np.nan
    assert trivial_code.get_distance_exact() is np.nan
    assert trivial_code.get_distance_bound(vector=random_vector) == np.count_nonzero(random_vector)
    assert trivial_code.get_distance_exact(vector=random_vector) == np.count_nonzero(random_vector)


def test_conversions_classical(bits: int = 5, checks: int = 3) -> None:
    """Conversions between matrix and graph representations of a classical code."""
    code = codes.ClassicalCode.random(bits, checks)
    graph = codes.ClassicalCode.matrix_to_graph(code.matrix)
    assert np.array_equal(code.matrix, codes.ClassicalCode.graph_to_matrix(graph))


####################################################################################################
# quantum code tests


def get_random_qudit_code(qudits: int, checks: int, field: int = 2) -> codes.QuditCode:
    """Construct a random (but probably trivial or invalid) QuditCode."""
    return codes.QuditCode(
        codes.ClassicalCode.random(2 * qudits, checks, field).matrix,
        conjugate=(0,),  # conjugate the first qubit
    )


def test_code_string() -> None:
    """Human-readable representation of a code."""
    code = codes.QuditCode([[0]], field=2)
    assert "qubits" in str(code)

    code = codes.QuditCode([[0]], field=3)
    assert "GF(3)" in str(code)

    code = codes.HGPCode(codes.RepetitionCode(2, field=2))
    assert "qubits" in str(code)

    code = codes.HGPCode(codes.RepetitionCode(2, field=3), conjugate=True)
    assert "GF(3)" in str(code) and "conjugated" in str(code)


def test_qubit_code(num_qubits: int = 5, num_checks: int = 3) -> None:
    """Random qubit code."""
    assert get_random_qudit_code(num_qubits, num_checks, field=2).num_qubits == num_qubits
    with pytest.raises(ValueError, match="qubit-only method"):
        assert get_random_qudit_code(num_qubits, num_checks, field=3).num_qubits


def test_qudit_code() -> None:
    """Miscellaneous qudit code tests and coverage."""
    assert codes.FiveQubitCode().dimension == 1


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
    code_b = codes.QuditCode.from_stabilizers(*stabilizers, field=field)
    assert np.array_equal(code_a.matrix, code_b.matrix)
    assert stabilizers == code_b.get_stabilizers()

    with pytest.raises(ValueError, match="different lengths"):
        codes.QuditCode.from_stabilizers("I", "I I", field=field)


def test_deformations(num_qudits: int = 5, num_checks: int = 3, field: int = 3) -> None:
    """Apply Pauli deformations to a qudit code."""
    code = get_random_qudit_code(num_qudits, num_checks, field)
    conjugate = tuple(qubit for qubit in range(num_qudits) if np.random.randint(2))
    transformed_matrix = codes.QuditCode.conjugate(code.matrix, conjugate)

    transformed_matrix = transformed_matrix.reshape(num_checks, 2, num_qudits)
    for node_check, node_qubit, data in code.graph.edges(data=True):
        vals = data[QuditOperator].value
        assert tuple(transformed_matrix[node_check.index, :, node_qubit.index]) == vals


def test_qudit_ops() -> None:
    """Logical operator construction for Galois qudit codes."""
    code: codes.QuditCode

    code = codes.FiveQubitCode()
    logical_ops = code.get_logical_ops()
    assert logical_ops.shape == (2, code.dimension, 2 * code.num_qudits)
    assert np.array_equal(logical_ops[0, 0], [0, 0, 0, 0, 1, 1, 0, 0, 1, 0])
    assert np.array_equal(logical_ops[1, 0], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    assert code.get_logical_ops() is code._full_logical_ops

    code = codes.QuditCode.from_stabilizers(*code.get_stabilizers(), "I I I I I")
    assert np.array_equal(logical_ops, code.get_logical_ops())


####################################################################################################
# CSS code tests


def test_CSS_code() -> None:
    """Miscellaneous CSS code tests and coverage."""
    code_x = codes.ClassicalCode.random(3, 2)

    code_z = ~code_x
    code = codes.CSSCode(code_x, code_z)
    assert code.get_weight() == max(code_x.get_weight(), code_z.get_weight())
    assert code.num_checks_x == code_x.num_checks
    assert code.num_checks_z == code_z.num_checks
    assert code.num_checks == code.num_checks_x + code.num_checks_z

    code_z = codes.ClassicalCode.random(4, 2)
    with pytest.raises(ValueError, match="incompatible"):
        codes.CSSCode(code_x, code_z)

    with pytest.raises(ValueError, match="different fields"):
        code_z = codes.ClassicalCode.random(3, 2, field=code_x.field.order**2)
        codes.CSSCode(code_x, code_z)


def test_CSS_ops() -> None:
    """Logical operator construction for CSS codes."""
    code: codes.CSSCode

    code = codes.HGPCode(codes.ClassicalCode.random(4, 2, field=3))
    code.get_random_logical_op(Pauli.X, ensure_nontrivial=False)
    code.get_random_logical_op(Pauli.X, ensure_nontrivial=True)

    # test that logical operators are dual to each other and have trivial syndromes
    logicals = code.get_logical_ops()
    logicals_x, logicals_z = logicals[0], logicals[1]
    assert np.array_equal(logicals_x @ logicals_z.T, np.eye(code.dimension, dtype=int))
    assert not np.any(code.matrix_z @ logicals_x.T)
    assert not np.any(code.matrix_x @ logicals_z.T)
    assert code.get_logical_ops() is code._logical_ops

    # verify consistency with QuditCode.get_logical_ops
    full_logicals = codes.QuditCode.get_logical_ops(code)
    full_logicals_x, full_logicals_z = full_logicals[0], full_logicals[1]
    assert np.array_equal(full_logicals_x, np.hstack([logicals_x, np.zeros_like(logicals_x)]))
    assert np.array_equal(full_logicals_z, np.hstack([np.zeros_like(logicals_x), logicals_z]))

    # successfullly construct and reduce logical operators in a code with "over-complete" checks
    dist = 4
    code = codes.ToricCode(dist, rotated=True)
    code.reduce_logical_ops()
    assert code.get_code_params() == (dist**2, 2, dist)
    assert not any(np.count_nonzero(op) < dist for op in code.get_logical_ops(Pauli.X))
    assert not any(np.count_nonzero(op) < dist for op in code.get_logical_ops(Pauli.Z))

    # reducing logical operator weight only supported for prime number fields
    code = codes.HGPCode(codes.ClassicalCode.random(4, 2, field=4))
    with pytest.raises(ValueError, match="prime number fields"):
        code.reduce_logical_op(Pauli.X, 0)


def test_distance_quantum() -> None:
    """Distance calculations for CSS codes."""
    code = codes.HGPCode(codes.RepetitionCode(2, field=3))
    assert code.get_distance(bound=True) == 2
    assert code.get_distance(bound=False) == 2

    # assert that the identity is a logical operator
    assert 0 == code.get_distance(Pauli.X, vector=[0] * code.num_qudits)
    assert 0 == code.get_distance(Pauli.X, vector=[0] * code.num_qudits, bound=True)

    # an empty quantum code has distance infinity
    trivial_code = codes.ClassicalCode([[1, 0], [1, 1]])
    code = codes.HGPCode(trivial_code)
    assert code.dimension == 0
    assert code.get_distance(bound=True) is np.nan
    assert code.get_distance(bound=False) is np.nan
