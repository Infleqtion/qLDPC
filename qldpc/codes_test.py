"""Unit tests for codes.py

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

import networkx as nx
import numpy as np
import pytest

from qldpc import abstract, codes


def get_random_qudit_code(qudits: int, checks: int, field: int = 2) -> codes.QuditCode:
    """Construct a random (but probably trivial or invalid) QuditCode."""
    return codes.QuditCode(codes.ClassicalCode.random(2 * qudits, checks, field).matrix)


def test_classical_codes() -> None:
    """Classical code constructions."""
    assert codes.ClassicalCode.random(5, 3).num_bits == 5
    assert codes.HammingCode(3).get_distance() == 3

    num_bits = 2
    for code in [
        codes.RepetitionCode(num_bits, field=3),
        codes.RingCode(num_bits, field=3),
    ]:
        assert code.num_bits == num_bits
        assert code.dimension == 1
        assert code.get_distance() == num_bits
        assert code.get_distance(bound=10) == num_bits
        assert code.get_weight() == 2
        assert code.get_random_word() in code

    # that rank of repetition and Hamming codes is independent of the field
    assert codes.RepetitionCode(3, 2).rank == codes.RepetitionCode(3, 3).rank
    assert codes.HammingCode(3, 2).rank == codes.HammingCode(3, 3).rank

    # invalid classical code construction
    with pytest.raises(ValueError, match="inconsistent"):
        codes.ClassicalCode(codes.ClassicalCode.random(2, 2, field=2), field=3)

    # construct a code from its generator matrix
    code = codes.ClassicalCode.random(5, 3)
    assert code == codes.ClassicalCode.from_generator(code.generator)

    # puncture a code
    assert codes.ClassicalCode.from_generator(code.generator[:, 1:]) == code.puncture(0)

    # shortening a repetition code yields a trivial code
    num_bits = 3
    code = codes.RepetitionCode(num_bits)
    words = [[0] * (num_bits - 1)]
    assert np.array_equal(code.shorten(0).words(), words)


def test_special_codes() -> None:
    """Reed-Solomon, BCH, and Reed-Muller codes."""
    assert codes.ReedSolomonCode(3, 2).dimension == 2

    bits, dimension = 7, 4
    assert codes.BCHCode(bits, dimension).dimension == dimension
    with pytest.raises(ValueError, match=r"2\^m - 1 bits"):
        codes.BCHCode(bits - 1, dimension)

    order, size = 1, 3
    code = codes.ReedMullerCode(order, size)
    assert code.dimension == codes.ClassicalCode(code.matrix).dimension
    assert ~code == codes.ReedMullerCode(size - order - 1, size)

    with pytest.raises(ValueError, match="0 <= r <= m"):
        codes.ReedMullerCode(-1, 0)


def test_named_codes(order: int = 2) -> None:
    """Named codes from the GAP computer algebra system."""
    code = codes.RepetitionCode(order)
    checks = [list(row) for row in code.matrix.view(np.ndarray)]

    with unittest.mock.patch("qldpc.named_codes.get_code", return_value=(checks, None)):
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

    n_a, k_a, d_a, _ = code_a.get_code_params()
    n_b, k_b, d_b, _ = code_b.get_code_params()
    n_ab, k_ab, d_ab, _ = code_ab.get_code_params()
    assert (n_ab, k_ab, d_ab) == (n_a * n_b, k_a * k_b, d_a * d_b)

    with pytest.raises(ValueError, match="Cannot take tensor product"):
        code_b = codes.ClassicalCode.random(*bits_checks_b, field=code_a.field.order**2)
        codes.ClassicalCode.tensor_product(code_a, code_b)


def test_conversions(bits: int = 5, checks: int = 3, field: int = 3) -> None:
    """Conversions between matrix and graph representations of a code."""
    code: codes.ClassicalCode | codes.QuditCode

    code = codes.ClassicalCode.random(bits, checks, field)
    graph = codes.ClassicalCode.matrix_to_graph(code.matrix)
    assert np.array_equal(code.matrix, codes.ClassicalCode.graph_to_matrix(graph))

    code = get_random_qudit_code(bits, checks, field)
    graph = codes.QuditCode.matrix_to_graph(code.matrix)
    assert np.array_equal(code.matrix, codes.QuditCode.graph_to_matrix(graph))


def test_distance_from_classical_code(bits: int = 3) -> None:
    """Distance of a vector from a classical code."""
    rep_code = codes.RepetitionCode(bits, field=2)
    for vector in itertools.product(rep_code.field.elements, repeat=bits):
        weight = np.count_nonzero(vector)
        dist_brute = rep_code.get_distance_exact(vector=vector)
        dist_bound = rep_code.get_distance(bound=True, vector=vector)
        assert dist_brute == min(weight, bits - weight)
        assert dist_brute <= dist_bound
    assert rep_code.get_distance(brute=False) == bits


def test_qubit_code(num_qubits: int = 5, num_checks: int = 3) -> None:
    """Random qubit code."""
    assert get_random_qudit_code(num_qubits, num_checks, field=2).num_qubits == num_qubits
    with pytest.raises(ValueError, match="qubit-only method"):
        assert get_random_qudit_code(num_qubits, num_checks, field=3).num_qubits


def test_CSS_code() -> None:
    """Miscellaneous CSS code tests and coverage."""
    code_x = codes.ClassicalCode.random(3, 2)
    code_z = codes.ClassicalCode.random(4, 2)
    with pytest.raises(ValueError, match="incompatible"):
        codes.CSSCode(code_x, code_z)

    with pytest.raises(ValueError, match="different fields"):
        code_z = codes.ClassicalCode.random(3, 2, field=code_x.field.order**2)
        codes.CSSCode(code_x, code_z)


def test_logical_ops() -> None:
    """Logical operator construction."""
    code = codes.HGPCode(codes.ClassicalCode.random(4, 2, field=3))
    code.get_random_logical_op(codes.Pauli.X, ensure_nontrivial=False)
    code.get_random_logical_op(codes.Pauli.X, ensure_nontrivial=True)

    # test that logical operators are dual to each other and have trivial syndromes
    logicals = code.get_logical_ops()
    logicals_x, logicals_z = logicals[0], logicals[1]
    assert np.array_equal(logicals_x @ logicals_z.T, np.eye(code.dimension, dtype=int))
    assert not np.any(code.code_z.matrix @ logicals_x.T)
    assert not np.any(code.code_x.matrix @ logicals_z.T)

    # minimizing logical operator weight only supported for prime number fields
    code = codes.HGPCode(codes.ClassicalCode.random(4, 2, field=4))
    with pytest.raises(ValueError, match="prime number fields"):
        code.reduce_logical_op(codes.Pauli.X, 0)


def test_deformations(num_qudits: int = 5, num_checks: int = 3, field: int = 3) -> None:
    """Apply Pauli deformations to a qudit code."""
    code = get_random_qudit_code(num_qudits, num_checks, field)
    conjugate = tuple(qubit for qubit in range(num_qudits) if np.random.randint(2))
    transformed_matrix = codes.CSSCode.conjugate(code.matrix, conjugate)

    transformed_matrix = transformed_matrix.reshape(num_checks, 2, num_qudits)
    for node_check, node_qubit, data in code.graph.edges(data=True):
        vals = data[codes.QuditOperator].value
        assert tuple(transformed_matrix[node_check.index, :, node_qubit.index]) == vals


@pytest.mark.parametrize("field", [2, 3])
def test_qudit_stabilizers(field: int, bits: int = 5, checks: int = 3) -> None:
    """Stabilizers of a QuditCode."""
    code_a = get_random_qudit_code(bits, checks, field)
    stabilizers = code_a.get_stabilizers()
    code_b = codes.QuditCode.from_stabilizers(stabilizers, field)
    assert np.array_equal(code_a.matrix, code_b.matrix)
    assert stabilizers == code_b.get_stabilizers()

    with pytest.raises(ValueError, match="different lengths"):
        codes.QuditCode.from_stabilizers(["I", "I I"], field)


@pytest.mark.parametrize("field", [2, 3])
def test_graph_product(
    field: int,
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (3, 2),
    conjugate: bool = True,
) -> None:
    """Equivalency of matrix-based and graph-based hypergraph products."""
    code_a = codes.ClassicalCode.random(*bits_checks_a, field=field)
    code_b = codes.ClassicalCode.random(*bits_checks_b, field=field)

    code = codes.HGPCode(code_a, code_b, conjugate=conjugate)
    graph = codes.HGPCode.get_graph_product(code_a.graph, code_b.graph, conjugate=conjugate)
    assert np.array_equal(code.matrix, codes.QuditCode.graph_to_matrix(graph))
    assert nx.utils.graphs_equal(code.graph, graph)


def test_trivial_lift(
    bits_checks_a: tuple[int, int] = (4, 3),
    bits_checks_b: tuple[int, int] = (3, 2),
    field: int = 3,
) -> None:
    """The lifted product code with a trivial lift reduces to the HGP code."""
    code_a = codes.ClassicalCode.random(*bits_checks_a, field)
    code_b = codes.ClassicalCode.random(*bits_checks_b, field)
    code_HGP = codes.HGPCode(code_a, code_b, field)

    protograph_a = abstract.TrivialGroup.to_protograph(code_a.matrix, field)
    protograph_b = abstract.TrivialGroup.to_protograph(code_b.matrix, field)
    code_LP = codes.LPCode(protograph_a, protograph_b)

    assert np.array_equal(code_HGP.matrix, code_LP.matrix)
    assert nx.utils.graphs_equal(code_HGP.graph, code_LP.graph)
    assert np.array_equal(code_HGP.sector_size, code_LP.sector_size)


def test_lift() -> None:
    """Verify lifting in Eqs. (8) and (10) of arXiv:2202.01702v3."""
    group = abstract.CyclicGroup(3)
    zero = abstract.Element(group)
    x0, x1, x2 = [abstract.Element(group, member) for member in group.generate()]
    base_matrix = [[x1 + x2, x0, zero], [zero, x0 + x1, x1]]

    protograph = abstract.Protograph(base_matrix)
    matrix = [
        [0, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 0],
    ]
    assert np.array_equal(protograph.lift(), matrix)

    # check that the lifted product code is indeed smaller than the HGP code!
    code_HP = codes.HGPCode(matrix)
    code_LP = codes.LPCode(protograph)
    assert code_HP.num_qudits > code_LP.num_qudits
    assert code_HP.num_checks > code_LP.num_checks

    # check total number of qubits
    assert code_HP.sector_size.sum() == code_HP.num_qudits + code_HP.num_checks
    assert code_LP.sector_size.sum() == code_LP.num_qudits + code_LP.num_checks


def test_twisted_XZZX(width: int = 3) -> None:
    """Verify twisted XZZX code in Eqs.(29) and (32) of arXiv:2202.01702v3."""
    num_qudits = 2 * width**2

    # construct check matrix directly
    ring = codes.RingCode(width).matrix
    mat_1 = codes.RingCode(num_qudits // 2).matrix
    mat_2 = np.kron(ring, np.eye(width, dtype=int))
    zero_1 = np.zeros((mat_1.shape[1],) * 2, dtype=int)
    zero_2 = np.zeros((mat_1.shape[0],) * 2, dtype=int)
    zero_3 = np.zeros((mat_2.shape[1],) * 2, dtype=int)
    zero_4 = np.zeros((mat_2.shape[0],) * 2, dtype=int)
    matrix = [
        [mat_1, zero_2, zero_3, mat_2.T],
        [zero_1, -mat_1.T, mat_2, zero_4],
    ]

    # construct lifted product code
    group = abstract.CyclicGroup(num_qudits // 2)
    unit = abstract.Element(group).one()
    shift = abstract.Element(group, group.generators[0])
    element_a = unit - shift**width
    element_b = unit - shift
    code = codes.LPCode([[element_a]], [[element_b]], conjugate=True)
    assert np.array_equal(np.block(matrix), code.matrix)


def test_cyclic_codes(field: int = 3) -> None:
    """Quasi-cyclic codes from arXiv:2308.07915."""
    dims: tuple[int, ...]

    dims = (6, 6)
    terms_a = [("x", 3), ("y", 1), ("y", 2)]
    terms_b = [("y", 3), ("x", 1), ("x", 2)]
    code = codes.QCCode(dims, terms_a, terms_b, field=2)
    assert code.num_qudits == 72
    assert code.dimension == 12
    assert code.get_weight() == 6

    dims = (15, 3)
    terms_a = [("x", 9), ("y", 1), ("y", 2)]
    terms_b = [("x", 0), ("x", 2), ("x", 7)]
    code = codes.QCCode(dims, terms_a, terms_b, field=3)
    assert code.num_qudits == 90
    assert code.dimension == 8
    assert code.get_weight() == 6

    with pytest.raises(ValueError, match="does not match"):
        codes.QCCode((2, 3, 4), terms_a, terms_b, field=2)


def test_GB_code_error() -> None:
    """Raise error when trying to construct incompatible generalized bicycle codes."""
    matrix_a = [[1, 0], [0, 2]]
    matrix_b = [[0, 1], [1, 0]]
    with pytest.raises(ValueError, match="incompatible"):
        codes.GBCode(matrix_a, matrix_b, field=3)


def test_lifted_product_codes() -> None:
    """Lifted product codes in Eq. (5) of arXiv:2308.08648."""
    for lift_dim, matrix in [
        (16, [[0, 0, 0, 0, 0], [0, 2, 4, 7, 11], [0, 3, 10, 14, 15]]),
        (21, [[0, 0, 0, 0, 0], [0, 4, 5, 7, 17], [0, 14, 18, 12, 11]]),
    ]:
        group = abstract.CyclicGroup(lift_dim)
        xx = group.generators[0]
        proto_matrix = [[abstract.Element(group, xx**power) for power in row] for row in matrix]
        protograph = abstract.Protograph(proto_matrix)
        code = codes.LPCode(protograph)
        rate = code.dimension / code.num_qudits
        assert rate >= 2 / 17


def test_tanner_code() -> None:
    """Classical Tanner codes on random regular graphs."""
    subcode = codes.ClassicalCode.random(5, 3)
    subgraph = nx.random_regular_graph(subcode.num_bits, subcode.num_bits * 2 + 2)

    tag = "sort_label"
    for node_a, node_b in subgraph.edges:
        subgraph[node_a][node_b]["sort"] = {node_a: tag, node_b: tag}

    code = codes.TannerCode(subgraph, subcode)
    assert code.num_bits == subgraph.number_of_edges()
    assert code.num_checks == subgraph.number_of_nodes() * code.subcode.num_checks
    assert all(code.subgraph.get_edge_data(*edge)["sort"] == tag for edge in code.subgraph.edges)


def test_toric_tanner_code(size: int = 4) -> None:
    """Rotated toric code as a quantum Tanner code."""
    group = abstract.Group.product(abstract.CyclicGroup(size), repeat=2)
    shift_x, shift_y = group.generators
    subset_a = [shift_x, ~shift_x]
    subset_b = [shift_y, ~shift_y]
    subcode_a = codes.RepetitionCode(2, field=2)
    code = codes.QTCode(subset_a, subset_b, subcode_a, bipartite=False)
    assert code.get_code_params() == (size**2, 2, size, 4)


def test_invalid_quantum_tanner() -> None:
    """Raise error if constructing QTCode with codes over different fields."""
    subcode_a = codes.RepetitionCode(2, field=2)
    subcode_b = codes.RepetitionCode(2, field=3)
    with pytest.raises(ValueError, match="different fields"):
        codes.QTCode([], [], subcode_a, subcode_b)


def test_surface_codes(rows: int = 3, cols: int = 2, field: int = 3) -> None:
    """Ordinary and rotated surface codes."""

    # "ordinary"/original surface code
    code = codes.SurfaceCode(rows, cols, rotated=False, field=field)
    assert code.dimension == 1
    assert code.num_qudits == rows * cols + (rows - 1) * (cols - 1)
    assert code.get_distance(codes.Pauli.X, bound=10) == cols
    assert code.get_distance(codes.Pauli.Z, bound=10) == rows

    # rotated surface code
    code = codes.SurfaceCode(rows, cols, rotated=True, field=field)
    assert code.dimension == 1
    assert code.num_qudits == rows * cols
    assert (
        code.get_distance(codes.Pauli.X)
        == codes.CSSCode.get_distance_exact(code, codes.Pauli.X)
        == cols
    )
    assert (
        code.get_distance(codes.Pauli.Z)
        == codes.CSSCode.get_distance_exact(code, codes.Pauli.Z)
        == rows
    )

    # test that the rotated surface code with conjugate=True is an XZZX code
    code = codes.SurfaceCode(max(rows, cols), rotated=True, field=2, conjugate=True)
    for row in code.matrix:
        row_x, row_z = row[: code.num_qudits], row[-code.num_qudits :]
        assert np.count_nonzero(row_x) == np.count_nonzero(row_z)


def test_toric_codes(field: int = 3) -> None:
    """Ordinary and rotated toric codes."""

    # "ordinary"/original toric code
    rows, cols = 5, 2
    code = codes.ToricCode(rows, cols, rotated=False, field=field)
    assert code.dimension == 2
    assert code.num_qudits == 2 * rows * cols

    # check minimal logical operator weights
    code.reduce_logical_ops(with_ILP=True)
    assert (
        {rows, cols}
        == {sum(op) for op in code.get_logical_ops(codes.Pauli.X).view(np.ndarray)}
        == {sum(op) for op in code.get_logical_ops(codes.Pauli.Z).view(np.ndarray)}
    )

    # rotated toric code
    distance = 4
    code = codes.ToricCode(distance, rotated=True, field=field)
    assert code.dimension == 2
    assert code.num_qudits == distance**2
    assert codes.CSSCode.get_distance(code) == distance

    # rotated toric XZZX code
    rows, cols = 6, 4
    code = codes.ToricCode(rows, cols, rotated=True, field=field, conjugate=True)
    for row in code.matrix:
        row_x, row_z = row[: code.num_qudits], row[-code.num_qudits :]
        assert np.count_nonzero(row_x) == np.count_nonzero(row_z)

    # rotated toric code must have even side lengths
    with pytest.raises(ValueError, match="must have even side lengths"):
        codes.ToricCode(3, rotated=True)


def test_quantum_distance(field: int = 2) -> None:
    """Distance calculations for qudit codes."""
    code = codes.HGPCode(codes.RepetitionCode(2, field=field))
    assert code.get_distance() == 2

    # assert that the identity is a logical operator
    assert 0 == code.get_distance(codes.Pauli.X, vector=[0] * code.num_qudits)
    assert 0 == code.get_distance(codes.Pauli.X, vector=[0] * code.num_qudits, bound=True)
