"""Unit tests for quantum.py

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

import io
import unittest.mock

import networkx as nx
import numpy as np
import pytest
import sympy

from qldpc import abstract, codes
from qldpc.objects import ChainComplex, Pauli


def test_small_codes() -> None:
    """Five-qubit and Steane codes."""
    assert codes.SteaneCode().num_qubits == 7

    code = codes.FiveQubitCode(conjugate=[1, 2])
    assert code.num_qubits == 5
    assert code.get_stabilizers()[0] == "X X X X I"


def test_GB_code_error() -> None:
    """Raise error when trying to construct incompatible generalized bicycle codes."""
    matrix_a = [[1, 0], [0, 2]]
    matrix_b = [[0, 1], [1, 0]]
    with pytest.raises(ValueError, match="incompatible"):
        codes.GBCode(matrix_a, matrix_b, field=3)


def test_cyclic_codes() -> None:
    """Bivariate bicycle codes from arXiv:2308.07915 and arXiv:2311.16980."""
    from sympy.abc import x, y, z

    dims: tuple[int, int] | dict[sympy.Symbol, int]

    # last code in Table II of arXiv:2311.16980
    dims = (12, 4)
    poly_a = 1 + y + x * y + x**9
    poly_b = 1 + x**2 + x**7 + x**9 * y**2
    code = codes.BBCode(dims, poly_a, poly_b)
    assert code.num_qudits == 96
    assert code.dimension == 10
    assert code.get_weight() == 8

    # [[144, 12, 12]] code in Table 3 and Figure 2 of arXiv:2308.07915
    dims = {x: 12, y: 6}
    poly_a = x**3 + y + y**2
    poly_b = y**3 + x + x**2
    code = codes.BBCode(dims, poly_a, poly_b)
    assert code.num_qudits == 144
    assert code.dimension == 12
    assert code.get_weight() == 6

    # check that every check qubit addresses its neighboring data qubits
    distance_one_shifts = {(0, 1), (1, 0), (0, -1), (-1, 0)}
    distance_two_shifts = {(0, 2), (2, 0), (0, -2), (-2, 0)} | distance_one_shifts
    for plaquette_map, torus_shape in code.toric_layouts:
        shifts_x, shifts_z = code.get_check_shifts(plaquette_map, torus_shape)
        assert distance_one_shifts.issubset(shifts_x)
        assert distance_one_shifts.issubset(shifts_z)

        shifts_x, shifts_z = code.get_check_shifts(plaquette_map, torus_shape, open_boundaries=True)
        assert distance_two_shifts.issubset(shifts_z)
        assert distance_two_shifts.issubset(shifts_z)

    # check a case with only one symbol
    dims = (36, 2)
    poly_a = 1 + x**9 + x**28 + x**31
    poly_b = 1 + x + x**21 + x**34
    code = codes.BBCode(dims, poly_a, poly_b)
    assert code.orders == (dims[0], 1)

    # check a case with no toric mappings
    dims = (6, 6)
    poly_a = 1 + y + y**2
    poly_b = y**3 + x**2 + x**4
    code = codes.BBCode(dims, poly_a, poly_b)
    assert not code.toric_layouts

    # codes with more than 2 symbols are unsupported
    with pytest.raises(ValueError, match="cannot have more than 2 symbols"):
        codes.BBCode({}, poly_a, x + y + z)

    # fail to match cyclic group orders to free variables
    with pytest.raises(ValueError, match="Could not match"):
        codes.BBCode({}, poly_a, poly_b)


@pytest.mark.parametrize("field", [2, 3])
def test_hypergraph_products(
    field: int,
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (3, 2),
    conjugate: bool = True,
) -> None:
    """Equivalency of matrix-based, graph-based, and chain-based hypergraph products."""
    code_a = codes.ClassicalCode.random(*bits_checks_a, field=field)
    code_b = codes.ClassicalCode.random(*bits_checks_b, field=field)

    code = codes.HGPCode(code_a, code_b, conjugate=conjugate)
    graph = codes.HGPCode.get_graph_product(code_a.graph, code_b.graph, conjugate=conjugate)
    chain = ChainComplex.tensor_product(code_a.matrix, code_b.matrix.T)
    matrix_x, matrix_z = chain.op(1), chain.op(2).T

    assert nx.utils.graphs_equal(code.graph, graph)
    assert np.array_equal(code.matrix, codes.QuditCode.graph_to_matrix(graph))
    assert np.array_equal(code.matrix_x, matrix_x)
    assert np.array_equal(code.matrix_z, matrix_z)


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

    assert np.array_equal(code_HGP.matrix_x, code_LP.matrix_x)
    assert np.array_equal(code_HGP.matrix_z, code_LP.matrix_z)
    assert nx.utils.graphs_equal(code_HGP.graph, code_LP.graph)
    assert np.array_equal(code_HGP.sector_size, code_LP.sector_size)

    chain = ChainComplex.tensor_product(protograph_a, protograph_b.T)
    matrix_x, matrix_z = chain.op(1), chain.op(2).T
    assert isinstance(matrix_x, abstract.Protograph)
    assert isinstance(matrix_z, abstract.Protograph)
    assert np.array_equal(matrix_x.lift(), code_HGP.matrix_x)
    assert np.array_equal(matrix_z.lift(), code_HGP.matrix_z)


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
    code: codes.CSSCode

    # construct check matrix directly
    ring = codes.RingCode(width).matrix
    mat_1 = np.kron(ring, np.eye(width, dtype=int))
    mat_2 = codes.RingCode(num_qudits // 2).matrix
    zero_1 = np.zeros((mat_1.shape[1],) * 2, dtype=int)
    zero_2 = np.zeros((mat_1.shape[0],) * 2, dtype=int)
    zero_3 = np.zeros((mat_2.shape[1],) * 2, dtype=int)
    zero_4 = np.zeros((mat_2.shape[0],) * 2, dtype=int)
    matrix = np.block(
        [
            [zero_1, mat_1.T, -mat_2, zero_4],
            [mat_1, zero_2, zero_3, mat_2.T],
        ]
    )

    # construct lifted product code
    group = abstract.CyclicGroup(num_qudits // 2)
    unit = abstract.Element(group).one()
    shift = abstract.Element(group, group.generators[0])
    element_a = unit - shift**width
    element_b = unit - shift
    code = codes.LPCode([[element_a]], [[element_b]], conjugate=True)
    assert np.array_equal(matrix, code.matrix)

    # same construction with a chain complex
    protograph_a = abstract.Protograph([[element_a]])
    protograph_b = abstract.Protograph([[element_b]])
    chain = ChainComplex.tensor_product(protograph_a, protograph_b.T)
    matrix_x, matrix_z = chain.op(1), chain.op(2).T
    assert isinstance(matrix_x, abstract.Protograph)
    assert isinstance(matrix_z, abstract.Protograph)
    code = codes.CSSCode(matrix_x.lift(), matrix_z.lift(), conjugate=code.conjugated)
    assert np.array_equal(matrix, code.matrix)


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


def test_quantum_tanner() -> None:
    """Quantum Tanner code."""
    # random quantum Tanner code
    group = abstract.CyclicGroup(12)
    subcode = codes.RepetitionCode(4, field=3)
    code = codes.QTCode.random(group, subcode)

    # assert that subgraphs have the right number of nodes, edges, and node degrees
    subgraph_x, subgraph_z = codes.QTCode.get_subgraphs(code.complex)
    group_size = code.complex.graph.number_of_nodes()
    num_faces = code.num_qudits
    for graph in [subgraph_x, subgraph_z]:
        assert graph.number_of_nodes() == num_faces + group_size / 2
        assert graph.number_of_edges() == num_faces * 2
        sources = [node for node in graph.nodes if graph.in_degree(node) == 0]
        assert {graph.out_degree(node) for node in sources} == {subcode.num_bits**2}

    # raise error if the generating data is underspecified
    subset_a = code.complex.subset_a
    subset_b = group.random_symmetric_subset(len(subset_a) - 1)
    subcode_a = codes.RepetitionCode(len(subset_a), field=2)
    with pytest.raises(ValueError, match="Underspecified generating data"):
        codes.QTCode(subset_a, subset_b, subcode_a)

    # raise error if seed codes are over different fields
    subcode_b = codes.RepetitionCode(2, field=3)
    with pytest.raises(ValueError, match="different fields"):
        codes.QTCode([], [], subcode_a, subcode_b)

    # saving and loading a QTCode
    contents = io.StringIO()
    with unittest.mock.patch("builtins.open", return_value=contents):
        # save to a file
        with (
            unittest.mock.patch("os.makedirs", return_value=None),
            unittest.mock.patch.object(contents, "close", lambda: None),
        ):
            code.save("path.txt", "header")

        # fail to load from a file
        with (
            unittest.mock.patch("os.path.isfile", return_value=False),
            pytest.raises(ValueError, match="Path does not exist"),
        ):
            codes.QTCode.load("path.txt")

        # load from a file
        with (
            unittest.mock.patch("os.path.isfile", return_value=True),
            unittest.mock.patch.object(contents, "read", lambda: contents.getvalue()),
        ):
            code_copy = codes.QTCode.load("path.txt")
        assert code_copy == code


def test_toric_tanner_code(size: int = 4) -> None:
    """Rotated toric code as a quantum Tanner code."""
    group = abstract.Group.product(abstract.CyclicGroup(size), repeat=2)
    shift_x, shift_y = group.generators
    subset_a = [shift_x, ~shift_x]
    subset_b = [shift_y, ~shift_y]
    subcode_a = codes.RepetitionCode(2)
    code = codes.QTCode(subset_a, subset_b, subcode_a, bipartite=False)
    assert code.get_code_params() == (size**2, 2, size)
    assert code.get_weight() == 4


def test_surface_codes(rows: int = 3, cols: int = 2, field: int = 3) -> None:
    """Ordinary and rotated surface codes."""

    # "ordinary"/original surface code
    code = codes.SurfaceCode(rows, cols, rotated=False, field=field)
    assert code.dimension == 1
    assert code.num_qudits == rows * cols + (rows - 1) * (cols - 1)
    assert code.get_distance(Pauli.X, bound=10) == cols
    assert code.get_distance(Pauli.Z, bound=10) == rows

    # rotated surface code
    code = codes.SurfaceCode(rows, cols, rotated=True, field=field)
    assert code.dimension == 1
    assert code.num_qudits == rows * cols
    assert code.get_distance(Pauli.X) == codes.CSSCode.get_distance_exact(code, Pauli.X) == cols
    assert code.get_distance(Pauli.Z) == codes.CSSCode.get_distance_exact(code, Pauli.Z) == rows

    # test that the rotated surface code with conjugate=True is an XZZX code
    code = codes.SurfaceCode(max(rows, cols), rotated=True, field=2, conjugate=True)
    for row in code.matrix:
        row_x, row_z = row[: code.num_qudits], row[-code.num_qudits :]
        assert np.count_nonzero(row_x) == np.count_nonzero(row_z)


def test_toric_codes() -> None:
    """Ordinary and rotated toric codes."""

    # "ordinary"/original toric code
    distance = 3
    code = codes.ToricCode(distance, rotated=False)
    assert code.dimension == 2
    assert code.num_qudits == 2 * distance**2

    # check minimal logical operator weights
    code.reduce_logical_ops(with_ILP=True)
    assert (
        {distance}
        == {sum(op) for op in code.get_logical_ops(Pauli.X).view(np.ndarray)}
        == {sum(op) for op in code.get_logical_ops(Pauli.Z).view(np.ndarray)}
    )

    # rotated toric code
    distance = 4
    code = codes.ToricCode(distance, rotated=True)
    assert code.dimension == 2
    assert code.num_qudits == distance**2
    assert codes.CSSCode.get_distance(code) == distance

    # rotated toric code must have even side lengths
    with pytest.raises(ValueError, match="even side lengths"):
        codes.ToricCode(3, rotated=True)

    # rotated toric XZZX code
    rows, cols = 6, 4
    code = codes.ToricCode(rows, cols, rotated=True, conjugate=True)
    for row in code.matrix:
        row_x, row_z = row[: code.num_qudits], row[-code.num_qudits :]
        assert np.count_nonzero(row_x) == np.count_nonzero(row_z)


def test_generalized_surface_codes(size: int = 3) -> None:
    """Multi-dimensional surface and toric codes."""

    # recover ordinary surface code in 2D
    assert np.array_equal(
        codes.GeneralizedSurfaceCode(size, dim=2, periodic=False).matrix,
        codes.SurfaceCode(size, rotated=False).matrix,
    )

    # recover ordinary toric code in 2D
    assert np.array_equal(
        codes.GeneralizedSurfaceCode(size, dim=2, periodic=True).matrix,
        codes.ToricCode(size, rotated=False).matrix,
    )

    for dim in [3, 4]:
        # surface code
        code = codes.GeneralizedSurfaceCode(size, dim, periodic=False)
        assert code.dimension == 1
        assert code.num_qudits == size**dim + (dim - 1) * size ** (dim - 2) * (size - 1) ** 2

        # toric code
        code = codes.GeneralizedSurfaceCode(size, dim, periodic=True)
        assert code.dimension == dim
        assert code.num_qudits == dim * size**dim

    with pytest.raises(ValueError, match=">= 2"):
        codes.GeneralizedSurfaceCode(size, dim=1)
