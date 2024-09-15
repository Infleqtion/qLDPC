"""Quantum error-correcting codes

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

import ast
import functools
import itertools
import os
from collections.abc import Collection, Sequence

import networkx as nx
import numpy as np
import numpy.typing as npt
import sympy

from qldpc import abstract
from qldpc.objects import (
    CayleyComplex,
    ChainComplex,
    Node,
    Pauli,
    QuditOperator,
)

from .classical import HammingCode, RepetitionCode, RingCode, TannerCode
from .common import ClassicalCode, CSSCode, QuditCode


class FiveQubitCode(QuditCode):
    """Smallest quantum error-correcting code."""

    def __init__(self, *, conjugate: slice | Sequence[int] | None = ()) -> None:
        code = QuditCode.from_stabilizers("X Z Z X I", "I X Z Z X", "X I X Z Z", "Z X I X Z")
        QuditCode.__init__(self, code, conjugate=conjugate)


class SteaneCode(CSSCode):
    """Smallest quantum error-correcting CSS code."""

    def __init__(self, *, conjugate: slice | Sequence[int] | None = ()) -> None:
        code = HammingCode(3)
        CSSCode.__init__(self, code, code, conjugate=conjugate)


################################################################################
# two-block and bicycle codes


class TBCode(CSSCode):
    """Two-block (TB) code.

    A TBCode code is built out of two matrices A and B, which are combined as
    - matrix_x = [A, B], and
    - matrix_z = [B.T, -A.T],
    to form the parity check matrices of a CSSCode.  If A and B commute, the parity check matrices
    matrix_x and matrix_z satisfy the requirements of a CSSCode.

    Two-block codes constructed out of circulant matrices are known as generalized bicycle codes.
    """

    def __init__(
        self,
        matrix_a: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        matrix_b: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
        *,
        conjugate: slice | Sequence[int] = (),
        promise_balanced_codes: bool = False,
        skip_validation: bool = False,
    ) -> None:
        """Construct a two-block quantum code."""
        matrix_a = ClassicalCode(matrix_a, field).matrix
        matrix_b = ClassicalCode(matrix_b, field).matrix
        if not skip_validation and not np.array_equal(matrix_a @ matrix_b, matrix_b @ matrix_a):
            raise ValueError("The matrices provided for this TBCode are incompatible")

        matrix_x = np.block([matrix_a, matrix_b])
        matrix_z = np.block([matrix_b.T, -matrix_a.T])
        CSSCode.__init__(
            self,
            matrix_x,
            matrix_z,
            field,
            conjugate=conjugate,
            promise_balanced_codes=promise_balanced_codes,
            skip_validation=True,
        )


class GBCode(TBCode):
    """Generalized bicycle code.

    A generalized bicycle code is a CSS code with subcode parity check matrices
    - matrix_x = [A, B], and
    - matrix_z = [B.T, -A.T].
    Here A and B are polynomials of the form A = sum_{i,j,k,...} A_{ijk...} x^i y^j z^k ...,
    where
    - A_{ijk...} is a scalar coefficient (over some finite field),
    - x, y, z, ... are generators of cyclic groups of orders R_x, R_y, R_z, ...
    - the monomial x^i y^j z^k ... represents by a tensor product of cyclic shift matrices.

    A generalized bicycle code is defined by...
    [1] a sequence of cyclic group orders, and
    [2] two sympy polynomials.
    By default, group orders are associated in lexicographic order with free variables of the
    polynomials.  Group orders can also be assigned to variables explicitly with a dictionary.
    """

    def __init__(
        self,
        orders: Sequence[int] | dict[sympy.Symbol, int],
        poly_a: sympy.Basic,
        poly_b: sympy.Basic,
        field: int | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Construct a generalized bicycle code."""
        self.poly_a = sympy.Poly(poly_a)
        self.poly_b = sympy.Poly(poly_b)

        # identify the symbols used to denote cyclic group generators
        symbols = poly_a.free_symbols | poly_b.free_symbols
        if len(orders) < len(symbols):
            raise ValueError(f"Provided {len(symbols)} symbols, but only {len(orders)} orders.")

        # identify cyclic group orders with symbols in the polynomials
        if isinstance(orders, dict):
            symbol_to_order = orders.copy()
        else:
            symbol_to_order = {}
            for symbol, order in zip(sorted(symbols, key=str), orders):
                assert isinstance(symbol, sympy.Symbol), f"Invalid symbol: {symbol}"
                symbol_to_order[symbol] = order

        # add more placeholder symbols if necessay
        while len(symbol_to_order) < len(orders):
            unique_symbol = sympy.Symbol("~" + "".join(map(str, symbols)))
            symbol_to_order[unique_symbol] = orders[len(symbol_to_order)]

        self.symbols = tuple(symbol_to_order.keys())
        self.orders = tuple(symbol_to_order.values())

        # identify the group generator associated with each symbol
        self.group = abstract.AbelianGroup(*self.orders, product_lift=True)
        self.gens = self.group.generators
        self.symbol_gens = dict(zip(self.symbols, self.gens))

        # if requested, hadamard-transform half of the qudits
        qudits_to_conjugate: slice | Sequence[int] = (
            slice(self.group.order, 2 * self.group.order + 1) if conjugate else ()
        )

        # build defining matrices of a generalized bicycle code
        matrix_a = self.eval(self.poly_a).lift()
        matrix_b = self.eval(self.poly_b).lift()
        TBCode.__init__(
            self,
            matrix_a,
            matrix_b,
            field,
            conjugate=qudits_to_conjugate,
            promise_balanced_codes=True,
            skip_validation=True,
        )

    def eval(
        self,
        expr: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul | sympy.Poly,
    ) -> abstract.Element:
        """Convert a sympy expression into an element of this code's group algebra."""
        # evaluate simple cases
        if isinstance(expr, sympy.Integer):
            return int(expr) * abstract.Element(self.group, self.group.identity)
        if isinstance(expr, sympy.Symbol):
            return abstract.Element(self.group, self.symbol_gens[expr])
        if isinstance(expr, (sympy.Symbol, sympy.Pow)):
            base, exp = expr.as_base_exp()
            return abstract.Element(self.group, self.symbol_gens[base] ** exp)

        # evaluate a product or polynomial
        element = abstract.Element(self.group)
        for term in expr.as_expr().args:
            element += functools.reduce(
                abstract.Element.__mul__,
                [self.eval(factor) for factor in term.as_ordered_factors()],
            )
        return element


# TODO: example notebook featuring this code
class BBCode(GBCode):
    """Bivariate bicycle codes from arXiv:2308.07915.

    A bivariate bicycle code is a CSS code with subcode parity check matrices
    - matrix_x = [A, B], and
    - matrix_z = [B.T, -A.T].
    Here A and B are polynomials of the form A = sum_{i,j} A_{ij} x^i y^j, where
    - A_{ij} is a scalar coefficient (over some finite field),
    - x and y are, respectively, generators of cyclic groups of orders R_x, R_y, and
    - the monomial x^i y^j represents a tensor product of cyclic shift matrices.

    A generalized bicycle code is defined by...
    [1] two cyclic group orders, and
    [2] two sympy polynomials.
    By default, group orders are associated in lexicographic order with free variables of the
    polynomials.  Group orders can also be assigned to variables explicitly with a dictionary.

    All qubits of a BBCode can be organized into plaquettes of four qubits that look like:
        L X
        Z R
    where L and R are data qubits, and X and Z are check qubits.  More specifically:
    - L and R data qubits are addressed by the left and right halves of matrix_x (or matrix_z).
    - X check qubits measure X-type parity checks, and are associated with rows of matrix_x.
    - Z check qubits measure Z-type parity checks, and are associated with rows of matrix_z.
    These plaquettes are arranged into a rectangle R_x plaquettes wide and R_y plaquettes tall.

    The the connections between ancilla and data qubits can be read off of the polynomials A and B.
    For example, if A_{ij} != 0, then every X-type parity check addresses an L-type qubit that is
    (i, j) plaquettes (right, up), assuming periodic boundary conditions, and every Z-type parity
    check addresses an R-type qubit that is (i, j) plaquettes (left, down).  The polynomial B
    similarly indicates where X-type (Z-type) checks address R-type (L-type) qubits.
    """

    def __init__(
        self,
        orders: Sequence[int] | dict[sympy.Symbol, int],
        poly_a: sympy.Basic,
        poly_b: sympy.Basic,
        field: int | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Construct a bivariate bicycle code."""
        self.poly_a = sympy.Poly(poly_a)
        self.poly_b = sympy.Poly(poly_b)
        symbols = poly_a.free_symbols | poly_b.free_symbols
        if len(orders) != 2 or len(symbols) != 2:
            raise ValueError(
                "BBCodes should have exactly two cyclic group orders and two symbols, not "
                f"{len(orders)} orders and {len(symbols)} symbols."
            )
        GBCode.__init__(self, orders, poly_a, poly_b, field, conjugate=conjugate)


################################################################################
# hypergraph and lifted product codes


class HGPCode(CSSCode):
    """Hypergraph product (HGP) code.

    A hypergraph product code AB is constructed from two classical codes, A and B.

    Consider the following:
    - Code A has 3 data and 2 check bits.
    - Code B has 4 data and 3 check bits.
    We represent data bits/qudits by circles (○) and check bits/qudits by squares (□).

    Denode the Tanner graph of code C by G_C.  The nodes of G_AB can be arranged into a matrix.  The
    rows of this matrix are labeled by nodes of G_A, and columns by nodes of G_B.  The matrix of
    nodes in G_AB can thus be organized into four sectors:

    ――――――――――――――――――――――――――――――――――
      | ○ ○ ○ ○ | □ □ □ ← nodes of G_B
    ――+―――――――――+――――――
    ○ | ○ ○ ○ ○ | □ □ □
    ○ | ○ ○ ○ ○ | □ □ □
    ○ | ○ ○ ○ ○ | □ □ □
    ――+―――――――――+――――――
    □ | □ □ □ □ | ○ ○ ○
    □ | □ □ □ □ | ○ ○ ○
    ↑ nodes of G_A
    ――――――――――――――――――――――――――――――――――

    We identify each sector by two bits.
    In the example above:
    - sector (0, 0) has 3×4=12 data qudits
    - sector (0, 1) has 3×3=9 check qudits
    - sector (1, 0) has 2×4=8 check qudits
    - sector (1, 1) has 2×3=6 data qudits

    Edges in G_AB are inherited across rows/columns from G_A and G_B.  For example, if rows r_1 and
    r_2 share an edge in G_A, then the same is true in every column of G_AB.

    By default, the check qudits in sectors (1, 0) of G_AB measure X-type operators.  Likewise with
    sector (0, 1) and Z-type operators.  If a HGP is constructed with `conjugate==True`, then the
    types of operators addressing the nodes in sector (1, 1) are switched.

    This class contains two equivalent constructions of an HGPCode:
    - A construction based on Tanner graphs (as discussed above).
    - A construction based on check matrices, taken from arXiv:2202.01702.
    The latter construction is less intuitive, but more efficient.

    References:
    - https://errorcorrectionzoo.org/c/hypergraph_product
    - https://arxiv.org/abs/2202.01702
    - https://www.youtube.com/watch?v=iehMcUr2saM
    - https://arxiv.org/abs/0903.0566
    - https://arxiv.org/abs/1202.0928
    """

    sector_size: npt.NDArray[np.int_]

    def __init__(
        self,
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Hypergraph product of two classical codes, as in arXiv:2202.01702.

        The parity check matrices of the hypergraph product code are:

        matrix_x = [ H1 ⨂ In2, Im1 ⨂ H2.T]
        matrix_z = [-In1 ⨂ H2, H1.T ⨂ Im2]

        Here (H1, H2) == (matrix_a, matrix_b), and I[m/n][1/2] are identity matrices,
        with (m1, n1) = H1.shape and (m2, n2) = H2.shape.

        A minus sign in one sector of matrix_x or matrix_z is necessary to satisfy CSS code
        requirements with nonbinary fields.  The placement of this sign is chosen for consistency
        with the tensor product of chain complexes.
        """
        if code_b is None:
            code_b = code_a
        code_a = ClassicalCode(code_a, field)
        code_b = ClassicalCode(code_b, field)
        field = code_a.field.order

        # use a matrix-based hypergraph product to identify X-sector and Z-sector parity checks
        matrix_x, matrix_z = HGPCode.get_matrix_product(code_a.matrix, code_b.matrix)

        # identify the number of qudits in each sector
        self.sector_size = np.outer(
            [code_a.num_bits, code_a.num_checks],
            [code_b.num_bits, code_b.num_checks],
        )

        # identify which qudits to conjugate (Hadamard-transform)
        qudits_to_conjugate = slice(self.sector_size[0, 0], None) if conjugate else None

        CSSCode.__init__(
            self,
            matrix_x.astype(int),
            matrix_z.astype(int),
            field,
            conjugate=qudits_to_conjugate,
            skip_validation=True,
        )

    @classmethod
    def get_matrix_product(
        cls,
        matrix_a: npt.NDArray[np.int_ | np.object_],
        matrix_b: npt.NDArray[np.int_ | np.object_],
    ) -> tuple[npt.NDArray[np.int_ | np.object_], npt.NDArray[np.int_ | np.object_]]:
        """Hypergraph product of two parity check matrices."""
        # construct the nontrivial blocks of the final parity check matrices
        mat_H1_In2 = np.kron(matrix_a, np.eye(matrix_b.shape[1], dtype=int))
        mat_In1_H2 = np.kron(np.eye(matrix_a.shape[1], dtype=int), matrix_b)
        mat_H1_Im2_T = np.kron(matrix_a.T, np.eye(matrix_b.shape[0], dtype=int))
        mat_Im1_H2_T = np.kron(np.eye(matrix_a.shape[0], dtype=int), matrix_b.T)

        # construct the X-sector and Z-sector parity check matrices
        matrix_x = np.block([mat_H1_In2, mat_Im1_H2_T])
        matrix_z = np.block([-mat_In1_H2, mat_H1_Im2_T])
        return matrix_x, matrix_z

    @classmethod
    def get_graph_product(
        cls, graph_a: nx.DiGraph, graph_b: nx.DiGraph, *, conjugate: bool = False
    ) -> nx.DiGraph:
        """Hypergraph product of two Tanner graphs."""

        # start with a cartesian products of the input graphs
        graph_product = nx.cartesian_product(graph_a, graph_b)

        # fix edge orientation, and tag each edge with a QuditOperator
        graph = nx.DiGraph()
        for node_fst, node_snd, data in graph_product.edges(data=True):
            # identify the sectors of two nodes
            sector_fst = cls.get_sector(*node_fst)
            sector_snd = cls.get_sector(*node_snd)

            # identify data-qudit vs. check nodes, and their sectors
            if sector_fst in [(0, 0), (1, 1)]:
                node_qudit, sector_qudit = node_fst, sector_fst
                node_check, sector_check = node_snd, sector_snd
            else:
                node_check, sector_check = node_fst, sector_fst
                node_qudit, sector_qudit = node_snd, sector_snd

            # start with an X-type operator
            op = QuditOperator((data.get("val", 1), 0))

            # switch to Z-type operator for check qudits in the (0, 1) sector
            if sector_check == (0, 1):
                op = ~op

            # account for the minus sign in the (0, 0) sector of the Z-type subcode
            if op.value[Pauli.Z] and sector_qudit == (0, 0):
                op = -op

            # for a conjugated code, flip X <--> Z operators in the (1, 1) sector
            if conjugate and sector_qudit == (1, 1):
                op = ~op

            graph.add_edge(node_check, node_qudit)
            graph[node_check][node_qudit][QuditOperator] = op

        # relabel nodes, from (node_a, node_b) --> node_combined
        node_map = HGPCode.get_product_node_map(graph_a.nodes, graph_b.nodes)
        graph = nx.relabel_nodes(graph, node_map)

        # remember order of the field, and use Pauli operators if appropriate
        if hasattr(graph_a, "order"):
            graph.order = graph_a.order
            if graph.order == 2:
                for _, __, data in graph.edges(data=True):
                    data[Pauli] = Pauli(data[QuditOperator].value)
                    del data[QuditOperator]

        return graph

    @classmethod
    def get_sector(cls, node_a: Node, node_b: Node) -> tuple[int, int]:
        """Get the sector of a node in a graph product."""
        return int(not node_a.is_data), int(not node_b.is_data)

    @classmethod
    def get_product_node_map(
        cls, nodes_a: Collection[Node], nodes_b: Collection[Node]
    ) -> dict[tuple[Node, Node], Node]:
        """Map (dictionary) that re-labels nodes in the hypergraph product of two codes."""
        num_qudits_a = sum(node.is_data for node in nodes_a)
        num_qudits_b = sum(node.is_data for node in nodes_b)
        num_checks_a = len(nodes_a) - num_qudits_a
        num_checks_b = len(nodes_b) - num_qudits_b
        sector_shapes = [
            [(num_qudits_a, num_qudits_b), (num_qudits_a, num_checks_b)],
            [(num_checks_a, num_qudits_b), (num_checks_a, num_checks_b)],
        ]

        node_map = {}
        for node_a, node_b in itertools.product(sorted(nodes_a), sorted(nodes_b)):
            # identify sector and whether this is a data vs. check qudit
            sector = cls.get_sector(node_a, node_b)
            is_data = sector in [(0, 0), (1, 1)]

            # identify node index
            sector_shape = sector_shapes[sector[0]][sector[1]]
            index = int(np.ravel_multi_index((node_a.index, node_b.index), sector_shape))
            if sector == (1, 1):
                index += num_qudits_a * num_qudits_b
            if sector == (0, 1):
                index += num_checks_a * num_qudits_b

            node_map[node_a, node_b] = Node(index=index, is_data=is_data)

        return node_map


class LPCode(CSSCode):
    """Lifted product (LP) code.

    A lifted product code is essentially the same as a hypergraph product code, except that the
    parity check matrices are "protographs", or matrices whose entries are members of a group
    algebra over a finite field F_q.  Each of these entries can be "lifted" to a representation as
    orthogonal matrices over F_q, in which case the protograph is interpreted as a block matrix;
    this is called "lifting" the protograph.

    Notes:
    - A lifted product code with protographs of size 1×1 is a two-block code (more specifically, a
        two-block group-algebra code).  If the base group of the protographs is a cyclic group, the
        resulting lifted product code is a generalized bicycle code.
    - A lifted product code with protographs whose entries get lifted to 1×1 matrices is a
        hypergraph product code built from the lifted protographs.
    - One way to get an LPCode: take a classical code with parity check matrix H and multiply it by
        a diagonal matrix D = diag(a_1, a_2, ... a_n), where all {a_j} are elements of a group
        algebra.  The protograph P = H @ D can then be used for one of the protographs of an LPCode.

    References:
    - https://errorcorrectionzoo.org/c/lifted_product
    - https://arxiv.org/abs/2202.01702
    - https://arxiv.org/abs/2012.04068
    - https://arxiv.org/abs/2306.16400
    """

    def __init__(
        self,
        protograph_a: npt.NDArray[np.object_] | Sequence[Sequence[object]],
        protograph_b: npt.NDArray[np.object_] | Sequence[Sequence[object]] | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        """Lifted product of two protographs, as in arXiv:2012.04068."""
        if protograph_b is None:
            protograph_b = protograph_a
        protograph_a = abstract.Protograph(protograph_a)
        protograph_b = abstract.Protograph(protograph_b)
        field = protograph_a.field.order

        # identify X-sector and Z-sector parity checks
        matrix_x, matrix_z = HGPCode.get_matrix_product(protograph_a, protograph_b)

        # identify the number of qudits in each sector
        self.sector_size = protograph_a.group.lift_dim * np.outer(
            protograph_a.shape[::-1],
            protograph_b.shape[::-1],
        )

        # identify which qudits to conjugate (Hadamard-transform)
        qudits_to_conjugate = slice(self.sector_size[0, 0], None) if conjugate else None

        CSSCode.__init__(
            self,
            abstract.Protograph(matrix_x.astype(object)).lift(),
            abstract.Protograph(matrix_z.astype(object)).lift(),
            field,
            conjugate=qudits_to_conjugate,
            skip_validation=True,
        )


################################################################################
# quantum Tanner code


# TODO: example notebook featuring this code
class QTCode(CSSCode):
    """Quantum Tanner code: a CSS code for qudits defined on the faces of a Cayley complex.

    Altogether, a quantum Tanner code is defined by:
    - two symmetric (self-inverse) subsets A and B of a group G, and
    - two classical codes C_A and C_B, respectively with block lengths |A| and |B|.

    The qudits of a quantum Tanner code live on the faces of a Cayley complex built out of A and B.
    Each face of the Cayley complex looks like:

         g ―――――――――― gb

         |  f(g,a,b)  |

        ag ――――――――― agb

    where (g,a,b) is an element of (G,A,B), and f(g,a,b) = {g, ab, gb, agb}.  We define two
    (directed) subgraphs on the Cayley complex:
    - subgraph_x with edges ( g, f(g,a,b)), and
    - subgraph_z with edges (ag, f(g,a,b)).

    The X-type parity checks of a quantum Tanner code are then given by the classical Tanner code on
    subgraph_x with subcode ~(C_A ⨂ C_B), where ~C is the dual code to C.  Z-type parity checks are
    similarly given by the classical Tanner code on subgraph_z with subcode ~(~C_A ⨂ ~C_B).

    Notes:
    - "Good" quantum Tanner code: projective special linear group and random classical codes.

    References:
    - https://errorcorrectionzoo.org/c/quantum_tanner
    - https://arxiv.org/abs/2206.07571
    - https://arxiv.org/abs/2202.13641
    """

    complex: CayleyComplex

    def __init__(
        self,
        subset_a: Collection[abstract.GroupMember],
        subset_b: Collection[abstract.GroupMember],
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        bipartite: bool = False,
        conjugate: slice | Sequence[int] | None = (),
    ) -> None:
        """Construct a quantum Tanner code."""
        code_a = ClassicalCode(code_a, field)
        if code_b is not None:
            code_b = ClassicalCode(code_b, field)
        elif len(subset_a) == len(subset_b):
            code_b = ~code_a
        else:
            raise ValueError(
                "Underspecified generating data for quantum Tanner code:\n"
                "no seed code provided for one of the generating subsets"
            )

        if field is None and code_a.field is not code_b.field:
            raise ValueError("The sub-codes provided for this QTCode are over different fields")

        self.code_a = code_a
        self.code_b = code_b
        self.complex = CayleyComplex(subset_a, subset_b, bipartite=bipartite)
        code_x, code_z = self.get_subcodes(self.complex, code_a, code_b)
        CSSCode.__init__(self, code_x, code_z, field, conjugate=conjugate, skip_validation=True)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QTCode)
            and other.code_a == other.code_a
            and other.code_b == other.code_b
            and other.complex.subset_a == other.complex.subset_a
            and other.complex.subset_b == other.complex.subset_b
            and other.complex.bipartite == other.complex.bipartite
            and other.conjugated == other.conjugated
        )

    @classmethod
    def get_subcodes(
        cls, cayplex: CayleyComplex, code_a: ClassicalCode, code_b: ClassicalCode
    ) -> tuple[TannerCode, TannerCode]:
        """Get the classical Tanner subcodes of a quantum Tanner code."""
        subgraph_x, subgraph_z = QTCode.get_subgraphs(cayplex)
        subcode_x = ~ClassicalCode.tensor_product(code_a, code_b)
        subcode_z = ~ClassicalCode.tensor_product(~code_a, ~code_b)
        return TannerCode(subgraph_x, subcode_x), TannerCode(subgraph_z, subcode_z)

    @classmethod
    def get_subgraphs(cls, cayplex: CayleyComplex) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Build the subgraphs of the inner (classical) Tanner codes for a quantum Tanner code.

        These subgraphs are defined using the faces of a Cayley complex.  Each face looks like:

         g ―――――――――― gb

         |  f(g,a,b)  |

        ag ――――――――― agb

        where f(g,a,b) = {g, ab, gb, agb}.  Specifically, the (directed) subgraphs are:
        - subgraph_x with edges ( g, f(g,a,b)), and
        - subgraph_z with edges (ag, f(g,a,b)).
        Classical Tanner codes on these subgraphs are used as to construct quantum Tanner code.

        As a matter of practice, defining classical Tanner codes on subgraph_x and subgraph_z
        requires choosing an ordering on the edges incident to every source node of these graphs.
        If the group G is equipped with a total order, a natural ordering of edges incident to every
        source node is induced by assigning the label (a, b) to edge (g, f(g,a,b)).  Consistency
        then requires that edge (ag, f(g,a,b)) has label (a^-1, b), as verified by defining g' = ag
        and checking that f(g,a,b) = f(g',a^-1,b).
        """
        subset_a = cayplex.cover_subset_a
        subset_b = cayplex.cover_subset_b

        # identify the identity element
        member = next(iter(subset_a))
        identity = member * ~member

        # identify the set of nodes for which we still need to add faces
        nodes_to_add = set([identity])

        # build the subgraphs one node at a time
        subgraph_x = nx.DiGraph()
        subgraph_z = nx.DiGraph()
        while nodes_to_add:
            gg = nodes_to_add.pop()

            # identify nodes we have already covered, and new nodes we may need to cover
            old_nodes = set(subgraph_x.nodes())
            new_nodes = set()

            # add all faces adjacent to this node
            for aa, bb in itertools.product(subset_a, subset_b):
                aa_gg, gg_bb = aa * gg, gg * bb
                aa_gg_bb = aa_gg * bb
                face = frozenset([gg, aa_gg, gg_bb, aa_gg_bb])
                subgraph_x.add_edge(gg, face, sort=(aa, bb))
                subgraph_z.add_edge(aa_gg, face, sort=(~aa, bb))

                new_nodes.add(aa_gg_bb)

            nodes_to_add |= new_nodes - old_nodes

        return subgraph_x, subgraph_z

    @classmethod
    def random(
        cls,
        group: abstract.Group,
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | None = None,
        *,
        bipartite: bool = False,
        conjugate: slice | Sequence[int] | None = (),
        one_subset: bool = False,
        seed: int | None = None,
    ) -> QTCode:
        """Construct a random quantum Tanner code from a base group and seed code(s).

        If only one code C is provided, use its dual ~C for the second code.
        """
        code_a = ClassicalCode(code_a, field)
        code_b = ClassicalCode(code_b if code_b is not None else ~code_a, field)
        subset_a = group.random_symmetric_subset(code_a.num_bits, seed=seed)
        subset_b = group.random_symmetric_subset(code_b.num_bits) if not one_subset else subset_a
        return QTCode(subset_a, subset_b, code_a, code_b, bipartite=bipartite, conjugate=conjugate)

    def save(self, path: str, *headers: str) -> None:
        """Save the generating data of this code to a file."""
        # convert subsets to arrays
        subset_a = np.array([gen.array_form for gen in self.complex.subset_a])
        subset_b = np.array([gen.array_form for gen in self.complex.subset_b])

        # create save directory if necessary
        save_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(save_dir, exist_ok=True)

        with open(path, "w") as file:
            # write provided headers
            for header in headers:
                for line in header.splitlines():
                    file.write(f"# {line}\n")

            # write subsets
            file.write("# subset_a:\n")
            np.savetxt(file, subset_a, fmt="%d")
            file.write("# subset_b:\n")
            np.savetxt(file, subset_b, fmt="%d")

            # write seed codes
            file.write("# code_a.matrix:\n")
            np.savetxt(file, self.code_a.matrix, fmt="%d")
            file.write("# code_b.matrix:\n")
            np.savetxt(file, self.code_b.matrix, fmt="%d")

            # write other data
            file.write(f"# base field: {self.field.order}\n")
            file.write(f"# bipartite: {self.complex.bipartite}\n")
            file.write(f"# conjugate: {self.conjugated}\n")

    @classmethod
    def load(cls, path: str) -> QTCode:
        """Load a QTCode from a file."""
        if not os.path.isfile(path):
            raise ValueError(f"Path does not exist: {path}")

        with open(path, "r") as file:
            lines = file.read().splitlines()

        # load miscellaneous data
        field = ast.literal_eval(lines[-3].split(":")[-1])
        bipartite = ast.literal_eval(lines[-2].split(":")[-1])
        conjugate = ast.literal_eval(lines[-1].split(":")[-1])

        # load integer arrays separated by comments
        arrays = []
        last_index = 0
        for index, line in enumerate(lines):
            if line.startswith("#"):
                if index > last_index + 1:
                    array = np.genfromtxt(lines[last_index + 1 : index], dtype=int, ndmin=2)
                    arrays.append(array)
                last_index = index

        # construct subsets and generating codes
        subset_a = set(abstract.GroupMember(gen) for gen in arrays[0])
        subset_b = set(abstract.GroupMember(gen) for gen in arrays[1])
        code_a = ClassicalCode(arrays[2], field)
        code_b = ClassicalCode(arrays[3], field)
        return QTCode(subset_a, subset_b, code_a, code_b, bipartite=bipartite, conjugate=conjugate)


################################################################################
# common quantum codes


class SurfaceCode(CSSCode):
    """The one and only!

    Actually, there are two variants: "ordinary" and "rotated" surface codes.
    The rotated code is more qubit-efficient.

    If constructed with conjugate=True, every other qubit is Hadamard-transformed in a checkerboard
    pattern.  The rotated surface code with conjugate=True is the XZZX code in arXiv:2009.07851.
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        rotated: bool = True,
        field: int | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        if cols is None:
            cols = rows

        # save known distances
        self._exact_distance_x = cols
        self._exact_distance_z = rows

        # which qubits should be Hadamard-transformed?
        qudits_to_conjugate: slice | Sequence[int] | None

        if rotated:
            # rotated surface code
            matrix_x, matrix_z = SurfaceCode.get_rotated_checks(rows, cols)

            if conjugate:
                # Hadamard-transform qubits in a checkerboard pattern
                qudits_to_conjugate = [
                    idx for idx, (row, col) in enumerate(np.ndindex(rows, cols)) if (row + col) % 2
                ]

            else:
                qudits_to_conjugate = None

        else:
            # "original" surface code
            code_a = RepetitionCode(rows, field)
            code_b = RepetitionCode(cols, field)
            code_ab = HGPCode(code_a, code_b, field, conjugate=conjugate)
            matrix_x = code_ab.matrix_x
            matrix_z = code_ab.matrix_z
            qudits_to_conjugate = code_ab.conjugated

        CSSCode.__init__(
            self,
            matrix_x,
            matrix_z,
            field=field,
            conjugate=qudits_to_conjugate,
            skip_validation=True,
        )

    @classmethod
    def get_rotated_checks(
        cls, rows: int, cols: int
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Build X-sector and Z-sector parity check matrices.

        Example 5x5 rotated surface code layout:

             ―――     ―――
            | ⋅ |   | ⋅ |
            ○―――○―――○―――○―――○―――
            | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○―――
        | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○―――
            | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○―――
        | × | ⋅ | × | ⋅ | × |
         ―――○―――○―――○―――○―――○
                | ⋅ |   | ⋅ |
                 ―――     ―――

        Here:
        - Circles (○) denote data qubits (of which there are 5×5 = 25 total).
        - Tiles with a cross (×) denote X-type parity checks (12 total).
        - Tiles with a dot (⋅) denote Z-type parity checks (12 total).

        Reference: https://errorcorrectionzoo.org/c/rotated_surface
        """

        def get_check(
            row_indices: Sequence[int], col_indices: Sequence[int]
        ) -> npt.NDArray[np.int_]:
            """Check on the qubits with the given indices, dropping any that are out of bounds."""
            check = np.zeros((rows, cols), dtype=int)
            for row, col in zip(row_indices, col_indices):
                if 0 <= row < rows and 0 <= col < cols:
                    check[row, col] = 1
            return check.ravel()

        checks_x = []
        checks_z = []
        for row in range(-1, rows):
            for col in range(-1, cols):
                row_indices = [row, row + 1, row, row + 1]
                col_indices = [col, col, col + 1, col + 1]
                check = get_check(row_indices, col_indices)

                # exclude exterior corner tiles that only touch one data qubit
                if np.count_nonzero(check) == 1:
                    continue

                if row % 2 == col % 2:
                    if 0 <= row < rows - 1:
                        # no X-type parity checks on the top/bottom boundaries
                        checks_x.append(check)
                elif 0 <= col < cols - 1:
                    # no Z-type parity checks on the left/right boundaries
                    checks_z.append(check)

        return np.array(checks_x), np.array(checks_z)


class ToricCode(CSSCode):
    """Surface code with periodic bounary conditions, encoding two logical qudits.

    Reference: https://errorcorrectionzoo.org/c/surface
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        rotated: bool = True,
        field: int | None = None,
        *,
        conjugate: bool = False,
    ) -> None:
        if cols is None:
            cols = rows

        # save known distances
        self._exact_distance_x = self._exact_distance_z = min(rows, cols)

        # which qubits should be Hadamard-transformed?
        qudits_to_conjugate: slice | Sequence[int] | None

        if rotated:
            if rows % 2 or cols % 2:
                raise ValueError(
                    f"Rotated toric code must have even side lengths, not {rows} and {cols}"
                )

            # rotated toric code
            matrix_x, matrix_z = ToricCode.get_rotated_checks(rows, cols)

            if conjugate:
                # Hadamard-transform qubits in a checkerboard pattern
                qudits_to_conjugate = [
                    idx for idx, (row, col) in enumerate(np.ndindex(rows, cols)) if (row + col) % 2
                ]

            else:
                qudits_to_conjugate = None

        else:
            # "original" toric code
            code_a = RingCode(rows, field)
            code_b = RingCode(cols, field)
            code_ab = HGPCode(code_a, code_b, field, conjugate=conjugate)
            matrix_x = code_ab.matrix_x
            matrix_z = code_ab.matrix_z
            qudits_to_conjugate = code_ab.conjugated

        CSSCode.__init__(
            self,
            matrix_x,
            matrix_z,
            field=field,
            conjugate=qudits_to_conjugate,
            skip_validation=True,
        )

    @classmethod
    def get_rotated_checks(
        cls, rows: int, cols: int
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Build X-sector and Z-sector parity check matrices.

        Same as in SurfaceCode.get_rotated_checks, but with periodic boundary conditions.
        """

        def get_check(
            row_indices: Sequence[int], col_indices: Sequence[int]
        ) -> npt.NDArray[np.int_]:
            """Check on the qubits with the given indices, with periodic boundary conditions."""
            check = np.zeros((rows, cols), dtype=int)
            for row, col in zip(row_indices, col_indices):
                check[row % rows, col % cols] = 1
            return check.ravel()

        checks_x = []
        checks_z = []
        for row in range(rows):
            for col in range(cols):
                row_indices = [row, row + 1, row, row + 1]
                col_indices = [col, col, col + 1, col + 1]
                check = get_check(row_indices, col_indices)
                if row % 2 == col % 2:
                    checks_x.append(check)
                else:
                    checks_z.append(check)

        return np.array(checks_x), np.array(checks_z)


class GeneralizedSurfaceCode(CSSCode):
    """Surface or toric code defined on a multi-dimensional hypercubic lattice.

    Reference: https://errorcorrectionzoo.org/c/higher_dimensional_surface
    """

    def __init__(
        self,
        size: int,
        dim: int,
        periodic: bool = False,
        field: int | None = None,
        *,
        conjugate: slice | Sequence[int] | None = (),
    ) -> None:
        if dim < 2:
            raise ValueError(
                f"The dimension of a generalized surface code should be >= 2 (provided: {dim})"
            )

        # save known distances
        # TODO: find and link source for these
        self._exact_distance_x = size ** (dim - 1)
        self._exact_distance_z = size

        base_code = RingCode(size, field) if periodic else RepetitionCode(size, field)

        # build a chain complex one link at a time
        chain = ChainComplex(base_code.matrix)
        link = ChainComplex(base_code.matrix.T)
        for _ in range(dim - 1):
            chain = ChainComplex.tensor_product(chain, link)

            # to reduce computational overhead, remove chain links that we don't care about
            chain = ChainComplex(*chain.ops[:2])

        matrix_x, matrix_z = chain.op(1), chain.op(2).T
        assert not isinstance(matrix_x, abstract.Protograph)
        assert not isinstance(matrix_z, abstract.Protograph)
        CSSCode.__init__(self, matrix_x, matrix_z, field, conjugate=conjugate, skip_validation=True)
