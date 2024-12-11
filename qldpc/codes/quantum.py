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
import itertools
import os
from collections.abc import Collection, Sequence

import networkx as nx
import numpy as np
import numpy.typing as npt
import sympy

from qldpc import abstract
from qldpc.objects import CayleyComplex, ChainComplex, Node, Pauli, QuditOperator

from .classical import HammingCode, RepetitionCode, RingCode, TannerCode
from .common import ClassicalCode, CSSCode, QuditCode


class FiveQubitCode(QuditCode):
    """Smallest quantum error-correcting code."""

    def __init__(self) -> None:
        self._exact_distance = 3
        code = QuditCode.from_stabilizers(
            "X Z Z X I",
            "I X Z Z X",
            "X I X Z Z",
            "Z X I X Z",
            field=2,
        )
        QuditCode.__init__(self, code, skip_validation=True)


class SteaneCode(CSSCode):
    """Smallest quantum error-correcting CSS code."""

    def __init__(self) -> None:
        self._exact_distance_x = self._exact_distance_z = 3
        code = HammingCode(3, field=2)
        CSSCode.__init__(self, code, code)


################################################################################
# two-block and bicycle codes


class TBCode(CSSCode):
    """Two-block (TB) code.

    A TBCode code is built out of two matrices A and B, which are combined as
    - matrix_x = [A, B], and
    - matrix_z = [B.T, -A.T],
    to form the parity check matrices of a CSSCode.  If A and B commute, the parity check matrices
    matrix_x and matrix_z satisfy the requirements of a CSSCode.

    Two-block codes constructed out of circulant matrices are known as quasi-cyclic codes.

    References:
    - https://errorcorrectionzoo.org/c/two_block_quantum
    """

    def __init__(
        self,
        matrix_a: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        matrix_b: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
        *,
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
            promise_balanced_codes=promise_balanced_codes,
            skip_validation=True,
        )


class QCCode(TBCode):
    """Quasi-cyclic code.

    A quasi-cyclic code is a CSS code with subcode parity check matrices
    - matrix_x = [A, B], and
    - matrix_z = [B.T, -A.T].
    Here A and B are polynomials of the form A = sum_{i,j,k,...} A_{ijk...} x^i y^j z^k ...,
    where
    - A_{ijk...} is a scalar coefficient (over some finite field),
    - x, y, z, ... are generators of cyclic groups of orders R_x, R_y, R_z, ...
    - the monomial x^i y^j z^k ... represents a tensor product of cyclic shift matrices.

    A quasi-cyclic code code is defined by...
    [1] a sequence of cyclic group orders, and
    [2] two multivariate polynomials.
    By default, group orders are associated in lexicographic order with free variables of the
    polynomials.  Group orders can also be assigned to variables explicitly with a dictionary.

    References:
    - https://errorcorrectionzoo.org/c/quantum_quasi_cyclic

    Univariate quasi-cyclic codes are generalized bicycle codes:
    - https://errorcorrectionzoo.org/c/generalized_bicycle
    - https://arxiv.org/pdf/2203.17216

    Bivariate quasi-cyclic codes are bivariate bicycle codes; see BBCode class.
    """

    def __init__(
        self,
        orders: Sequence[int] | dict[sympy.Symbol, int],
        poly_a: sympy.Basic,
        poly_b: sympy.Basic,
        field: int | None = None,
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
        self.group = abstract.AbelianGroup(*self.orders, field=field, product_lift=True)
        self.gens = self.group.generators
        self.symbol_gens = dict(zip(self.symbols, self.gens))

        # build defining matrices of a generalized bicycle code
        matrix_a = self.eval(self.poly_a).lift()
        matrix_b = self.eval(self.poly_b).lift()
        TBCode.__init__(
            self, matrix_a, matrix_b, field, promise_balanced_codes=True, skip_validation=True
        )

    def eval(
        self,
        expression: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul | sympy.Poly,
    ) -> abstract.Element:
        """Convert a sympy expression into an element of this code's group algebra."""
        # evaluate a monomial
        if isinstance(expression, (sympy.Integer, sympy.Symbol, sympy.Pow, sympy.Mul)):
            coeff, monomial = expression.as_coeff_Mul()
            member = self.to_group_member(monomial)
            return int(coeff) * abstract.Element(self.group, member)

        # evaluate a polynomial
        element = abstract.Element(self.group)
        for term in expression.as_expr().args:
            element += self.eval(term)
        return element

    def to_group_member(
        self, expression: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul
    ) -> abstract.GroupMember:
        """Convert a monomial into an associated member of this code's base group."""
        coeff, exponents = self.get_coefficient_and_exponents(expression)
        assert coeff == 1

        output = self.group.identity
        for base, exponent in exponents.items():
            output *= self.symbol_gens[base] ** exponent
        return output

    @classmethod
    def get_coefficient_and_exponents(
        self, expression: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul
    ) -> tuple[int, dict[sympy.Symbol, int]]:
        """Extract the coefficients and exponents in a monomial expression.

        For example, this method takes 5 x**3 y**2 to (5, {x: 3, y: 2})."""
        coeff, monomial = expression.as_coeff_Mul()
        exponents = {}
        if isinstance(monomial, sympy.Integer):
            coeff *= int(monomial)
        elif isinstance(monomial, sympy.Symbol):
            exponents[monomial] = 1
        elif isinstance(monomial, sympy.Pow):
            base, exponent = monomial.as_base_exp()
            exponents[base] = exponent
        elif isinstance(monomial, sympy.Mul):
            for factor in monomial.args:
                base, exponent = factor.as_base_exp()
                exponents[base] = exponent
        return coeff, exponents

    def get_canonical_form(
        self, poly: sympy.Poly, orders: tuple[int, ...] | None = None
    ) -> tuple[sympy.Poly, sympy.Poly]:
        """Canonicalize the given polynomial, shifting exponents to (-order/2, order/2]."""
        orders = orders or self.orders
        assert len(orders) == len(self.symbols)

        # canonialize and add one term ata time
        new_poly = sympy.core.numbers.Zero()
        for term in poly.args:
            coeff, exponents = self.get_coefficient_and_exponents(term)

            new_term = sympy.core.numbers.One()
            for symbol, order in zip(self.symbols, orders):
                new_exponent = exponents.get(symbol, 0) % order
                if new_exponent > order / 2:
                    new_exponent -= order
                new_term *= coeff * symbol**new_exponent

            new_poly += new_term

        return new_poly


# TODO: example notebook featuring this code
class BBCode(QCCode):
    """Bivariate bicycle code.

    A bivariate bicycle code is a CSS code with subcode parity check matrices
    - matrix_x = [A, B], and
    - matrix_z = [B.T, -A.T].
    Here A and B are polynomials of the form A = sum_{i,j} A_{ij} x^i y^j, where
    - A_{ij} is a scalar coefficient (over some finite field),
    - x and y are, respectively, generators of cyclic groups of orders R_x and R_y, and
    - the monomial x^i y^j represents a tensor product of cyclic shift matrices.

    A bivariate bicycle code is defined by...
    [1] two cyclic group orders, and
    [2] two bivariate polynomials.
    The polynomials should be sympy expressions such as 1 + x + x * y**2 with sympy.abc variables x
    and y.  Group orders are, by default, associated with the free variables of the polynomials in
    lexicographic order.  Group orders can also be assigned to variables explicitly with a
    dictionary, as in {x: 12, y: 6}.

    The polynomials A and B induce a "canonical" layout of the data and check qubits of a BBCode.
    In the canonical layout, qubits are organized into plaquettes of four qubits that look like
        Z L
        R X
    where L and R are data qubits, and X and Z are check qubits.  More specifically:
    - L and R data qubits are addressed by the left and right halves of matrix_x (or matrix_z).
    - X are check qubits measure X-type parity checks, and are associated with rows of matrix_x.
    - Z are check qubits measure Z-type parity checks, and are associated with rows of matrix_z.
    These four-qubit plaquettes are arranged into a rectangular grid that is R_x plaquettes tall and
    R_y plaquettes wide, where R_x and R_y are the orders of the cyclic groups generated by x and y.
    Each qubit can then be labeled by coordinates (aa, bb) of a plaquette, corresponding to a row
    and column in the grid, and a "sector" (L, R, X, or Z) within a plaquette.

    If we associate (L, R) ~ (0, 1), then the qubit addressed by column qq of matrix_x (or matrix_z)
    has the label (sector, aa, bb) = numpy.unravel_index(qq, [2, R_x, R_y]).  That is, the integer
    index of a data qubit its label are essentially related to each other by array reshaping.  The
    label of a check qubit is similarly obtained by associating (X, Z) ~ (0, 1).  In this case, the
    numerical index of a check qubit is the index of its corresponding row in the full parity check
    matrix of a BBCode.

    The connections between data and check qubits can be read directly from the polynomials A and B:
    - If A_{ij} != 0, then...
      - every X qubit addresses an L qubit that is (i, j) plaquettes (down, right), and
      - every Z qubit addresses an R qubit that is (i, j) plaquettes (up, left).
    - If B_{ij} != 0, then...
      - every X qubit addresses an R qubit that is (i, j) plaquettes (down, right), and
      - every Z qubit addresses an L qubit that is (i, j) plaquettes (up, left).
    Here the grid of plaquettes is assumed to have periodic boundary conditions, so going one
    plaquette "up" from the top row of the grid gets you to the bottom row of the grid.

    References:
    - https://errorcorrectionzoo.org/c/qcga
    - https://arxiv.org/abs/2308.07915
    - https://arxiv.org/pdf/2408.10001
    - https://arxiv.org/pdf/2404.18809
    """

    def __init__(
        self,
        orders: Sequence[int] | dict[sympy.Symbol, int],
        poly_a: sympy.Basic,
        poly_b: sympy.Basic,
        field: int | None = None,
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
        QCCode.__init__(self, orders, poly_a, poly_b, field)

    def get_node_label(self, node: Node) -> tuple[str, int, int]:
        """Convert a node of this code's Tanner graph into a qubit label."""
        ss, aa, bb = np.unravel_index(node.index, (2,) + self.orders)
        if node.is_data:
            sector = "L" if ss == 0 else "R"
        else:
            sector = "X" if ss == 0 else "Z"
        return sector, int(aa), int(bb)

    def get_qubit_pos(
        self,
        qubit: Node | tuple[str, int, int],
        folded_layout: bool = False,
    ) -> tuple[int, int]:
        """Get the canonical position of a qubit with the given label.

        If a folded_layout is True, "fold" the array of qubits as in Figure 2 of arXiv:2404.18809.
        """
        if isinstance(qubit, Node):
            qubit = self.get_node_label(qubit)
        sector, aa, bb = qubit
        assert sector in ["L", "R", "X", "Z"]

        xx = 2 * aa + int(sector in ["R", "X"])
        yy = 2 * bb + int(sector in ["L", "X"])
        if folded_layout:
            xx = 2 * xx if xx < self.orders[0] else (2 * self.orders[0] - 1 - xx) * 2 + 1
            yy = 2 * yy if yy < self.orders[1] else (2 * self.orders[1] - 1 - yy) * 2 + 1
        return xx, yy

    def get_equivalent_toric_layout_code_data(
        self,
    ) -> Sequence[tuple[tuple[int, int], sympy.Poly, sympy.Poly]]:
        """Get the generating data for equivalent BBCodes with "manifest" toric layouts.

        For simplicity, we consider BBCodes for qubits (with base field F_2) in the text below.

        A BBCode has a "manifest" toric layout if it is generated by polynomials that look like
            poly_a = 1 + x + ..., and
            poly_b = 1 + y + ...,
        These codes are "equivalent" in the sense that they can be obtained from one another by a
        permutation of data and check qubits.

        To an find equivalent BBCode with a manifest toric layouts, we take
            poly_a = sum_j A_j --> poly_a / A_k = 1 + sum_{j != k} A_j/A_k, and
            poly_b = sum_j B_j --> poly_b / B_l = 1 + sum_{j != l} B_j/B_l.
        Each pair of terms (A_j/A_k, B_j/B_l) is then a candidate for generators (g, h).

        This modification of polynomials and change-of-basis from the original generators (x, y)
        to (g, h) produces an equivalent BBCode so long as g and h satisfy the conditions in Lemma
        of arXiv:2308.07915, which boils down to the requirement that
            order(g) * order(h) = order(<g, h>) = order(<x, y>),
        where (for example) <x, y> is the Abelian group generated by x and y.
        """
        if not nx.is_weakly_connected(self.graph):
            # a connected tanner graph is required for a toric layout to exist
            return []

        # identify unique symbols to use as placeholders for the new generators
        symbol_g = sympy.Symbol("".join(map(str, self.symbols)))
        symbol_h = sympy.Symbol("".join(map(str, self.symbols * 2)))

        # identify individual monomials (terms without their coefficients) in the polynomials
        monomials_a = [term.as_coeff_Mul()[1] for term in self.poly_a.as_expr().args]
        monomials_b = [term.as_coeff_Mul()[1] for term in self.poly_b.as_expr().args]

        # identify collections of monomials that can be combined to obtain a toric layout
        toric_params = []
        for (a_1, a_2), (b_1, b_2) in itertools.product(
            itertools.combinations(monomials_a, 2), itertools.combinations(monomials_b, 2)
        ):
            member_g = self.to_group_member(a_2 / a_1)
            member_h = self.to_group_member(b_2 / b_1)
            if (
                member_g.order() * member_h.order() == self.group.order
                and abstract.Group(member_g, member_h).order == self.group.order
            ):
                toric_params.append((a_1, a_2, b_1, b_2))
                toric_params.append((a_1, a_2, b_2, b_1))
                toric_params.append((a_2, a_1, b_1, b_2))
                toric_params.append((a_2, a_1, b_2, b_1))

        toric_layout_generating_data = []
        for a_1, a_2, b_1, b_2 in toric_params:
            # new generators and their corresponding cyclic group orders
            gen_g = a_2 / a_1
            gen_h = b_2 / b_1
            orders = (self.to_group_member(gen_g).order(), self.to_group_member(gen_h).order())

            # new "shifted" polynomials
            shifted_poly_a = self.poly_a / a_1
            shifted_poly_b = self.poly_b / b_1

            # without loss of generality, enforce that the toric layout "width" >= "height"
            if orders[0] < orders[1]:
                orders = orders[::-1]
                gen_g, gen_h = gen_h, gen_g
                shifted_poly_a, shifted_poly_b = shifted_poly_b, shifted_poly_a

            """
            Let x and y be the generators of the polynomials in the original BBCode.
            We want to "change basis" from generators (x, y) to generators (g, h), where
                g = x^p y^q,
                h = x^u y^v.
            To do so, need to find integer exponents (i, j, a, b) for which
                x^i y^j = g^a h^b,
            or equivalently
                i = a p + b u  mod order(x),
                j = a q + b v  mod order(y).
            More specifically, we need to find exponents (gx, hx) and (gy, hy) for which
                x = g^gx h^hx,
                y = g^gy h^hy.
            We find these exponents with a brute force search...
            """
            _, exponents_g = self.get_coefficient_and_exponents(gen_g)
            _, exponents_h = self.get_coefficient_and_exponents(gen_h)
            pp, qq = [exponents_g.get(symbol, 0) for symbol in self.symbols]
            uu, vv = [exponents_h.get(symbol, 0) for symbol in self.symbols]
            for aa, bb in np.ndindex(orders):
                ii = (aa * pp + bb * uu) % self.orders[0]
                jj = (aa * qq + bb * vv) % self.orders[1]
                if (ii, jj) == (1, 0):
                    gx, hx = aa, bb
                elif (ii, jj) == (0, 1):
                    gy, hy = aa, bb
            gen_x = symbol_g**gx * symbol_h**hx
            gen_y = symbol_g**gy * symbol_h**hy

            # build polynomials for an equivalent BBCode with a manifest toric layout
            new_polys = []
            for poly in [shifted_poly_a, shifted_poly_b]:
                # expand (x, y) in terms of (g, h), then "rename" (g, h) to (x, y)
                poly = poly.expand()
                poly = poly.subs({self.symbols[0]: gen_x, self.symbols[1]: gen_y})
                poly = poly.subs({symbol_g: self.symbols[0], symbol_h: self.symbols[1]})

                # add canonical form of this polynomial, with exponents in (-order/2, order/2]
                new_polys.append(self.get_canonical_form(poly, orders))

            new_poly_a, new_poly_b = new_polys
            generating_data = (orders, new_poly_a, new_poly_b)
            if generating_data not in toric_layout_generating_data:
                toric_layout_generating_data.append(generating_data)

        return toric_layout_generating_data


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

    By default, the check qudits in sectors...
    - (1, 0) of G_AB measure X-type operators, and
    - (0, 1) of G_AB measure Z-type operators.

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

        # if Hadamard-transforming qudits, conjugate those in the (1, 1) sector by default
        self._default_conjugate = slice(self.sector_size[0, 0], None)

        CSSCode.__init__(
            self, matrix_x.astype(int), matrix_z.astype(int), field, skip_validation=True
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
    def get_graph_product(cls, graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> nx.DiGraph:
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

        # if Hadamard-transforming qudits, conjugate those in the (1, 1) sector by default
        self._default_conjugate = slice(self.sector_size[0, 0], None)

        CSSCode.__init__(
            self,
            abstract.Protograph(matrix_x.astype(object)).lift(),
            abstract.Protograph(matrix_z.astype(object)).lift(),
            field,
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
        CSSCode.__init__(self, code_x, code_z, field, skip_validation=True)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QTCode)
            and other.code_a == other.code_a
            and other.code_b == other.code_b
            and other.complex.subset_a == other.complex.subset_a
            and other.complex.subset_b == other.complex.subset_b
            and other.complex.bipartite == other.complex.bipartite
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
        return QTCode(subset_a, subset_b, code_a, code_b, bipartite=bipartite)

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

    @classmethod
    def load(cls, path: str) -> QTCode:
        """Load a QTCode from a file."""
        if not os.path.isfile(path):
            raise ValueError(f"Path does not exist: {path}")

        with open(path, "r") as file:
            lines = file.read().splitlines()

        # load miscellaneous data
        field = ast.literal_eval(lines[-2].split(":")[-1])
        bipartite = ast.literal_eval(lines[-1].split(":")[-1])

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
        return QTCode(subset_a, subset_b, code_a, code_b, bipartite=bipartite)


################################################################################
# common quantum codes


class SurfaceCode(CSSCode):
    """The one and only!

    Actually, there are two variants: "ordinary" and "rotated" surface codes.
    The rotated code is more qubit-efficient.
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        rotated: bool = True,
        field: int | None = None,
    ) -> None:
        if cols is None:
            cols = rows
        self.rows = rows
        self.cols = cols

        # save known distances
        self._exact_distance_x = cols
        self._exact_distance_z = rows

        if rotated:
            # rotated surface code
            matrix_x, matrix_z = SurfaceCode.get_rotated_checks(rows, cols)
            self._default_conjugate: list[int] | slice = [
                idx
                for idx, (row, col) in enumerate(np.ndindex(self.rows, self.cols))
                if (row + col) % 2
            ]

        else:
            # "original" surface code
            code_a = RepetitionCode(rows, field)
            code_b = RepetitionCode(cols, field)
            code_ab = HGPCode(code_a, code_b, field)
            matrix_x = code_ab.matrix_x
            matrix_z = code_ab.matrix_z
            self._default_conjugate = slice(code_ab.sector_size[0, 0], None)

        CSSCode.__init__(self, matrix_x, matrix_z, field=field, skip_validation=True)

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
    ) -> None:
        if cols is None:
            cols = rows
        self.rows = rows
        self.cols = cols

        # save known distances
        self._exact_distance_x = self._exact_distance_z = min(rows, cols)

        if rotated:
            if rows % 2 or cols % 2:
                raise ValueError(
                    f"Rotated toric code must have even side lengths, not {rows} and {cols}"
                )

            # rotated toric code
            matrix_x, matrix_z = ToricCode.get_rotated_checks(rows, cols)
            self._default_conjugate: list[int] | slice = [
                idx
                for idx, (row, col) in enumerate(np.ndindex(self.rows, self.cols))
                if (row + col) % 2
            ]

        else:
            # "original" toric code
            code_a = RingCode(rows, field)
            code_b = RingCode(cols, field)
            code_ab = HGPCode(code_a, code_b, field)
            matrix_x = code_ab.matrix_x
            matrix_z = code_ab.matrix_z
            self._default_conjugate = slice(code_ab.sector_size[0, 0], None)

        CSSCode.__init__(self, matrix_x, matrix_z, field=field, skip_validation=True)

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
        CSSCode.__init__(self, matrix_x, matrix_z, field, skip_validation=True)
