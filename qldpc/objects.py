"""Instrumental objects used to construct error-correcting codes

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

import dataclasses
import enum
import itertools
from collections.abc import Collection

import networkx as nx

from qldpc import abstract

# from typing import Literal


class Pauli(enum.Enum):
    """Pauli operators."""

    I = (0, 0)  # noqa: E741
    Z = (0, 1)
    X = (1, 0)
    Y = (1, 1)

    def __mul__(self, other: Pauli) -> Pauli:
        """Product of two Pauli operators."""
        val_x = (self.value[0] + other.value[0]) % 2
        val_z = (self.value[1] + other.value[1]) % 2
        return Pauli((val_x, val_z))

    def __invert__(self) -> Pauli:
        """Hadamard-transform this Pauli operator."""
        return Pauli(self.value[::-1])

    def __str__(self) -> str:
        if self == Pauli.I:
            return "I"
        elif self == Pauli.Z:
            return "Z"
        elif self == Pauli.X:
            return "X"
        return "Y"

    @classmethod
    def from_string(cls, string: str) -> Pauli:
        """Build a Pauli operator from a string."""
        if string == "I":
            return Pauli.I
        elif string == "Z":
            return Pauli.Z
        elif string == "X":
            return Pauli.X
        elif string == "Y":
            return Pauli.Y
        raise ValueError(f"Invalid Pauli operator: {string}")

    @property
    def index(self) -> int:
        """Numerical index for Pauli operators."""
        if self == Pauli.X:
            return 0
        if self == Pauli.Z:
            return 1
        raise AttributeError(f"No index for {self}.")


class QuditOperator:
    """A qudit operator of the form X(val_x)*Z(val_z)."""

    def __init__(self, value: tuple[int, int] = (0, 0)) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, QuditOperator) and self.value == other.value

    def __invert__(self) -> QuditOperator:
        """Fourier-transform this qudit operator."""
        return QuditOperator(self.value[::-1])

    def __neg__(self) -> QuditOperator:
        """Invert the shifts and phases on this qudit operator."""
        return QuditOperator((-self.value[0], -self.value[1]))

    def __str__(self) -> str:
        val_x, val_z = self.value
        if not val_x and not val_z:
            return "I"
        if val_x == val_z:
            return f"Y({val_z})"
        ops = []
        if val_x:
            ops.append(f"X({val_x})")
        if val_z:
            ops.append(f"Z({val_z})")
        return "*".join(ops)

    @classmethod
    def from_string(cls, string: str) -> QuditOperator:
        """Build a qudit operator from its string representation."""
        if string == "I":
            return QuditOperator((0, 0))

        invalid_op = f"Invalid qudit operator: {string}"

        val_x, val_z = 0, 0
        factors = string.split("*")
        if len(factors) > 2:
            raise ValueError(invalid_op)

        for factor in factors:
            pauli = factor[0]
            val_str = factor[2:-1]
            _factor = f"{pauli}({val_str})"
            if pauli not in "XYZ" or not val_str.isnumeric() or factor != _factor:
                raise ValueError(invalid_op)

            val = int(val_str)
            if pauli == "X":
                val_x = val
            elif pauli == "Z":
                val_z = val
            else:  # pauli == "Y"
                val_x = val_z = val

        return QuditOperator((val_x, val_z))


@dataclasses.dataclass
class Node:
    """Node in a Tanner graph.

    A node essentially an integer index, together with a boolean flag to distinguish "data" node
    from a "check" node in an error-correcting code.
    """

    index: int
    is_data: bool = True

    def __hash__(self) -> int:
        return hash((self.index, self.is_data))

    def __lt__(self, other: Node) -> bool:
        if self.is_data == other.is_data:
            return self.index < other.index
        return self.is_data  # data bits "precede" check bits

    def __str__(self) -> str:
        tag = "d" if self.is_data else "c"
        return f"{tag}_{self.index}"


class CayleyComplex:
    """Left-right Cayley complex, used for constructing quantum Tanner codes.

    A Cayley complex is a geometric structure built out of a two subsets A and B of a group G.  The
    subsets respectively act on elements of G from the left and right, and must be symmetric, which
    is to say (for example) that a ∈ A iff a^-1 ∈ A.  To avoid constructing a complex that factors
    into disconnected pieces, we can define G as the group generated by all elements of A and B.

    The generating data (A,B) is used to build vertices V, edges E, and faces F as follows:
    - vertices are members of G,
    - edges have the form (g, ag) and (g, gb), and
    - faces f(g,a,b) have the form {g, ab, gb, agb}:

         g  →  gb
         ↓     ↓
        ag  → agb

    The complex (V,E,F) is in turn used to construct two bipartite directed graphs:
    - subgraph_0 with edges ( g, f(g,a,b)), and
    - subgraph_1 with edges (ag, f(g,a,b)).
    These graphs are used to construct classical Tanner codes that serve as the X and Z sectors of a
    quantum CSS code (namely, a quantum Tanner code).

    There are, however, two complications to keep in mind.  First, in order for the faces to be non
    degenerate (that is, for each face to contain four vertices), the generating data (A,B) must
    satisfy the Total No Conjugacy condition:

    [1] ag != gb for all g,a,b in (G,A,B).

    Second, in order to construct a valid quantum Tanner code out of subgraph_0 and subgraph_1, the
    graph (V,E) must be bipartite, V = V_0 ∪ V_1, such that (for example) nodes {g,agb} are in one
    partition, while nodes {ag,gb} are in the other partition.  The nodes V_i are then used as the
    sources of subgraph_i.  The graph (V,E) is bipartite if:

    [2] The Cayley graphs (G;A) and (G;B) both are bipartite.

    The Cayley graphs (G;A) and (G;B) are graphs whose
    - vertices are members of G, and
    - edges are pairs of vertices connected by A or B, as in (g, ag) or (g, gb).

    If both [1] and [2] are satisfied, when we can construct a Cayley complex out of (G,A,B)
    directly, which we call a "rank-0" complex.

    If [1] is satisfied but [2] is not, then we can construct a "rank-1" complex that enforces
    requirement [2] by taking the double cover of G and modifying members of A and B as:
    - G --> G ⊗ {0,1},
    - a --> (a,1), and
    - b --> (b,1),
    where (a,1) acts on (g,i) as (a,1) * (g,i) = (ag,i+1), and similarly (b,1) * (g,i) = (gb,i+1).

    If requirement [1] is not satisfied, then we can construct a "rank-2" complex that enforces both
    [1] and [2] by taking the quadruple cover of G and modifying members of A and B as:
    - G -->  G ⊗ {0,1} ⊗ {0,1},
    - a --> (a,1,0), and
    - b --> (b,0,1),
    where similarly to before (a,1,0) * (g,i,j) = (ag,i+1,j) and (b,0,1) * (g,i,j) = (gb,i,j+1).

    References:
    - https://arxiv.org/abs/2202.13641
    - https://arxiv.org/abs/2206.07571
    - https://www.youtube.com/watch?v=orWcstqWGGo
    """

    # generating data
    subset_a: set[abstract.GroupMember]
    subset_b: set[abstract.GroupMember]
    group: abstract.Group

    # rank and graph (vertices and edges)
    rank: int
    graph: nx.Graph
    faces: set[frozenset[abstract.GroupMember]]

    # subgraphs used for a quantum Tanner code
    subgraph_0: nx.DiGraph
    subgraph_1: nx.DiGraph

    def __init__(
        self,
        subset_a: Collection[abstract.GroupMember],
        subset_b: Collection[abstract.GroupMember] | None = None,
        *,
        bipartite: bool | None = False,
    ) -> None:
        """Construct a left-right Cayley complex."""
        if subset_b is None:
            subset_b = subset_a
        subset_a = set(subset_a)
        subset_b = set(subset_b)
        assert all(~member in subset_a for member in subset_a), "subset_a is not symmetric"
        assert all(~member in subset_b for member in subset_b), "subset_b is not symmetric"

        # identify the group generated by the provided (sub)sets
        self.group = abstract.Group.from_generators(*subset_a, *subset_b)
        self.subset_a = subset_a
        self.subset_b = subset_b

        # TODO: Might be some issue with the sort
        if bipartite:
            assert CayleyComplex.satisfies_total_no_conjugacy(
                self.group, self.subset_a, self.subset_b
            )
            self.subgraph_0 = nx.DiGraph()
            self.subgraph_1 = nx.DiGraph()
            for gg, aa, bb in itertools.product(
                self.group.generate(), self.subset_a, self.subset_b
            ):
                aa_gg, gg_bb, aa_gg_bb = aa * gg, gg * bb, aa * gg * bb
                square = [(gg, 0), (aa_gg, 1), (gg_bb, 1), (aa_gg_bb, 0)]
                face = frozenset(square)
                self.subgraph_0.add_edge((gg, 0), face, sort=(aa, bb))
                self.subgraph_0.add_edge((aa_gg_bb, 0), face, sort=(aa, bb))
                self.subgraph_1.add_edge((aa_gg, 1), face, sort=(aa, bb))
                self.subgraph_1.add_edge((gg_bb, 1), face, sort=(aa, bb))

        # construct the vertices, edges, and faces of this complex
        # TODO: Check the sort thing needed for Tanner graph
        else:
            self.subgraph_0 = nx.DiGraph()
            self.subgraph_1 = nx.DiGraph()
            for gg, aa, bb in itertools.product(
                self.group.generate(), self.subset_a, self.subset_b
            ):
                aa_gg, gg_bb, aa_gg_bb = aa * gg, gg * bb, aa * gg * bb
                square = [(gg, 0, 0), (aa_gg, 1, 0), (gg_bb, 0, 1), (aa_gg_bb, 1, 1)]
                face = frozenset(square)
                self.subgraph_0.add_edge((gg, 0, 0), face, sort=(aa, bb))
                self.subgraph_0.add_edge((aa_gg_bb, 1, 1), face, sort=(aa, bb))
                self.subgraph_1.add_edge((aa_gg, 1, 0), face, sort=(aa, bb))
                self.subgraph_1.add_edge((gg_bb, 0, 1), face, sort=(aa, bb))

    @classmethod
    def satisfies_total_no_conjugacy(
        cls,
        group: abstract.Group,
        subset_a: Collection[abstract.GroupMember],
        subset_b: Collection[abstract.GroupMember],
    ) -> bool:
        """Check the Total No-Conjugacy condition: aa gg != gg bb for all gg, aa, bb."""
        return all(
            aa * gg != gg * bb
            for gg, aa, bb in itertools.product(group.generate(), subset_a, subset_b)
        )
