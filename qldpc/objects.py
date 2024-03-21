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
        return self.__index__()

    def __index__(self) -> int:
        """Allow indexing arrays with Pauli operators."""
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

    A Cayley complex is a geometric structure that is built out of two subsets A and B of a group G.
    The subsets respectively act on elements of G from the left and right, and must be symmetric,
    which is to say (for example) that if a ∈ A, then a^-1 ∈ A.

    The generating data (A,B) is used to build vertices V, edges E, and faces F as follows:
    - vertices are members of G,
    - edges have the form (g, ag) and (g, gb), and
    - faces f(g,a,b) have the form {g, ab, gb, agb}:

         g  →  gb
         ↓     ↓
        ag  → agb

    This complex may generally be disconnected, so we keep only the component that is connected to
    the identity element.  Equivalently, we can define G to be the group generated by A ∪ B.

    After constructing the complex (V,E,F), we can define two bipartite directed graphs:
    - subgraph_0 with edges ( g, f(g,a,b)), and
    - subgraph_1 with edges (ag, f(g,a,b)).
    These graphs are used to construct classical Tanner codes that serve as the X and Z sectors of a
    quantum CSS code.  A CSS code constructed in this way is called a quantum Tanner code.

    There are, however, two complications to keep in mind.  First, in order for the faces to be
    nondegenerate (that is, for each face to contain four vertices), the generating data (A,B) must
    satisfy the Total No Conjugacy condition:

    [1] ag != gb for all g,a,b in (G,A,B).

    Second, in order to construct a valid quantum Tanner code out of subgraph_0 and subgraph_1, the
    graph (V,E) must be bipartite, V = V_0 ∪ V_1, such that (for example) nodes {g, agb} are in one
    partition, while nodes {ag, gb} are in the other partition.  The nodes V_i are then used as the
    source nodes of subgraph_i.  The graph (V,E) is bipartite if:

    [2] The Cayley graphs (G;A) and (G;B) both are bipartite.

    The Cayley graphs (G;A) and (G;B) are graphs whose
    - vertices are members of G, and
    - edges are pairs of vertices connected by A or B, as in (g, ag) or (g, gb).

    In fact, condition [2] can be enforced at no added cost by taking the double cover of G and
    modifying members of A and B as:
    - G --> G ⊗ Z_2,
    - a --> (a,1), and
    - b --> (b,1),
    where Z_2 ~ {0,1} is the 2-element group (under addition), such that (a,1) * (g,i) = (ag,i+1)
    and (g,i) * (b,1) = (gb,i+1).  The complex generated by A and B after this modification is the
    "bipartite" Cayley complex constructed in https://arxiv.org/abs/2202.13641.

    If requirement [1] is not satisfied, then we can construct a "quadripartite" complex that
    enforces [1] and [2] by taking the quadruple cover of G and modifying members of A and B as:
    - G -->  G ⊗ Z_2 ⊗ Z_2,
    - a --> (a,1,0), and
    - b --> (b,0,1),
    where similarly to before (a,1,0) * (g,i,j) = (ag,i+1,j) and (g,i,j) * (b,0,1) = (gb,i,j+1).
    This modification of A and B corresponds to the "quadripartite" Cayley complex constructed in
    https://arxiv.org/abs/2206.07571.

    References:
    - https://arxiv.org/abs/2202.13641
    - https://arxiv.org/abs/2206.07571
    - https://www.youtube.com/watch?v=orWcstqWGGo
    """

    # identifying data
    group: abstract.Group
    subset_a: set[abstract.GroupMember]
    subset_b: set[abstract.GroupMember]
    bipartite: bool

    # geometric data
    faces: set[frozenset[abstract.GroupMember]]
    graph: nx.Graph

    def __init__(
        self,
        subset_a: Collection[abstract.GroupMember],
        subset_b: Collection[abstract.GroupMember] | None = None,
        *,
        bipartite: bool = False,
    ) -> None:
        """Construct a left-right Cayley complex."""
        if subset_b is None:
            subset_b = subset_a

        # assert that the generating subsets are symmetric
        for subset, name in [(subset_a, "subset_a"), (subset_b, "subset_b")]:
            for member in subset:
                if ~member not in subset:
                    message = (
                        "Provided generating subsets are not symmetric\n"
                        + f"Generating {name} contains {member} but not its inverse, {~member}"
                    )
                    raise ValueError(message)

        # take the double cover(s) of the group as appropriate
        identity, shift = abstract.CyclicGroup(2).generate()
        if bipartite:
            shift_a = shift_b = shift
        else:
            shift_a = shift @ identity
            shift_b = identity @ shift
        subset_a = set(aa @ shift_a for aa in subset_a)
        subset_b = set(bb @ shift_b for bb in subset_b)

        # identify the group generated by the subsets
        group = abstract.Group.from_generators(*subset_a, *subset_b)
        if bipartite and not CayleyComplex.satisfies_total_no_conjugacy(group, subset_a, subset_b):
            raise ValueError("Provided group and subsets do not satisfy Total No Conjugacy")

        # save identifying data
        self.group = group
        self.subset_a = subset_a
        self.subset_b = subset_b
        self.bipartite = bipartite

        # save geometric data
        self.faces = set()
        self.graph = nx.Graph()
        for gg, aa, bb in itertools.product(group.generate(), subset_a, subset_b):
            aa_gg, gg_bb, aa_gg_bb = aa * gg, gg * bb, aa * gg * bb
            face = frozenset([gg, aa_gg, gg_bb, aa_gg_bb])
            self.faces.add(face)
            self.graph.add_edge(gg, aa_gg)
            self.graph.add_edge(gg, gg_bb)
            self.graph.add_edge(aa_gg, aa_gg_bb)
            self.graph.add_edge(gg_bb, aa_gg_bb)

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

    def subgraphs(self) -> tuple[nx.DiGraph, nx.DiGraph]:
        """Build the subgraphs that are used to construct a quantum Tanner code.

        These subgraphs are defined using the faces of a Cayley complex.  Each face looks like:

         g ―――――――――― gb

         |  f(g,a,b)  |

        ag ――――――――― agb

        where f(g,a,b) = {g, ab, gb, agb}.  Specifically, the (directed) subgraphs are:
        - subgraph_0 with edges ( g, f(g,a,b)), and
        - subgraph_1 with edges (ag, f(g,a,b)).
        These subgraphs define a CSS code whose X-type parity checks are a classical Tanner code on
        subgraph_0, and Z-type parity checks are a classical Tanner code on subgraph_1.

        As a matter of practice, defining Tanner codes on subgraph_0 and subgrah_1 requires choosing
        an ordering on the edges incident to every source node of these graphs.  If the group G is
        equipped with a total order, a natural ordering of edges incident to every source node is
        induced by assigning the label (a, b) to edge (g, f(g,a,b)).  Consistency then requires that
        edge (ag, f(g,a,b)) has label (a^-1, b), as can be verified by defining g' = ag and
        checking that f(g,a,b) = f(g',a^-1,b).
        """
        subgraph_0 = nx.DiGraph()
        subgraph_1 = nx.DiGraph()
        nodes_0, _ = nx.bipartite.sets(self.graph)
        for gg, aa, bb in itertools.product(nodes_0, self.subset_a, self.subset_b):
            aa_gg, gg_bb, aa_gg_bb = aa * gg, gg * bb, aa * gg * bb
            face = frozenset([gg, aa_gg, gg_bb, aa_gg_bb])
            subgraph_0.add_edge(gg, face, sort=(aa, bb))
            subgraph_1.add_edge(aa_gg, face, sort=(~aa, bb))
        return subgraph_0, subgraph_1
