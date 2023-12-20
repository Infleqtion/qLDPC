"""Unit tests for objects.py

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
import numpy as np
import pytest

from qldpc import abstract, objects


def test_pauli() -> None:
    """Test Pauli operator capabilities."""
    assert str(objects.Pauli.I) == "I"
    assert str(objects.Pauli.X) == "X"
    assert str(objects.Pauli.Y) == "Y"
    assert str(objects.Pauli.Z) == "Z"

    assert ~objects.Pauli.Z == objects.Pauli.X
    assert ~objects.Pauli.X == objects.Pauli.Z
    assert ~objects.Pauli.Y == objects.Pauli.Y
    assert ~objects.Pauli.I == objects.Pauli.I

    paulis = [objects.Pauli.I, objects.Pauli.Z, objects.Pauli.X, objects.Pauli.Y]
    table = [[paulis.index(pp * qq) for qq in paulis] for pp in paulis]
    group = abstract.Group.product(abstract.CyclicGroup(2), repeat=2)
    assert np.array_equal(table, group.table)

    assert objects.Pauli.X.index != objects.Pauli.Z.index
    for pauli in [objects.Pauli.I, objects.Pauli.Y]:
        with pytest.raises(AttributeError, match="No index"):
            assert pauli.index


def test_node() -> None:
    """Node properties."""
    node_d1 = objects.Node(1, is_data=True)
    node_d2 = objects.Node(2, is_data=True)
    node_c1 = objects.Node(1, is_data=False)
    assert node_d1 < node_d2 < node_c1
    assert str(node_d1) == "q_1"
    assert hash(node_c1) != hash(node_d1) != hash(node_d2)


def test_cayley_complex() -> None:
    """Construct and test Cayley complexes."""
    group: abstract.Group
    subset_a: list[abstract.GroupMember]
    subset_b: list[abstract.GroupMember]

    # rank-2 complex
    group = abstract.CyclicGroup(3)
    shift = group.generators[0]
    subset_a = [shift, ~shift]
    cayplex = objects.CayleyComplex(subset_a)
    assert cayplex.rank == 2
    assert_valid_complex(cayplex)

    # rank-1 complex
    group = abstract.CyclicGroup(6)
    shift = group.generators[0]
    subset_a = [shift, shift**2, ~shift, (~shift) ** 2]
    subset_b = [shift**3]
    cayplex = objects.CayleyComplex(subset_a, subset_b)
    assert cayplex.rank == 1
    assert_valid_complex(cayplex)

    # rank-0 complex
    group = abstract.Group.product(abstract.CyclicGroup(2), abstract.CyclicGroup(5))
    shift_x, shift_y = group.generators
    subset_a = [shift_x * shift_y, ~(shift_x * shift_y)]
    subset_b = [shift_x * shift_y**2, ~(shift_x * shift_y**2)]
    cayplex = objects.CayleyComplex(subset_a, subset_b)
    assert cayplex.rank == 0
    assert_valid_complex(cayplex)

    # test setting rank manually
    trivial_group = abstract.TrivialGroup()
    subset = [trivial_group.identity]
    assert objects.CayleyComplex(subset, rank=2).rank == 2

    # test setting incompatible rank
    group = abstract.CyclicGroup(3)
    shift = group.generators[0]
    subset_a = [shift, shift**2]
    with pytest.raises(ValueError, match="Cannot set CayleyComplex rank"):
        cayplex = objects.CayleyComplex(subset_a, rank=0)


def assert_valid_complex(cayplex: objects.CayleyComplex) -> None:
    """Run various sanity checks on a Cayley complex."""
    # assert that the complex has the right number of vertices, edges, and faces
    size_g = cayplex.group.order()
    size_a = len(cayplex.subset_a)
    size_b = len(cayplex.subset_b)
    assert cayplex.graph.number_of_nodes() == size_g
    assert cayplex.graph.number_of_edges() == size_g * (size_a + size_b) // 2
    assert len(cayplex.faces) == size_g * size_a * size_b // 4

    # check that the subgraphs have the correct number of checks
    for graph in [cayplex.subgraph_0, cayplex.subgraph_1]:
        sources = [node for node in graph.nodes if graph.in_degree(node) == 0]
        assert {graph.out_degree(node) for node in sources} == {size_a * size_b}
