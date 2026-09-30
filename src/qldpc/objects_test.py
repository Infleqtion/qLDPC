# SPDX-License-Identifier: Apache-2.0

"""Unit tests for objects.py."""

from __future__ import annotations

import math

import galois
import numpy as np
import numpy.typing as npt
import pytest
import scipy.linalg
import sympy.combinatorics as comb

from qldpc import abstract, objects


def test_pauli() -> None:
    """Pauli operator capabilities."""
    for string in ["I", "X", "Y", "Z"]:
        assert str(objects.Pauli.from_string(string)) == string
    with pytest.raises(ValueError, match="Invalid Pauli operator"):
        objects.Pauli.from_string("Q")

    # from_string is case-sensitive, since QuditCode.from_strings relies on it
    with pytest.raises(ValueError, match="Invalid Pauli operator"):
        objects.Pauli.from_string("x")

    # Pauli.coerce_xz accepts Pauli.X, Pauli.Z, and case-insensitive "X" or "Z" strings
    xz_inputs: dict[objects.PauliXZ, tuple[objects.PauliXZLike, ...]] = {
        objects.Pauli.X: (objects.Pauli.X, "X", "x"),
        objects.Pauli.Z: (objects.Pauli.Z, "Z", "z"),
    }
    for expected_pauli, values in xz_inputs.items():
        for value in values:
            assert objects.Pauli.coerce_xz(value) is expected_pauli
    for invalid_value in [objects.Pauli.I, objects.Pauli.Y, "I", "y", "XZ", "", 0]:
        with pytest.raises(ValueError, match=r"Pauli\.X or Pauli\.Z"):
            objects.Pauli.coerce_xz(invalid_value)  # type: ignore[arg-type]

    assert ~objects.Pauli.Z == objects.Pauli.Z.swap_xz() == objects.Pauli.X
    assert ~objects.Pauli.X == objects.Pauli.X.swap_xz() == objects.Pauli.Z
    assert ~objects.Pauli.Y == objects.Pauli.Y
    assert ~objects.Pauli.I == objects.Pauli.I
    with pytest.raises(ValueError, match=r"Pauli\.X and Pauli\.Z"):
        objects.Pauli.Y.swap_xz()

    paulis = [objects.Pauli.I, objects.Pauli.Z, objects.Pauli.X, objects.Pauli.Y]
    table = [[paulis.index(pp * qq) for qq in paulis] for pp in paulis]
    group = abstract.Group.product(abstract.CyclicGroup(2), repeat=2)
    assert np.array_equal(table, group.table)

    assert objects.Pauli.X.index != objects.Pauli.Z.index
    for pauli in [objects.Pauli.I, objects.Pauli.Y]:
        with pytest.raises(AttributeError, match="No index"):
            assert pauli.index


def test_qudit_operator() -> None:
    """Qudit operator capabilities."""
    assert objects.QuditPauli((0, 0)) == objects.QuditPauli()
    assert objects.QuditPauli((0, 1)) == ~objects.QuditPauli((1, 0))
    assert -objects.QuditPauli((0, 1)) == objects.QuditPauli((0, -1))
    for op in ["I", "Y(1)", "X(1)*Z(2)"]:
        assert str(objects.QuditPauli.from_string(op)) == op
    for op in ["a*b*c", "a(1)", "*", "X(1)*", "", "X(²)"]:
        with pytest.raises(ValueError, match="Invalid qudit operator"):
            objects.QuditPauli.from_string(op)


def test_node() -> None:
    """Node properties."""
    node_d1 = objects.Node(1, is_data=True)
    node_d2 = objects.Node(2, is_data=True)
    node_c1 = objects.Node(1, is_data=False)
    assert node_d1 < node_d2 < node_c1
    assert str(node_d1) == "d_1"
    assert hash(node_c1) != hash(node_d1) != hash(node_d2)


def test_cayley_complex() -> None:
    """Construct and test Cayley complexes."""
    group: abstract.Group
    subset_a: list[abstract.GroupMember]
    subset_b: list[abstract.GroupMember]

    # raise error when trying to build a complex from non-symmetric generating sets
    with pytest.raises(ValueError, match="not symmetric"):
        group = abstract.CyclicGroup(3)
        subset_a = [group.generators[0]]
        objects.CayleyComplex(subset_a, bipartite=True)

    # quadripartite complex
    group = abstract.CyclicGroup(3)
    shift = group.generators[0]
    subset_a = [shift, ~shift]
    cayplex = objects.CayleyComplex(subset_a, bipartite=False)
    assert_valid_complex(cayplex)
    with pytest.raises(ValueError, match="do not satisfy Total No Conjugacy"):
        objects.CayleyComplex(subset_a, bipartite=True)

    # complexes that may be bipartite or quadripartite
    for bipartite in [True, False]:
        group = abstract.CyclicGroup(6)
        shift = group.generators[0]
        subset_a = [shift, shift**2, ~shift, (~shift) ** 2]
        subset_b = [shift**3]
        cayplex = objects.CayleyComplex(subset_a, subset_b, bipartite=bipartite)
        assert_valid_complex(cayplex)

        group = abstract.Group.product(abstract.CyclicGroup(2), abstract.CyclicGroup(5))
        shift_x, shift_y = group.generators
        subset_a = [shift_x * shift_y, ~(shift_x * shift_y)]
        subset_b = [shift_x * shift_y**2, ~(shift_x * shift_y**2)]
        cayplex = objects.CayleyComplex(subset_a, subset_b, bipartite=bipartite)
        assert_valid_complex(cayplex)


def assert_valid_complex(cayplex: objects.CayleyComplex) -> None:
    """Sanity check on the number of edges in a Cayley complex."""
    size_a = len(cayplex.subset_a)
    size_b = len(cayplex.subset_b)
    size_g = cayplex.graph.number_of_nodes()
    assert cayplex.graph.number_of_edges() == size_g * (size_a + size_b) // 2


def test_chain_complex(field: int = 3) -> None:
    """Chain complex construction and errors."""

    # tensor product of one-complexes
    matrix = np.random.randint(field, size=(2, 3))
    two_chain = objects.ChainComplex.tensor_product(matrix, matrix, field)
    assert not np.any(two_chain.op(0))
    assert not np.any(two_chain.op(two_chain.num_links + 1))

    # tensor product of a two-complex and its dual
    four_chain = objects.ChainComplex.tensor_product(two_chain, two_chain.T, field)
    four_chain._validate_ops()

    # tensor product of one-complexes over a group algebra
    ring = abstract.GroupRing(abstract.TrivialGroup(), field)
    ring_matrix = abstract.RingArray.build(matrix, ring)
    two_chain = objects.ChainComplex.tensor_product(ring_matrix, ring_matrix)
    assert not np.any(two_chain.op(0))
    assert not np.any(two_chain.op(two_chain.num_links + 1))

    # a tensor product over a nontrivial commutative group algebra must yield boundary operators
    # whose entries are all ring members (and can therefore be lifted to matrices), including the
    # three-or-more-link case where some operator blocks are entirely zero
    cyclic_ring = abstract.GroupRing(abstract.CyclicGroup(3), field)
    cyclic_matrix = abstract.RingArray.build(matrix, cyclic_ring)
    ring_chain = objects.ChainComplex.tensor_product(cyclic_matrix, cyclic_matrix)
    ring_chain = objects.ChainComplex.tensor_product(ring_chain, cyclic_matrix)
    ring_chain._validate_ops()
    for ring_op in ring_chain.ops:
        # every op of a ring-valued chain is a RingArray; assert narrows the union type for .lift()
        assert isinstance(ring_op, abstract.RingArray)
        ring_op.lift()

    # invalid chain complex constructions
    with pytest.raises(ValueError, match="inconsistent operator types"):
        objects.ChainComplex([matrix, abstract.RingArray.build([[0]])])
    with pytest.raises(ValueError, match="Inconsistent base fields"):
        objects.ChainComplex([galois.GF(field)(matrix)], field=field**2)
    with pytest.raises(ValueError, match=r"boundary operators .* must compose to zero"):
        objects.ChainComplex([matrix] * 2, field=field)
    with pytest.raises(ValueError, match="different base fields"):
        objects.ChainComplex.tensor_product(galois.GF(field)(matrix), galois.GF(field**2)(matrix))

    # tensor products of chain complexes over non-commutative rings are not supported
    with pytest.raises(ValueError, match="non-commutative rings are not supported"):
        ring = abstract.GroupRing(abstract.DihedralGroup(3), field=2)
        ring_matrix = abstract.RingArray.build([[1]], ring)
        objects.ChainComplex.tensor_product(ring_matrix, ring_matrix)


def test_chain_complex_action() -> None:
    """Permutation actions on chain complexes."""
    group = abstract.CyclicGroup(3)
    chain = objects.ChainComplex([np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])], field=5)
    representations = [
        abstract.PermutationRepresentation(group, [[1, 2, 0]], 3),
        abstract.PermutationRepresentation(group, [[1, 2, 0]], 4),
    ]
    action = objects.ChainComplexAction(chain, representations)
    assert action.group is group
    assert action.T.representations == tuple(representations[::-1])
    assert np.array_equal(action.T.chain.op(1), chain.op(1).T)

    with pytest.raises(ValueError, match="does not commute"):
        bad_chain = objects.ChainComplex([np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]])])
        objects.ChainComplexAction(bad_chain, representations)
    with pytest.raises(ValueError, match="Expected 2 permutation representations"):
        objects.ChainComplexAction(chain, representations[:1])
    with pytest.raises(ValueError, match="acts on 3 points"):
        objects.ChainComplexAction(chain, [representations[0], representations[0]])
    with pytest.raises(ValueError, match="same Group instance"):
        other_group = abstract.CyclicGroup(3)
        other_representation = abstract.PermutationRepresentation(other_group, [[1, 2, 0]], 4)
        objects.ChainComplexAction(chain, [representations[0], other_representation])
    with pytest.raises(ValueError, match="over a finite field"):
        ring = abstract.GroupRing(abstract.TrivialGroup())
        ring_chain = objects.ChainComplex([abstract.RingArray.build([[0]], ring)])
        objects.ChainComplexAction(
            ring_chain, [abstract.PermutationRepresentation.trivial(group, 1)] * 2
        )


def get_orbit_projection(
    representation_a: abstract.PermutationRepresentation,
    representation_b: abstract.PermutationRepresentation,
) -> npt.NDArray[np.int_]:
    """Project pairs of points onto their orbits under the diagonal group action.

    Orbits are ordered by their smallest pair, and pairs are flattened in row-major order.
    """
    shape = (representation_a.degree, representation_b.degree)
    diagonal_group = comb.PermutationGroup(
        [
            comb.Permutation(
                [
                    np.ravel_multi_index((image_a.array_form[aa], image_b.array_form[bb]), shape)
                    for aa, bb in np.ndindex(shape)
                ]
            )
            for image_a, image_b in zip(
                representation_a.generator_images, representation_b.generator_images
            )
        ]
    )
    orbits = sorted(diagonal_group.orbits(), key=min)
    projection = np.zeros((len(orbits), math.prod(shape)), dtype=int)
    for index, orbit in enumerate(orbits):
        projection[index, list(orbit)] = 1
    return projection


def test_balanced_product() -> None:
    """Balanced products of chain complexes are quotients of tensor products."""
    # non-free actions of a cyclic group
    group = abstract.CyclicGroup(2)
    chain_a = objects.ChainComplex(
        [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])], field=3
    )
    chain_b = objects.ChainComplex(
        [np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])], field=3
    )
    swap_one, swap_two = [1, 0, 2, 3], [1, 0, 3, 2]
    action_a = objects.ChainComplexAction(
        chain_a,
        [
            abstract.PermutationRepresentation(group, [swap_one], 4),
            abstract.PermutationRepresentation(group, [swap_two], 4),
        ],
    )
    action_b = objects.ChainComplexAction(
        chain_b,
        [
            abstract.PermutationRepresentation(group, [swap_one], 4),
            abstract.PermutationRepresentation.trivial(group, 4),
        ],
    )
    assert_valid_balanced_product(action_a, action_b, [10, 20, 8])

    # orbits that contribute to the same boundary operator entry, whose contributions cancel
    regular = abstract.PermutationRepresentation.regular(group)
    chain = objects.ChainComplex([np.ones((2, 1), dtype=int)])
    action = objects.ChainComplexAction(
        chain, [regular, abstract.PermutationRepresentation.trivial(group, 1)]
    )
    balanced = assert_valid_balanced_product(action, action, [2, 2, 1])
    assert not np.any(balanced.op(2))

    # the regular action of a non-abelian group, for which ignoring the distinction between left
    # and right actions would identify too many basis elements
    frobenius_group = abstract.Group(
        comb.Permutation(0, 1, 2, 3, 4, 5, 6), comb.Permutation(1, 2, 4)(3, 6, 5)
    )
    regular = abstract.PermutationRepresentation.regular(frobenius_group)
    chain = objects.ChainComplex([np.eye(frobenius_group.order, dtype=int)])
    action = objects.ChainComplexAction(chain, [regular, regular])
    assert_valid_balanced_product(action, action, [21, 42, 21])

    with pytest.raises(ValueError, match="same field"):
        chain_b = objects.ChainComplex([np.eye(frobenius_group.order, dtype=int)], field=3)
        objects.ChainComplex.balanced_product(
            action, objects.ChainComplexAction(chain_b, [regular, regular])
        )
    with pytest.raises(ValueError, match="same Group instance"):
        other_group = abstract.TrivialGroup()
        other_action = objects.ChainComplexAction(
            objects.ChainComplex([np.eye(1, dtype=int)]),
            [abstract.PermutationRepresentation.trivial(other_group, 1)] * 2,
        )
        objects.ChainComplex.balanced_product(action, other_action)


def assert_valid_balanced_product(
    action_a: objects.ChainComplexAction,
    action_b: objects.ChainComplexAction,
    dims: list[int],
) -> objects.ChainComplex:
    """Check that projecting onto orbits maps a tensor product onto a balanced product."""
    chain_a, chain_b = action_a.chain, action_b.chain
    tensor = objects.ChainComplex.tensor_product(chain_a, chain_b)
    balanced = objects.ChainComplex.balanced_product(action_a, action_b)
    assert [balanced.dim(degree) for degree in range(len(dims))] == dims

    projections = [
        chain_a.field(
            scipy.linalg.block_diag(
                *[
                    get_orbit_projection(
                        action_a.representations[deg_a], action_b.representations[deg_b]
                    )
                    for deg_a, deg_b in objects.ChainComplex._get_degree_pairs(
                        chain_a, chain_b, degree
                    )
                ]
            )
        )
        for degree in range(tensor.num_links + 1)
    ]
    for degree in range(1, tensor.num_links + 1):
        assert np.array_equal(
            balanced.op(degree) @ projections[degree], projections[degree - 1] @ tensor.op(degree)
        )
    return balanced
