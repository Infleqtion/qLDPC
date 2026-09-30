# SPDX-License-Identifier: Apache-2.0

"""Unit tests for objects.py."""

from __future__ import annotations

import itertools
from typing import cast

import galois
import numpy as np
import pytest
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


def get_chain_action_data() -> tuple[
    objects.ChainComplex,
    abstract.Group,
    tuple[abstract.PermutationRepresentation, abstract.PermutationRepresentation],
]:
    """Build a one-link chain with compatible actions having different orbit structures."""
    field = galois.GF(5)
    differential = field(np.column_stack((np.identity(3, dtype=int), np.ones(3, dtype=int))))
    chain = objects.ChainComplex([differential])
    group = abstract.CyclicGroup(3)
    representations = (
        abstract.PermutationRepresentation(group, [[1, 2, 0]], 3),
        abstract.PermutationRepresentation(group, [[1, 2, 0]], 4),
    )
    return chain, group, representations


def test_chain_complex_action() -> None:
    """Construct nontrivial right and left actions with distinct degree-wise orbit structures."""
    chain, group, representations = get_chain_action_data()
    original_field = chain.field

    for right in [True, False]:
        action = objects.ChainComplexAction(chain, representations, right=right)
        assert action.chain is chain
        assert action.representations == representations
        assert action.group is group
        assert action.right is right
        assert action.representation(0) is representations[0]
        assert action.representation(1) is representations[1]
        assert chain.field is original_field

    assert representations[0].generator_images[0].array_form == [1, 2, 0]
    assert representations[1].generator_images[0].array_form == [1, 2, 0, 3]


def test_chain_complex_action_validation() -> None:
    """Reject incompatible chain complexes and permutation representations."""
    chain, group, representations = get_chain_action_data()

    with pytest.raises(TypeError, match="ChainComplex"):
        objects.ChainComplexAction(cast(objects.ChainComplex, object()), representations)
    with pytest.raises(TypeError, match="boolean 'right' value"):
        objects.ChainComplexAction(chain, representations, right=cast(bool, 1))
    with pytest.raises(ValueError, match="exactly 2"):
        objects.ChainComplexAction(chain, representations[:1])
    with pytest.raises(TypeError, match="Every chain degree"):
        invalid_representation = cast(abstract.PermutationRepresentation, object())
        objects.ChainComplexAction(chain, (representations[0], invalid_representation))
    with pytest.raises(ValueError, match=r"degree 1 has degree 3.*dimension 4"):
        wrong_dimension = abstract.PermutationRepresentation(group, [[1, 2, 0]], 3)
        objects.ChainComplexAction(chain, (representations[0], wrong_dimension))
    with pytest.raises(ValueError, match="exact same Group instance"):
        other_group = abstract.CyclicGroup(3)
        other_representation = abstract.PermutationRepresentation(other_group, [[1, 2, 0]], 4)
        objects.ChainComplexAction(chain, (representations[0], other_representation))

    bad_differential = np.column_stack((np.identity(3, dtype=int), [1, 0, 0]))
    bad_chain = objects.ChainComplex([bad_differential], field=5)
    with pytest.raises(ValueError, match="degree 1 is not equivariant under the right action"):
        objects.ChainComplexAction(bad_chain, representations)

    ring = abstract.GroupRing(abstract.TrivialGroup())
    ring_chain = objects.ChainComplex([abstract.RingArray.build([[0]], ring)])
    trivial_representations = (
        abstract.PermutationRepresentation.trivial(group, 1),
        abstract.PermutationRepresentation.trivial(group, 1),
    )
    with pytest.raises(ValueError, match="RingArray-valued"):
        objects.ChainComplexAction(ring_chain, trivial_representations)


def test_chain_complex_action_transpose() -> None:
    """Dualize a right action to a left action without inverting its images."""
    chain, group, representations = get_chain_action_data()
    action = objects.ChainComplexAction(chain, representations)
    dual = action.T

    assert dual.group is group
    assert not dual.right
    assert dual.chain.field is chain.field
    assert dual.representations == representations[::-1]
    assert dual.representation(0).generator_images == representations[1].generator_images
    assert dual.representation(1).generator_images == representations[0].generator_images
    assert np.array_equal(dual.chain.op(1), chain.op(1).T)

    round_trip = dual.T
    assert round_trip.right
    assert round_trip.representations == representations
    assert round_trip.chain.field is chain.field
    assert np.array_equal(round_trip.chain.op(1), chain.op(1))


def test_chain_complex_action_zero_dimension() -> None:
    """Support chain degrees with a zero-dimensional module."""
    field = galois.GF(3)
    chain = objects.ChainComplex([field.Zeros((0, 2))])
    group = abstract.CyclicGroup(2)
    representations = (
        abstract.PermutationRepresentation(group, [[]], 0),
        abstract.PermutationRepresentation(group, [[1, 0]], 2),
    )

    action = objects.ChainComplexAction(chain, representations)
    assert action.representation(0).degree == 0
    assert action.T.representation(1).degree == 0

    with pytest.raises(IndexError, match="out of range"):
        action.representation(-1)
    with pytest.raises(IndexError, match="out of range"):
        action.representation(2)


def get_test_degree_pairs(
    chain_a: objects.ChainComplex, chain_b: objects.ChainComplex, degree: int
) -> list[tuple[int, int]]:
    """Get total-product sectors without using the implementation helper."""
    return [
        (deg_a, degree - deg_a)
        for deg_a in range(min(chain_a.num_links, degree), -1, -1)
        if degree - deg_a <= chain_b.num_links
    ]


def get_diagonal_orbits(
    representation_a: abstract.PermutationRepresentation,
    representation_b: abstract.PermutationRepresentation,
) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate simultaneous-action orbits independently using every group member."""
    members = tuple(representation_a.group.generate())
    remaining = set(
        itertools.product(range(representation_a.degree), range(representation_b.degree))
    )
    orbits = []
    while remaining:
        representative = min(remaining)
        orbit = {
            (
                representation_a(member).apply(representative[0]),
                representation_b(member).apply(representative[1]),
            )
            for member in members
        }
        orbits.append(tuple(sorted(orbit)))
        remaining -= orbit
    return orbits


def get_quotient_basis(
    action_a: objects.ChainComplexAction,
    action_b: objects.ChainComplexAction,
    degree: int,
) -> list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]]:
    """List quotient basis orbits in tensor-sector order."""
    basis: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = []
    for deg_a, deg_b in get_test_degree_pairs(action_a.chain, action_b.chain, degree):
        orbits = get_diagonal_orbits(action_a.representation(deg_a), action_b.representation(deg_b))
        basis.extend(((deg_a, deg_b), orbit) for orbit in orbits)
    return basis


def get_quotient_maps(
    action_a: objects.ChainComplexAction,
    action_b: objects.ChainComplexAction,
    degree: int,
    basis: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] | None = None,
) -> tuple[galois.FieldArray, galois.FieldArray]:
    """Build an orbit projection and a representative lift into the full tensor module."""
    chain_a, chain_b = action_a.chain, action_b.chain
    basis = basis if basis is not None else get_quotient_basis(action_a, action_b, degree)

    sector_offsets = {}
    full_dimension = 0
    for deg_a, deg_b in get_test_degree_pairs(chain_a, chain_b, degree):
        sector_offsets[deg_a, deg_b] = full_dimension
        full_dimension += chain_a.dim(deg_a) * chain_b.dim(deg_b)

    projection = chain_a.field.Zeros((len(basis), full_dimension))
    lift = chain_a.field.Zeros((full_dimension, len(basis)))
    for quotient_index, ((deg_a, deg_b), orbit) in enumerate(basis):
        sector_offset = sector_offsets[deg_a, deg_b]
        dim_b = chain_b.dim(deg_b)
        for index_a, index_b in orbit:
            projection[quotient_index, sector_offset + index_a * dim_b + index_b] = 1
        representative_a, representative_b = orbit[0]
        lift[sector_offset + representative_a * dim_b + representative_b, quotient_index] = 1

    return projection, lift


def assert_balanced_quotient_diagram(
    action_a: objects.ChainComplexAction, action_b: objects.ChainComplexAction
) -> objects.ChainComplex:
    """Check the quotient diagram against a separately materialized tensor product."""
    tensor = objects.ChainComplex.tensor_product(action_a.chain, action_b.chain)
    balanced = objects.ChainComplex.balanced_product(action_a, action_b)
    projections = [
        get_quotient_maps(action_a, action_b, degree)[0] for degree in range(tensor.num_links + 1)
    ]

    for degree in range(1, tensor.num_links + 1):
        tensor_op = tensor.op(degree)
        balanced_op = balanced.op(degree)
        assert isinstance(tensor_op, galois.FieldArray)
        assert isinstance(balanced_op, galois.FieldArray)
        assert np.array_equal(
            balanced_op @ projections[degree],
            projections[degree - 1] @ tensor_op,
        )

    return balanced


@pytest.mark.parametrize("field_order", [2, 3])
def test_balanced_product_trivial_group(field_order: int) -> None:
    """A balanced product by the trivial group is the ordinary tensor product."""
    field = galois.GF(field_order)
    chain_a = objects.ChainComplex([field([[1, 1, 0], [0, 1, 1]])])
    chain_b = objects.ChainComplex([field([[1, 0], [1, 1], [0, 1]])])
    group = abstract.TrivialGroup()
    representations_a = tuple(
        abstract.PermutationRepresentation.trivial(group, chain_a.dim(degree))
        for degree in range(chain_a.num_links + 1)
    )
    representations_b = tuple(
        abstract.PermutationRepresentation.trivial(group, chain_b.dim(degree))
        for degree in range(chain_b.num_links + 1)
    )
    action_a = objects.ChainComplexAction(chain_a, representations_a)
    action_b = objects.ChainComplexAction(chain_b, representations_b, right=False)

    balanced = assert_balanced_quotient_diagram(action_a, action_b)
    tensor = objects.ChainComplex.tensor_product(chain_a, chain_b)
    assert balanced.field is field
    assert balanced.ring is None
    assert all(
        np.array_equal(op_balanced, op_tensor)
        for op_balanced, op_tensor in zip(balanced.ops, tensor.ops)
    )


def test_balanced_product_nonfree_actions() -> None:
    """Count quotient sectors for C2 actions with degree-dependent fixed points."""
    field = galois.GF(5)
    group = abstract.CyclicGroup(2)
    one_swap = [1, 0, 2, 3]
    two_swaps = [1, 0, 3, 2]
    identity = [0, 1, 2, 3]

    chain_a = objects.ChainComplex(
        [
            field(
                [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],
                    [0, 0, 0, 0],
                ]
            )
        ]
    )
    action_a = objects.ChainComplexAction(
        chain_a,
        (
            abstract.PermutationRepresentation(group, [one_swap], 4),
            abstract.PermutationRepresentation(group, [two_swaps], 4),
        ),
    )

    chain_b = objects.ChainComplex(
        [
            field(
                [
                    [1, 0, 0, 0],
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                ]
            )
        ]
    )
    action_b = objects.ChainComplexAction(
        chain_b,
        (
            abstract.PermutationRepresentation(group, [one_swap], 4),
            abstract.PermutationRepresentation(group, [identity], 4),
        ),
        right=False,
    )

    balanced = assert_balanced_quotient_diagram(action_a, action_b)
    assert [balanced.dim(degree) for degree in range(3)] == [10, 20, 8]
    middle_basis = get_quotient_basis(action_a, action_b, 1)
    assert [
        sum(degrees == sector for degrees, _ in middle_basis) for sector in [(1, 0), (0, 1)]
    ] == [8, 12]


@pytest.mark.parametrize("group_order", [2, 3])
def test_balanced_product_modular_cancellation(group_order: int) -> None:
    """Accumulate orbit coefficients in the field, including modular cancellation."""
    field = galois.GF(group_order)
    group = abstract.CyclicGroup(group_order)
    chain = objects.ChainComplex([field.Ones((group_order, 1))])
    representations = (
        abstract.PermutationRepresentation.regular(group),
        abstract.PermutationRepresentation.trivial(group, 1),
    )
    action_a = objects.ChainComplexAction(chain, representations)
    action_b = objects.ChainComplexAction(chain, representations, right=False)

    balanced = assert_balanced_quotient_diagram(action_a, action_b)
    assert [balanced.dim(degree) for degree in range(3)] == [group_order, 2, 1]
    assert not np.any(balanced.op(2))
    assert np.any(balanced.op(1))


def test_balanced_product_nonabelian_diagonal_action() -> None:
    """Use a Frobenius group to distinguish diagonal from inverse-paired generators."""
    group = abstract.Group(
        comb.Permutation(0, 1, 2, 3, 4, 5, 6),
        comb.Permutation(1, 2, 4)(3, 6, 5),
    )
    assert group.order == 21
    field = galois.GF(2)
    chain = objects.ChainComplex([field.Identity(group.order)])
    regular = abstract.PermutationRepresentation.regular(group)
    action_a = objects.ChainComplexAction(chain, (regular, regular))
    action_b = objects.ChainComplexAction(chain, (regular, regular), right=False)

    assert len(get_diagonal_orbits(regular, regular)) == 21
    balanced = objects.ChainComplex.balanced_product(action_a, action_b)
    assert [balanced.dim(degree) for degree in range(3)] == [21, 42, 21]


def get_basis_mapping(
    field: type[galois.FieldArray],
    canonical_basis: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]],
    alternative_basis: list[tuple[tuple[int, int], tuple[tuple[int, int], ...]]],
) -> galois.FieldArray:
    """Map canonical quotient coordinates to a reordered orbit basis."""
    mapping = field.Zeros((len(alternative_basis), len(canonical_basis)))
    canonical_orbits = [(degrees, frozenset(orbit)) for degrees, orbit in canonical_basis]
    for alternative_index, (degrees, orbit) in enumerate(alternative_basis):
        canonical_index = canonical_orbits.index((degrees, frozenset(orbit)))
        mapping[alternative_index, canonical_index] = 1
    return mapping


def test_balanced_product_representative_independence() -> None:
    """Changing every orbit representative only changes the explicitly ordered quotient basis."""
    field = galois.GF(5)
    group = abstract.CyclicGroup(3)
    chain = objects.ChainComplex([field.Identity(group.order)])
    regular = abstract.PermutationRepresentation.regular(group)
    action_a = objects.ChainComplexAction(chain, (regular, regular))
    action_b = objects.ChainComplexAction(chain, (regular, regular), right=False)
    tensor = objects.ChainComplex.tensor_product(chain, chain)
    balanced = objects.ChainComplex.balanced_product(action_a, action_b)

    canonical_bases = [get_quotient_basis(action_a, action_b, degree) for degree in range(3)]
    alternative_bases = [
        [(degrees, orbit[1:] + orbit[:1]) for degrees, orbit in reversed(canonical_basis)]
        for canonical_basis in canonical_bases
    ]
    assert all(
        orbit[0] != alternative_orbit[0]
        for canonical_basis, alternative_basis in zip(canonical_bases, alternative_bases)
        for (_, orbit), (_, alternative_orbit) in zip(canonical_basis, reversed(alternative_basis))
    )

    basis_mappings = [
        get_basis_mapping(field, canonical_basis, alternative_basis)
        for canonical_basis, alternative_basis in zip(canonical_bases, alternative_bases)
    ]
    quotient_maps = [
        get_quotient_maps(action_a, action_b, degree, alternative_basis)
        for degree, alternative_basis in enumerate(alternative_bases)
    ]
    for degree in range(1, 3):
        tensor_op = tensor.op(degree)
        balanced_op = balanced.op(degree)
        assert isinstance(tensor_op, galois.FieldArray)
        assert isinstance(balanced_op, galois.FieldArray)
        target_projection = quotient_maps[degree - 1][0]
        source_lift = quotient_maps[degree][1]
        alternative_op = target_projection @ tensor_op @ source_lift
        assert np.array_equal(
            alternative_op @ basis_mappings[degree],
            basis_mappings[degree - 1] @ balanced_op,
        )


def test_balanced_product_zero_dimensions() -> None:
    """Preserve empty modules and zero-row and zero-column differentials."""
    field = galois.GF(3)
    group = abstract.CyclicGroup(2)
    chain = objects.ChainComplex([field.Zeros((0, 2))])
    representations = (
        abstract.PermutationRepresentation(group, [[]], 0),
        abstract.PermutationRepresentation(group, [[1, 0]], 2),
    )
    action_a = objects.ChainComplexAction(chain, representations)
    action_b = action_a.T

    balanced = assert_balanced_quotient_diagram(action_a, action_b)
    assert [balanced.dim(degree) for degree in range(3)] == [0, 2, 0]
    assert [op.shape for op in balanced.ops] == [(0, 2), (2, 0)]


def test_balanced_product_validation() -> None:
    """Reject invalid action types, sides, fields, and group identities."""
    field = galois.GF(2)
    group = abstract.TrivialGroup()
    chain = objects.ChainComplex([field.Identity(1)])
    representations = (
        abstract.PermutationRepresentation.trivial(group, 1),
        abstract.PermutationRepresentation.trivial(group, 1),
    )
    right_action = objects.ChainComplexAction(chain, representations)
    left_action = objects.ChainComplexAction(chain, representations, right=False)

    invalid_action = cast(objects.ChainComplexAction, object())
    with pytest.raises(TypeError, match="two ChainComplexAction"):
        objects.ChainComplex.balanced_product(invalid_action, left_action)
    with pytest.raises(TypeError, match="two ChainComplexAction"):
        objects.ChainComplex.balanced_product(right_action, invalid_action)
    with pytest.raises(ValueError, match=r"first.*right action"):
        objects.ChainComplex.balanced_product(left_action, left_action)
    with pytest.raises(ValueError, match=r"second.*left action"):
        objects.ChainComplex.balanced_product(right_action, right_action)

    other_field = galois.GF(3)
    other_chain = objects.ChainComplex([other_field.Identity(1)])
    other_field_representations = (
        abstract.PermutationRepresentation.trivial(group, 1),
        abstract.PermutationRepresentation.trivial(group, 1),
    )
    other_field_action = objects.ChainComplexAction(
        other_chain, other_field_representations, right=False
    )
    with pytest.raises(ValueError, match="same finite field"):
        objects.ChainComplex.balanced_product(right_action, other_field_action)

    other_group = abstract.TrivialGroup()
    other_group_representations = (
        abstract.PermutationRepresentation.trivial(other_group, 1),
        abstract.PermutationRepresentation.trivial(other_group, 1),
    )
    other_group_action = objects.ChainComplexAction(chain, other_group_representations, right=False)
    with pytest.raises(ValueError, match="exact same Group instance"):
        objects.ChainComplex.balanced_product(right_action, other_group_action)
