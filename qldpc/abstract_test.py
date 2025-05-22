"""Unit tests for abstract.py

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

import math
import unittest.mock

import galois
import numpy as np
import pytest

from qldpc import abstract


def test_permutation_group() -> None:
    """Permutation members and group construction."""
    gens = [abstract.GroupMember(seq) for seq in ([0, 1, 2], [1, 2, 0], [2, 0, 1])]
    assert gens[0] < gens[1] < gens[2]

    group = abstract.Group(*gens)
    assert all(perm in group for perm in gens)
    assert len(group.generators) == 2
    assert group.random() in group
    assert group.random(seed=0) == group.random(seed=0)
    assert group.to_sympy() == group._group
    assert hash(group) == hash(group.to_sympy())

    assert abstract.Group.from_generating_mats([[1]]) == abstract.CyclicGroup(1)

    with pytest.raises(ValueError, match="inconsistent"):
        gen = galois.GF(2)([[1]])
        abstract.Group.from_generating_mats(gen, field=3)

    assert isinstance(hash(group.hashable_generators()), int)


def test_trivial_group() -> None:
    """Trivial group tests."""
    group = abstract.TrivialGroup()
    group_squared = group**2
    assert group == group_squared == group * group
    assert group.lift_dim == 1
    assert group_squared.lift_dim == 1
    assert group.random() == group.identity
    assert np.array_equal(group.lift(group.identity), np.array(1, ndmin=2))
    assert group == abstract.Group.from_generating_mats()
    assert str(group) == "TrivialGroup"


def test_lift() -> None:
    """Lift named group elements."""
    assert_valid_lift(abstract.TrivialGroup())
    assert_valid_lift(abstract.CyclicGroup(3))
    assert_valid_lift(abstract.AbelianGroup(2, 3))
    assert_valid_lift(abstract.AbelianGroup(2, 3, product_lift=True))
    assert_valid_lift(abstract.DihedralGroup(3))
    assert_valid_lift(abstract.AlternatingGroup(3))
    assert_valid_lift(abstract.SymmetricGroup(3))
    assert_valid_lift(abstract.QuaternionGroup())


def assert_valid_lift(group: abstract.Group) -> None:
    """Assert faithfulness of the group representation (lift)."""
    assert all(
        aa == bb or not np.array_equal(group.lift(aa), group.lift(bb))
        for aa in group.generate()
        for bb in group.generate()
    )
    assert all(
        np.array_equal(group.lift(aa) @ group.lift(bb), group.lift(aa * bb))
        for aa in group.generate()
        for bb in group.generate()
    )


def test_group_product() -> None:
    """Direct product of groups."""
    cycle = abstract.CyclicGroup(2)
    identity, shift = cycle.generate()
    table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    group = abstract.Group.product(cycle, cycle)
    assert_valid_lift(group)
    assert group.generators == [shift @ identity, identity @ shift]
    assert np.array_equal(table, group.table)
    assert np.array_equal(table, abstract.Group.from_table(table).table)

    # product of groups over different fields results in a group over the binary field
    assert abstract.TrivialGroup(2) * abstract.TrivialGroup(3) == abstract.TrivialGroup(2)


def test_algebra() -> None:
    """Construct elements of a group algebra."""
    group = abstract.TrivialGroup(field=3)
    zero = abstract.Element(group)
    one = abstract.Element(group).one()
    assert bool(one) and not bool(zero)
    assert zero.group == group
    assert one + 2 == group.identity + 2 * one == -one + 1 == one - 1 == zero
    assert group.identity * one == one * group.identity == one**2 == one
    assert np.array_equal(zero.lift(), np.array(0, ndmin=2))
    assert np.array_equal(one.lift(), np.array(1, ndmin=2))


def test_protograph() -> None:
    """Construct and lift a protograph."""
    matrix = np.random.randint(2, size=(3, 3))
    protograph = abstract.TrivialGroup.to_protograph(matrix)
    assert protograph.group == abstract.TrivialGroup()
    assert protograph.field == abstract.TrivialGroup().field
    assert np.array_equal(protograph.lift(), matrix)
    assert np.array_equal((protograph @ protograph).lift(), protograph.lift() @ protograph.lift())

    # fail to construct a valid protograph
    with pytest.raises(ValueError, match="must be Element-valued"):
        abstract.Protograph([[0]])
    with pytest.raises(ValueError, match="Inconsistent base groups"):
        groups = [abstract.TrivialGroup(), abstract.CyclicGroup(1)]
        abstract.Protograph([[abstract.Element(group) for group in groups]])
    with pytest.raises(ValueError, match="Cannot determine the underlying group"):
        abstract.Protograph([])
    with pytest.raises(ValueError, match="different base groups"):
        new_protograph = abstract.Protograph.build(abstract.CyclicGroup(1), [[1]])
        protograph @ new_protograph


def test_transpose() -> None:
    """Transpose various objects."""
    group = abstract.CyclicGroup(4)
    for member in group.generate():
        element = abstract.Element(group, member)
        assert element.T.T == element

    x0, x1, x2, x3 = group.generate()
    matrix = [[x0, 0, x1], [x2, 0, abstract.Element(group, x3)]]
    protograph = abstract.Protograph.build(group, matrix)
    assert np.array_equal(protograph.T.T, protograph)


def test_random_symmetric_subset() -> None:
    """Cover Group.random_symmetric_subset."""
    group = abstract.CyclicGroup(2) * abstract.CyclicGroup(3)
    for seed in [0, 1]:
        subset = group.random_symmetric_subset(size=2, seed=seed)
        assert subset == {~member for member in subset}

    subset = group.random_symmetric_subset(size=1, exclude_identity=False, seed=0)
    assert subset == {group.identity}

    with pytest.raises(ValueError, match="must have a size between"):
        group.random_symmetric_subset(size=0)


@pytest.mark.parametrize("dimension,field,linear_rep", [(2, 4, True), (2, 2, False)])
def test_SL(dimension: int, field: int, linear_rep: bool) -> None:
    """Special linear group."""
    group = abstract.SL(dimension, field=field, linear_rep=linear_rep)
    order = np.prod([field**dimension - field**jj for jj in range(dimension)]) // (field - 1)
    mats = tuple(abstract.SL.iter_mats(dimension, field))
    assert group.order == len(mats) == order

    gens = group.generators
    gen_mats = group.get_generating_mats(dimension, field)
    assert np.array_equal(group.lift(gens[0]), gen_mats[0])
    assert np.array_equal(group.lift(gens[1]), gen_mats[1])


@pytest.mark.parametrize("dimension,field,linear_rep", [(2, 2, True), (2, 3, False)])
def test_PSL(dimension: int, field: int, linear_rep: bool) -> None:
    """Projective special linear group."""
    group = abstract.PSL(dimension, field, linear_rep=linear_rep)
    order_SL = np.prod([field**dimension - field**jj for jj in range(dimension)]) // (field - 1)
    order = order_SL // math.gcd(dimension, field - 1)
    mats = tuple(abstract.PSL.iter_mats(dimension, field))
    assert group.order == len(mats) == order

    if field == 2:
        gens = group.generators
        gen_mats = group.get_generating_mats(dimension, field)
        assert np.array_equal(group.lift(gens[0]), gen_mats[0])
        assert np.array_equal(group.lift(gens[1]), gen_mats[1])


def test_small_group() -> None:
    """Groups indexed by the GAP computer algebra system."""
    order, index = 2, 1
    desired_group = abstract.CyclicGroup(order)

    # invalid group index
    with (
        pytest.raises(ValueError, match="Index for SmallGroup"),
        unittest.mock.patch("qldpc.external.groups.get_small_group_number", return_value=index),
    ):
        abstract.SmallGroup(order, 0)

    # everything works as expected
    generators = [tuple(gen.array_form) for gen in desired_group.generators]
    with (
        unittest.mock.patch("qldpc.external.groups.get_small_group_number", return_value=index),
        unittest.mock.patch("qldpc.external.groups.get_generators", return_value=generators),
    ):
        group = abstract.SmallGroup(order, index)
        assert group.generators == desired_group.generators
        assert list(abstract.SmallGroup.generator(order)) == [desired_group]

        # retrieve group structure
        structure = "test"
        with unittest.mock.patch(
            "qldpc.external.groups.get_small_group_structure", return_value=structure
        ):
            assert group.structure == structure

    # cover a special case
    with unittest.mock.patch("qldpc.external.groups.get_small_group_number", return_value=1):
        group = abstract.SmallGroup(1, 1)
    assert group == abstract.TrivialGroup()
    assert group.random() == group.identity
