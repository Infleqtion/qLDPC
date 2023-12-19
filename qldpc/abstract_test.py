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
import numpy as np

from qldpc import abstract


def test_permutation_group() -> None:
    """Permutation members and group construction."""
    gens = [abstract.GroupMember(seq) for seq in ([0, 1, 2], [1, 2, 0], [2, 0, 1])]
    assert gens[0] < gens[1] < gens[2]

    group = abstract.Group.from_generators(*gens)
    assert all(perm in group for perm in gens)
    assert len(group.generators) == 2


def test_trivial_group() -> None:
    """Trivial group tests."""
    group = abstract.TrivialGroup()
    group_squared = group * group
    assert group == group_squared
    assert group.lift_dim == 1
    assert group_squared.lift_dim == 1
    assert np.array_equal(group.lift(group.identity), np.array(1, ndmin=2))


def test_lift() -> None:
    """Lift named group elements."""
    assert_valid_lift(abstract.TrivialGroup())
    assert_valid_lift(abstract.CyclicGroup(3))
    assert_valid_lift(abstract.DihedralGroup(3))
    assert_valid_lift(abstract.QuaternionGroup())


def assert_valid_lift(group: abstract.Group) -> None:
    """Assert faithfulness of the group representation (lift)."""
    assert all(
        aa == bb or not np.array_equal(group.lift(aa), group.lift(bb))
        for aa in group.generate()
        for bb in group.generate()
    )
    assert all(
        np.array_equal(
            group.lift(aa) @ group.lift(bb),
            group.lift(aa * bb),
        )
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
    group = cycle * cycle
    assert_valid_lift(group)
    assert group == abstract.Group.product(cycle, cycle)
    assert group.generators == [shift @ identity, identity @ shift]
    assert np.array_equal(table, group.table)
    assert np.array_equal(table, abstract.Group.from_table(table).table)


def test_algebra() -> None:
    """Construct elements of a group algebra."""
    group = abstract.TrivialGroup()
    zero = abstract.Element(group, field=3)
    one = abstract.Element(group, field=3).one()
    assert zero.group == group
    assert zero.field == abstract.FiniteField(3)
    assert one + one + one == group.identity + 2 * one == zero
    assert group.identity * one == one * group.identity == one**2 == one
    assert np.array_equal(zero.lift(), np.array(0, ndmin=2))
    assert np.array_equal(one.lift(), np.array(1, ndmin=2))


def test_protograph() -> None:
    """Construct and lift a protograph."""
    matrix = np.random.randint(2, size=(3, 3))
    protograph = abstract.TrivialGroup.to_protograph(matrix)
    assert protograph.group == abstract.TrivialGroup()
    assert 1 * protograph == protograph * 1 == protograph
    assert protograph == abstract.Protograph(protograph)
    assert np.array_equal(protograph.lift(), matrix)


def test_transpose() -> None:
    """Transpose various objects."""
    group = abstract.CyclicGroup(4)
    for member in group.generate():
        element = abstract.Element(group, member)
        assert element.T.T == element

    x0, x1, x2, x3 = group.generate()
    matrix = [[x0, 0, x1], [x2, 0, x3]]
    protograph = abstract.Protograph.build(group, matrix)
    assert protograph.T.T == protograph
