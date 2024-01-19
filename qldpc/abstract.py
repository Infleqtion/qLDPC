"""Module for abstract algebra: groups, algebras, and representations thereof

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

All groups in this module are finite, and represented under the hood as a Sympy PermutationGroup, or
a subgroup of the symmetric group.  Group members are essentially represented by Sympy Permutation
objects.  Groups additionally come equipped with a representation, or "lift", that maps group
elements to square matrices, such that the group action gets lifted to matrix multiplication.

!!! WARNINGS !!!

Whereas matrices are "left-acting" (that is, act on objects from the left) by standard convention,
Sympy permutations are "right-acting", which is to say that the action of two permutations p and q
on an integer i compose as (p*q)(i) = q(p(i)) = i^p^q.  To preserve the order of products before and
after lifting permutations to matrices, which ensures that the lift L(p*q) = L(p) @ L(q), we
therefore make representations likewise right-acting, which is to say that a permutation matrix M
transposes a vector v as v --> v @ M.  In practice, this simply means that matrices are the
transpose of what one might expect.

This module only supports representations of group members by orthogonal matrices over finite
fields.  The restriction to orthogonal representations is not fundamental, but is convenient for
identifying the "transpose" a group member p with respect to a representation (lift) L.  This
transpose is defined as the group member p.T for which L(p.T) = L(p).T.  If the representation is
orthogonal, then p.T is equal to the inverse ~p = p**-1.
"""
from __future__ import annotations

import collections
import copy
import functools
import itertools
from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

import galois
import numpy as np
import numpy.typing as npt
import sympy.combinatorics as comb
from sympy.combinatorics.perm_groups import PermutationGroup

DEFAULT_FIELD_ORDER = 2

################################################################################
# groups


UnknownType = TypeVar("UnknownType")


class GroupMember(comb.Permutation):
    """Wrapper for Sympy Permutation class.

    Supports sorting permutations (by their rank), and taking their tensor product.
    """

    def __mul__(self, other: UnknownType) -> UnknownType:
        if isinstance(other, GroupMember):
            return GroupMember(super().__mul__(other).array_form)  # type:ignore[return-value]
        elif hasattr(other, "__rmul__"):
            return other.__rmul__(self)
        return NotImplemented  # pragma: no cover

    def __add__(self, other: UnknownType) -> UnknownType:
        if hasattr(other, "__radd__"):
            return other.__radd__(self)
        return NotImplemented  # pragma: no cover

    def __lt__(self, other: GroupMember) -> bool:
        return self.rank() < other.rank()

    def __matmul__(self, other: GroupMember) -> GroupMember:
        """Take the "tensor product" of two permutations."""
        return GroupMember(self.array_form + [val + self.size for val in other.array_form])


IntegerArray = npt.NDArray[np.int_]
Lift = Callable[[GroupMember], IntegerArray]
IntegerLift = Callable[[int], IntegerArray]


def default_lift(member: GroupMember) -> IntegerArray:
    """Default lift: represent a permutation object by a permutation matrix.

    For consistency with how Sympy composes permutations, this matrix is right-acting, meaning that
    it acts on a vector p from the right: p --> p @ M.
    """
    matrix = np.zeros((member.size,) * 2, dtype=int)
    for ii in range(member.size):
        matrix[ii, member.apply(ii)] = 1
    return matrix


class Group:
    """Base class for a finite group.

    Under the hood, a Group is represented by a Sympy PermutationGroup.
    Group elements are represented by Sympy permutations.

    A group additionally comes equipped with a "lift", or a representation that maps group elements
    to orthogonal matrices over a finite field.  The group action gets lifted to matrix
    multiplication.  If no lift is provided, the group will default to the representation of group
    members by explicit permutation matrices.
    """

    _group: PermutationGroup
    _field: type[galois.FieldArray]
    _lift: Lift

    def __init__(
        self, group: PermutationGroup, field: int | None = None, lift: Lift | None = None
    ) -> None:
        self._group = group
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        self._lift = lift if lift is not None else default_lift

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Group) and self._field == other._field and self._group == other._group
        )

    def __mul__(self, other: Group) -> Group:
        """Direct product of two groups."""
        if self.field != other.field:
            raise ValueError("Cannot multiply groups with lifts defined over different fields.")
        permutation_group = self._group * other._group

        def lift(member: GroupMember) -> galois.FieldArray:
            degree = self._group.degree
            left = member.array_form[:degree]
            right = [index - degree for index in member.array_form[degree:]]
            matrix = np.kron(self.lift(GroupMember(left)), other.lift(GroupMember(right)))
            return self.field(matrix)

        return Group(permutation_group, self.field.order, lift)

    def __contains__(self, member: GroupMember) -> bool:
        return comb.Permutation(member.array_form) in self._group

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this group."""
        return self._field

    def order(self) -> int:
        """Number of members in this group."""
        return self._group.order()

    @property
    def generators(self) -> Sequence[GroupMember]:
        """Generators of this group."""
        return [GroupMember(member.array_form) for member in self._group.generators]

    def generate(self) -> Iterator[GroupMember]:
        """Iterate over all group members."""
        for member in self._group.generate():
            yield GroupMember(member.array_form)

    @property
    def identity(self) -> GroupMember:
        """The identity element of this group."""
        return GroupMember(self._group.identity.array_form)

    @classmethod
    def product(cls, *groups: Group, repeat: int = 1) -> Group:
        """Direct product of Groups."""
        return functools.reduce(cls.__mul__, groups * repeat)

    def lift(self, member: GroupMember) -> galois.FieldArray:
        """Lift a group member to its representation by an orthogonal matrix."""
        return self.field(self._lift(member))

    @functools.cached_property
    def lift_dim(self) -> int:
        """Dimension of the repesentation for this group."""
        return self._lift(self.generators[0]).shape[0]

    @functools.cached_property
    def table(self) -> IntegerArray:
        """Multiplication (Cayley) table for this group."""
        members = {member: idx for idx, member in enumerate(self.generate())}
        return np.array(
            [members[aa * bb] for aa in self.generate() for bb in self.generate()],
            dtype=int,
        ).reshape((self.order(),) * 2)

    @classmethod
    def from_table(
        cls,
        table: IntegerArray | Sequence[Sequence[int]],
        field: int | None = None,
        integer_lift: IntegerLift | None = None,
    ) -> Group:
        """Construct a group from a multiplication (Cayley) table."""

        if integer_lift is None:
            group = PermutationGroup(*[GroupMember(row) for row in table])
            return Group(group, lift=default_lift)

        members = {GroupMember(row): idx for idx, row in enumerate(table)}

        def lift(member: GroupMember) -> IntegerArray:
            return integer_lift(members[member])

        return Group(PermutationGroup(*members.keys()), field, lift)

    @classmethod
    def from_generators(
        cls, *generators: GroupMember, field: int | None = None, lift: Lift | None = None
    ) -> Group:
        """Construct a group from generators."""
        return Group(PermutationGroup(*generators), field, lift)


################################################################################
# elements of a group algebra


class Element:
    """An element of a group algebra over a finite field F_q.

    Each Element x is a sum of group members with coefficients in F_q:
    x = sum_{g in G} x_g g, with each x_g in F_q.

    The field F_q is taken to be the same as that of the representation of the group.
    """

    _group: Group
    _vec: collections.defaultdict[GroupMember, galois.FieldArray]

    def __init__(self, group: Group, *members: GroupMember):
        self._group = group
        self._vec = collections.defaultdict(lambda: self.field(0))
        for member in members:
            self._vec[member] += self.field(1)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Element)
            and self._group == other._group
            and all(self._vec[member] == other._vec[member] for member in self._vec)
            and all(self._vec[member] == other._vec[member] for member in other._vec)
        )

    def __iter__(self) -> Iterator[tuple[GroupMember, galois.FieldArray]]:
        yield from self._vec.items()

    def __add__(self, other: GroupMember | Element) -> Element:
        new_element = self.copy()

        if isinstance(other, GroupMember):
            new_element._vec[other] += self.field(1)
            return new_element

        # isinstance(other, Element)
        for member, val in other:
            new_element._vec[member] += val
        return new_element

    def __sub__(self, other: GroupMember | Element) -> Element:
        return self + (-1) * other

    def __radd__(self, other: GroupMember) -> Element:
        return self + other

    def __mul__(self, other: int | GroupMember | Element) -> Element:
        new_element = self.zero()

        if isinstance(other, int):
            # multiply coefficients by 'other'
            for member, val in self:
                new_element._vec[member] = val * other
            return new_element

        if isinstance(other, GroupMember):
            # multiply group members by 'other'
            for member, val in self:
                new_element._vec[member * other] = val

        # collect and multiply pairs of terms from 'self' and 'other'
        for (aa, x_a), (bb, y_b) in itertools.product(self, other):
            new_element._vec[aa * bb] += x_a * y_b
        return new_element

    def __rmul__(self, other: int | GroupMember) -> Element:
        if isinstance(other, int):
            return self * other

        # multiply group members by "other"
        new_element = self.zero()
        for member, val in self:
            new_element._vec[other * member] = val
        return new_element

    def __neg__(self) -> Element:
        return self * (-1)

    def __pow__(self, power: int) -> Element:
        return functools.reduce(Element.__mul__, [self] * power, self.one())

    def copy(self) -> Element:
        """Copy of self."""
        element = self.zero()
        for member, val in self:
            element._vec[member] = copy.deepcopy(val)
        return element

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this algebra."""
        return self.group.field

    @property
    def group(self) -> Group:
        """Base group of this algebra."""
        return self._group

    def lift(self) -> galois.FieldArray:
        """Lift this element using the underlying group representation."""
        return sum(
            (val * self._group.lift(member) for member, val in self),
            start=self.field.Zeros((self._group.lift_dim,) * 2),
        )

    def zero(self) -> Element:
        """Zero (additive identity) element."""
        return Element(self._group)

    def one(self) -> Element:
        """One (multiplicative identity) element."""
        return Element(self._group, self._group.identity)

    @property
    def T(self) -> Element:
        """Transpose of this element.

        If this element is x = sum_{g in G) x_g g, return x.T = sum_{g in G} x_g g.T, where g.T is
        the group member for which the lift L(g.T) = L(g).T.  The fact that group members get lifted
        to orthogonal matrices implies that g.T = ~g = g**-1.
        """
        new_element = self.zero()
        for member, val in self:
            new_element._vec[~member] = val
        return new_element


################################################################################
# protographs: Element-valued matrices


ObjectMatrix = npt.NDArray[np.object_] | Sequence[Sequence[object]]


class Protograph:
    """Matrix with Element entries."""

    _matrix: npt.NDArray[np.object_]

    def __init__(self, matrix: Protograph | ObjectMatrix) -> None:
        if isinstance(matrix, Protograph):
            self._matrix = matrix.matrix
        else:
            self._matrix = np.array(matrix, ndmin=2)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Protograph) and np.array_equal(self._matrix, other._matrix)

    def __rmul__(self, val: int) -> Protograph:
        return Protograph(self._matrix * val)

    def __mul__(self, val: int) -> Protograph:
        return val * self

    @property
    def matrix(self) -> npt.NDArray[np.object_]:
        """Element-valued numpy matrix of this protograph."""
        return self._matrix

    @property
    def shape(self) -> tuple[int, ...]:
        """Dimensions (shape) of this protograph."""
        return self._matrix.shape

    @property
    def group(self) -> Group:
        """Group associated with this protograph."""
        return self._matrix[0, 0].group

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this protograph."""
        return self.group.field

    def lift(self) -> galois.FieldArray:
        """Block matrix obtained by lifting each entry of the protograph."""
        vals = [val.lift() for val in self.matrix.ravel()]
        tensor = np.transpose(np.reshape(vals, self.shape + vals[0].shape), [0, 2, 1, 3])
        rows = tensor.shape[0] * tensor.shape[1]
        cols = tensor.shape[2] * tensor.shape[3]
        return tensor.reshape(rows, cols)  # type:ignore[return-value]

    @property
    def T(self) -> Protograph:
        """Transpose of this protograph, which also transposes every matrix entry."""
        entries = [entry.T for entry in self._matrix.ravel()]
        return Protograph(np.array(entries).reshape(self._matrix.shape).T)

    @classmethod
    def build(cls, group: Group, matrix: ObjectMatrix, *, field: int = 2) -> Protograph:
        """Construct a protograph.

        The constructed protograph is built from (i) a group, and (ii) a matrix populated by group
        members or zero/"falsy" entries.  The protograph is obtained by elevating the group memebers
        to elements of the group algebra (over the prime number field).  Zero/"falsy" entries of the
        matrix are interpreted as zeros of the group algebra.
        """
        matrix = np.array(matrix)
        vals = [Element(group, member) if member else Element(group) for member in matrix.ravel()]
        return Protograph(np.array(vals, dtype=object).reshape(matrix.shape))


################################################################################
# named groups


class TrivialGroup(Group):
    """The trivial group with one member: the identity."""

    def __init__(self, field: int | None = None) -> None:
        super().__init__(
            PermutationGroup(GroupMember()),
            field,
            lambda _: np.array(1, ndmin=2, dtype=int),
        )

    @classmethod
    def to_protograph(
        cls, matrix: IntegerArray | Sequence[Sequence[int]], field: int | None = None
    ) -> Protograph:
        """Convert a matrix of 0s and 1s into a protograph of the trivial group."""
        matrix = np.array(matrix)
        group = TrivialGroup(field)
        zero = Element(group)
        unit = Element(group, group.identity)
        terms = np.array([val * unit if val else zero for val in matrix.ravel()], dtype=object)
        return Protograph(terms.reshape(matrix.shape))


class CyclicGroup(Group):
    """Cyclic group of a specified order.

    The cyclic group has one generator, g.  All members of the cyclic group of order R can be
    written as g^p for an integer power p in {0, 1, ..., R-1}.  The member g^p can be represented by
    (that is, lifted to) an R×R "shift matrix", or the identity matrix with all rows shifted down
    (equivalently, all columns shifted right) by p.  That is, the lift L(g^p) acts on a standard
    basis vector <i| as <i| L(g^p) = < i + p mod R |.
    """

    def __init__(self, order: int) -> None:
        super().__init__(comb.named_groups.CyclicGroup(order))


class DihedralGroup(Group):
    """Dihedral group of a specified order."""

    def __init__(self, order: int) -> None:
        super().__init__(comb.named_groups.DihedralGroup(order))


class QuaternionGroup(Group):
    """Quaternion group: 1, i, j, k, -1, -i, -j, -k."""

    def __init__(self) -> None:
        table = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 4, 3, 6, 5, 0, 7, 2],
            [2, 7, 4, 1, 6, 3, 0, 5],
            [3, 2, 5, 4, 7, 6, 1, 0],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 0, 7, 2, 1, 4, 3, 6],
            [6, 3, 0, 5, 2, 7, 4, 1],
            [7, 6, 1, 0, 3, 2, 5, 4],
        ]

        def lift(member: int) -> IntegerArray:
            """Representation from https://en.wikipedia.org/wiki/Quaternion_group."""
            assert 0 <= member < 8
            sign = 1 if member < 4 else -1
            base = member % 4  # +/- 1, i, j, k
            zero = np.zeros((2, 2), dtype=int)
            unit = np.eye(2, dtype=int)
            imag = np.array([[0, -1], [1, 0]], dtype=int)
            if base == 0:  # +/- 1
                blocks = [[unit, zero], [zero, unit]]
            elif base == 1:  # +/- i
                blocks = [[imag, zero], [zero, -imag]]
            elif base == 2:  # +/- j
                blocks = [[zero, -unit], [unit, zero]]
            else:  # if base == 3; +/- k
                blocks = [[zero, -imag], [-imag, zero]]
            return sign * np.block(blocks).astype(int).T % 3

        group = Group.from_table(table, integer_lift=lift)
        super().__init__(group._group, field=3, lift=group._lift)
