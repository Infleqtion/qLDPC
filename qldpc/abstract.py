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

All groups in this module are finite, and represented under the hood as a SymPy PermutationGroup, or
a subgroup of the symmetric group.  Group members subclass the SymPy Permutation class.

Groups additionally come equipped with a representation, or "lift", that maps group elements to
square matrices, such that the group action gets lifted to matrix multiplication.

!!! WARNINGS !!!

Whereas matrices are "left-acting" (that is, act on objects from the left) by standard convention,
SymPy permutations are "right-acting", which is to say that the action of two permutations p and q
on an integer i compose as (p*q)(i) = q(p(i)) = i^p^q.  To preserve the order of products before and
after lifting permutations to matrices, which ensures that the lift L(p*q) = L(p) @ L(q), we
therefore make representations likewise right-acting, which is to say that a permutation matrix M
transposes a vector v as v --> v @ M.  In practice, this simply means that matrices are the
transpose of what one might expect.

This module only supports representations of group members by orthogonal matrices over finite
fields.  The restriction to orthogonal representations allows identifying the "transpose" of a group
member p with respect to a representation (lift) L, which is defined by enforcing L(p.T) = L(p).T.
If the representation is orthogonal, then p.T is equal to the inverse ~p = p**-1.
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
import sympy.core

from qldpc import external

DEFAULT_FIELD_ORDER = 2


################################################################################
# groups and group members


UnknownType = TypeVar("UnknownType")


class GroupMember(comb.Permutation):
    """Wrapper for SymPy Permutation class.

    Supports sorting permutations (by their rank), and taking their tensor product.
    """

    @classmethod
    def from_sympy(cls, other: comb.Permutation) -> GroupMember:
        """Convert a SymPy Permutation into a GroupMember."""
        if isinstance(other, GroupMember):
            return other
        new = GroupMember()
        new.__dict__ = other.__dict__
        return new

    def __mul__(self, other: comb.Permutation) -> GroupMember:
        if isinstance(other, comb.Permutation):
            return GroupMember.from_sympy(super().__mul__(other))
        return NotImplemented

    def __add__(self, other: UnknownType) -> UnknownType:
        if hasattr(other, "__radd__"):
            return other.__radd__(self)
        return NotImplemented  # pragma: no cover

    def __lt__(self, other: GroupMember) -> bool:
        return self.rank() < other.rank()

    def __matmul__(self, other: GroupMember) -> GroupMember:
        """Take the "tensor product" of two group members.

        If group members g_1 and g_2 are, respectively, elements of the groups G_1 and G_2, then the
        "tensor product" g_1 @ g_2 is an element of the direct product of G_1 and G_2.
        """
        return GroupMember(self.array_form + [val + self.size for val in other.array_form])


Lift = Callable[[GroupMember], npt.NDArray[np.int_]]
IntegerLift = Callable[[int], npt.NDArray[np.int_]]


def default_lift(member: GroupMember) -> npt.NDArray[np.int_]:
    """Default lift: represent a permutation object by a permutation matrix.

    For consistency with how SymPy composes permutations, this matrix is right-acting, meaning that
    it acts on a vector p from the right: p --> p @ M.
    """
    matrix = np.zeros((member.size,) * 2, dtype=int)
    for ii in range(member.size):
        matrix[ii, member.apply(ii)] = 1
    return matrix


class Group:
    """Base class for a finite group.

    Under the hood, a Group is represented by a SymPy PermutationGroup.
    Group elements are represented by SymPy permutations.

    A group additionally comes equipped with a "lift", or a representation that maps group elements
    to orthogonal matrices over a finite field.  The group action gets lifted to matrix
    multiplication.  If no lift is provided, the group will default to the representation of group
    members by explicit permutation matrices.
    """

    _group: comb.PermutationGroup
    _field: type[galois.FieldArray]
    _lift: Lift
    _name: str | None

    def __init__(
        self,
        *generators: comb.Permutation,
        field: int | None = None,
        lift: Lift | None = None,
        name: str | None = None,
    ) -> None:
        self._init_from_group(comb.PermutationGroup(*generators), field, lift, name)

    def _init_from_group(
        self,
        group: comb.PermutationGroup | Group,
        field: int | None = None,
        lift: Lift | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize from an existing group."""
        self._name = name
        if isinstance(group, comb.PermutationGroup):
            self._group = group
            self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
            self._lift = lift if lift is not None else default_lift
        else:
            assert isinstance(group, Group)
            assert field is None or field == group.field.order
            self._group = group._group
            self._field = group._field
            self._lift = lift if lift is not None else group._lift
            self._name = self._name or group._name  # explicitly provided name overrides group name

    @property
    def name(self) -> str:
        """A name for this group, which is not required to uniquely identify the group."""
        return self._name or f"{type(self).__name__} of order {self.order}"

    def __str__(self) -> str:
        return self.name

    def to_sympy(self) -> comb.PermutationGroup:
        """The underlying SymPy permutation group of this Group."""
        return self._group

    @classmethod
    def from_sympy(
        cls, group: comb.PermutationGroup, field: int | None = None, lift: Lift | None = None
    ) -> Group:
        """Instantiate a Group from a SymPy permutation group."""
        new_group = Group(field=field, lift=lift)
        new_group._group = group
        return new_group

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Group) and self._field == other._field and self._group == other._group
        )

    def __hash__(self) -> int:
        return hash(self._group)

    def __contains__(self, member: GroupMember) -> bool:
        return member in self._group

    def __mul__(self, other: Group) -> Group:
        """Direct product of two groups."""
        group = self._group * other._group

        if self.field == other.field:
            left_lift = self._lift
            right_lift = other._lift
        else:
            left_lift = right_lift = default_lift

        def lift(member: GroupMember) -> galois.FieldArray:
            degree = self._group.degree
            left = GroupMember(member.array_form[:degree])
            right = GroupMember([index - degree for index in member.array_form[degree:]])
            matrix = np.kron(left_lift(left), right_lift(right))
            return self.field(matrix)

        return Group.from_sympy(group, self.field.order, lift)

    def __pow__(self, power: int) -> Group:
        """Direct product of self multiple times."""
        assert power > 0
        return functools.reduce(Group.__mul__, [self] * power)

    @classmethod
    def product(cls, *groups: Group, repeat: int = 1) -> Group:
        """Direct product of Groups."""
        return functools.reduce(Group.__mul__, groups * repeat)

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this group."""
        return self._field

    @property
    def order(self) -> int:
        """Number of members in this group."""
        return self._group.order()

    @property
    def generators(self) -> Sequence[GroupMember]:
        """Generators of this group."""
        return list(map(GroupMember.from_sympy, self._group.generators))

    def generate(self) -> Iterator[GroupMember]:
        """Iterate over all group members."""
        yield from map(GroupMember.from_sympy, self._group.generate())

    @property
    def identity(self) -> GroupMember:
        """The identity element of this group."""
        return GroupMember.from_sympy(self._group.identity)

    def random(self, *, seed: int | None = None) -> GroupMember:
        """A random element this group."""
        if seed is not None:
            sympy.core.random.seed(seed)
        return GroupMember.from_sympy(self._group.random())

    def lift(self, member: GroupMember) -> galois.FieldArray:
        """Lift a group member to its representation by an orthogonal matrix."""
        return self.field(self._lift(member))

    @functools.cached_property
    def lift_dim(self) -> int:
        """Dimension of the repesentation for this group."""
        return self._lift(next(iter(self._group.generators))).shape[0]

    @functools.cached_property
    def table(self) -> npt.NDArray[np.int_]:
        """Multiplication (Cayley) table for this group."""
        members = {member: idx for idx, member in enumerate(self.generate())}
        return np.array(
            [members[aa * bb] for aa in members for bb in members],
            dtype=int,
        ).reshape(self.order, self.order)

    @classmethod
    def from_table(
        cls,
        table: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
        integer_lift: IntegerLift | None = None,
    ) -> Group:
        """Construct a group from a multiplication (Cayley) table."""
        members = {GroupMember(row): idx for idx, row in enumerate(table)}

        if integer_lift is None:
            return Group(*members, lift=default_lift)

        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            return integer_lift(members[member])

        return Group(*members, field=field, lift=lift)

    @classmethod
    def from_generating_mats(
        cls,
        *matrices: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | None = None,
    ) -> Group:
        """Constructs a Group from a given set of generating matrices.

        Group members are represented by how they permute elements of the group itself.
        """
        if not matrices:
            return TrivialGroup(field=field)

        # identify the field we are working over
        if isinstance(matrices[0], galois.FieldArray):
            base_field = type(matrices[0])
        else:
            base_field = galois.GF(field or DEFAULT_FIELD_ORDER)

        if field is not None and field != base_field.order:
            raise ValueError(
                f"Field argument {field} is inconsistent with the given generators, which are"
                f" defined over F_{base_field.order}"
            )

        # keep track of group members and a multiplication table
        index_to_member = {idx: base_field(gen) for idx, gen in enumerate(matrices)}
        hash_to_index = {hash(gen.data.tobytes()): idx for idx, gen in index_to_member.items()}
        table_as_dict = {}

        new_members: dict[int, galois.FieldArray]

        def _account_for_product(aa_idx: int, bb_idx: int) -> None:
            """Account for the product of two matrices."""
            cc_mat = index_to_member[aa_idx] @ index_to_member[bb_idx]
            cc_hash = hash(cc_mat.data.tobytes())
            if cc_hash not in hash_to_index:
                hash_to_index[cc_hash] = cc_idx = len(hash_to_index)
                new_members[cc_idx] = cc_mat
            else:
                cc_idx = hash_to_index[cc_hash]
            table_as_dict[aa_idx, bb_idx] = cc_idx

        # generate all members of the group and build the group multiplication table
        members_to_add = index_to_member.copy()
        while members_to_add:
            new_members = {}
            for aa_idx, bb_idx in itertools.product(members_to_add, index_to_member):
                _account_for_product(aa_idx, bb_idx)
                _account_for_product(bb_idx, aa_idx)
            index_to_member |= new_members
            members_to_add = new_members

        # convert the multiplication table into a 2-D array
        table = np.zeros((len(index_to_member), len(index_to_member)), dtype=int)
        for (aa, bb), cc in table_as_dict.items():
            table[aa, bb] = cc

        # dictionary from a permutation to the index of a group member
        permutation_to_index = {tuple(row): idx for idx, row in enumerate(table)}

        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            """Lift a member to its matrix representation."""
            return index_to_member[permutation_to_index[tuple(member.array_form)]]

        # identify generating permutations and build the group itself
        generators = [GroupMember(table[row]) for row in range(len(matrices))]
        return Group(*generators, field=base_field.order, lift=lift)

    def random_symmetric_subset(
        self, size: int, *, exclude_identity: bool = False, seed: int | None = None
    ) -> set[GroupMember]:
        """Construct a random symmetric subset of a given size.

        Note: this is not a uniformaly random subset, only a "sufficiently random" one.

        WARNING: if excluding the identity element, not all groups have symmetric subsets of
        arbitrary size.  If called with a poor choice of group and subset size, this method may
        never terminate.
        """
        if not 0 < size <= self.order:
            raise ValueError(
                "A random symmetric subset of this group must have a size between 1 and"
                f" {self.order} (provided: {size})"
            )
        if seed is not None:
            sympy.core.random.seed(seed)

        singles = set()  # group members equal to their own inverse
        doubles = set()  # pairs of group members and their inverses
        while True:  # sounds dangerous, but bear with me...
            member = self.random()
            if exclude_identity and member == self.identity:
                continue  # pragma: no cover

            # always add group members and their inverses
            if member == ~member:
                singles.add(member)
            else:
                doubles.add(member)
                doubles.add(~member)

            # count how many extra group members we have found
            num_extra = len(singles) + len(doubles) - size

            if not num_extra:
                # if we have the correct number of group members, we are done
                return singles | doubles

            elif num_extra > 0 and len(singles):
                # we have overshot, so throw away members to get down to the right size
                for _ in range(num_extra // 2):
                    member = sorted(doubles)[sympy.core.random.randint(0, len(doubles) - 1)]
                    doubles.remove(member)
                    doubles.remove(~member)
                if num_extra % 2:
                    member = sorted(singles)[sympy.core.random.randint(0, len(singles) - 1)]
                    singles.remove(member)
                return singles | doubles

    @classmethod
    def from_name(cls, name: str) -> Group:
        """Named group in the GAP computer algebra system."""
        standardized_name = name.strip().replace(" ", "")  # remove whitespace
        generators = [GroupMember(gen) for gen in external.groups.get_generators(standardized_name)]
        group = Group(*generators, name=standardized_name)
        return group


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

    def __bool__(self) -> bool:
        return any(self._vec.values())

    def __iter__(self) -> Iterator[tuple[GroupMember, galois.FieldArray]]:
        yield from self._vec.items()

    def __add__(self, other: int | GroupMember | Element) -> Element:
        if isinstance(other, int):
            return self + other * self.one()

        if isinstance(other, GroupMember):
            new_element = self.copy()
            new_element._vec[other] += self.field(1)
            return new_element

        if isinstance(other, Element):
            new_element = self.copy()
            for member, val in other:
                new_element._vec[member] += val
            return new_element

        return NotImplemented  # pragma: no cover

    def __sub__(self, other: Element | GroupMember | int) -> Element:
        return self + (-1) * other

    def __radd__(self, other: GroupMember) -> Element:
        return self + other

    def __mul__(self, other: int | GroupMember | Element) -> Element:
        if isinstance(other, int):
            # multiply coefficients by 'other'
            new_element = self.zero()
            for member, val in self:
                new_element._vec[member] = val * other
            return new_element

        if isinstance(other, GroupMember):
            # multiply group members by 'other'
            new_element = self.zero()
            for member, val in self:
                new_element._vec[member * other] = val
            return new_element

        if isinstance(other, Element):
            # collect and multiply pairs of terms from 'self' and 'other'
            new_element = self.zero()
            for (aa, x_a), (bb, y_b) in itertools.product(self, other):
                new_element._vec[aa * bb] += x_a * y_b
            return new_element

        return NotImplemented  # pragma: no cover

    def __rmul__(self, other: int | GroupMember) -> Element:
        if isinstance(other, int):
            return self * other

        if isinstance(other, GroupMember):
            new_element = self.zero()
            for member, val in self:
                new_element._vec[other * member] = val
            return new_element

        return NotImplemented  # pragma: no cover

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
            (val * self._group.lift(member) for member, val in self if val),
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


class Protograph(npt.NDArray[np.object_]):
    """Array whose entries are members of a group algebra over a finite field."""

    _group: Group

    def __new__(
        cls, data: npt.NDArray[np.object_] | Sequence[Sequence[object]], group: Group | None = None
    ) -> Protograph:
        protograph = np.asarray(data).view(cls)

        # identify the base group for this protograph
        for value in protograph.ravel():
            if not isinstance(value, Element):
                raise ValueError(
                    "Requirement failed: all entries of a protograph must be Element-valued"
                )
            else:
                if not (group is None or group == value.group):
                    raise ValueError("Inconsistent base groups provided for protograph")
                group = value.group

        if group is None:
            raise ValueError("Cannot determine underlying group for a protograh")
        protograph._group = group

        return protograph

    @property
    def group(self) -> Group:
        """Base group of this protograph."""
        return self._group

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this protograph."""
        return self.group.field

    def lift(self) -> galois.FieldArray:
        """Block matrix obtained by lifting each entry of the protograph."""
        vals = [val.lift() for val in self.ravel()]
        tensor = np.transpose(np.reshape(vals, self.shape + vals[0].shape), [0, 2, 1, 3])
        rows = tensor.shape[0] * tensor.shape[1]
        cols = tensor.shape[2] * tensor.shape[3]
        return self.field(tensor.reshape(rows, cols))

    @property
    def T(self) -> Protograph:
        """Transpose of this protograph, which also transposes every array entry."""
        vals = [val.T for val in self.ravel()]
        return Protograph(np.array(vals, dtype=object).reshape(self.shape).T)

    @classmethod
    def build(
        cls,
        group: Group,
        data: npt.NDArray[np.object_ | np.int_] | Sequence[Sequence[object | int]],
    ) -> Protograph:
        """Construct a protograph.

        The constructed protograph is built from:
        - a group, and
        - an array populated by
            (a) elements of the group algebra,
            (b) group members, or
            (c) integers.
        Integers and group members are "elevated" to elements of the group algebra.
        """
        array = np.asarray(data)

        def elevate(value: Element | GroupMember | int) -> Element:
            """Elevate a value to an element of a group algebra."""
            if isinstance(value, Element):
                return value
            if isinstance(value, GroupMember):
                return Element(group, value)
            return int(value) * Element(group, group.identity)

        vals = [elevate(value) for value in array.ravel()]
        return Protograph(np.array(vals).reshape(array.shape), group)


################################################################################
# "simple" named groups


class TrivialGroup(Group):
    """The trivial group with one member: the identity."""

    def __init__(self, field: int | None = None) -> None:
        super().__init__(
            field=field,
            lift=lambda _: np.array(1, ndmin=2, dtype=int),
            name=TrivialGroup.__name__,
        )

    def random(self, *, seed: int | None = None) -> GroupMember:
        """A random (albeit unique) element this group.

        Necessary to circumvent an error thrown by sympy when "unranking" an empty Permutation.
        """
        return self.identity

    @classmethod
    def to_protograph(
        cls, data: npt.NDArray[np.int_] | Sequence[Sequence[int]], field: int | None = None
    ) -> Protograph:
        """Convert a matrix of 0s and 1s into a protograph of the trivial group."""
        array = np.asarray(data)
        group = TrivialGroup(field)
        zero = Element(group)
        unit = Element(group, group.identity)
        terms = [val * unit if val else zero for val in array.ravel()]
        return Protograph(np.array(terms).reshape(array.shape))


class CyclicGroup(Group):
    """Cyclic group of a specified order.

    The cyclic group has one generator, g.  All members of the cyclic group of order R can be
    written as g^p for an integer power p in {0, 1, ..., R-1}.  The member g^p can be represented by
    (that is, lifted to) an R×R "shift matrix", or the identity matrix with all rows shifted down
    (equivalently, all columns shifted right) by p.  That is, the lift L(g^p) acts on a standard
    basis vector <i| as <i| L(g^p) = < i + p mod R |.
    """

    def __init__(self, order: int) -> None:
        field = DEFAULT_FIELD_ORDER
        identity_mat = np.eye(order, dtype=int)

        # build lift manually, which is faster than the default_lift
        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            return galois.GF(field)(np.roll(identity_mat, member.apply(0), axis=1))

        super()._init_from_group(comb.named_groups.CyclicGroup(order), field, lift)


class AbelianGroup(Group):
    """Direct product of cyclic groups of the specified orders.

    By default, an AbelianGroup member of the form ∏_i g_i^{a_i}, where {g_i} are the generators of
    the group, gets lifted to a direct sum ⨁_i L(g_i)^{a_i}.  If an AbelianGroup is initalized with
    product_lift=True, the group members get lifted to a Kronecker product ⨂_i L(g_i)^{a_i}.
    """

    def __init__(self, *orders: int, product_lift: bool = False) -> None:
        field = DEFAULT_FIELD_ORDER
        identity_mats = [np.eye(order, dtype=int) for order in orders]
        vals = [sum(orders[:idx]) for idx in range(len(orders))]

        # identify method to "combine" two cyclic matrices
        _combine: Callable[[npt.NDArray[np.int_], npt.NDArray[np.int_]], npt.NDArray[np.int_]]
        if product_lift:
            _combine = np.kron

        else:

            def _combine(
                mat_a: npt.NDArray[np.int_], mat_b: npt.NDArray[np.int_]
            ) -> npt.NDArray[np.int_]:
                zero_ab = np.zeros((mat_a.shape[0], mat_b.shape[1]), dtype=int)
                zero_ba = np.zeros((mat_b.shape[0], mat_a.shape[1]), dtype=int)
                return np.block([[mat_a, zero_ab], [zero_ba, mat_b]])

        # build lift manually, which is faster than the default_lift
        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            shifts = [member.apply(val) - val for val in vals]
            mats = [
                np.roll(identity_mat, shift, axis=1)
                for identity_mat, shift in zip(identity_mats, shifts)
            ]
            mat = functools.reduce(_combine, mats)
            return galois.GF(field)(mat)

        group = comb.named_groups.AbelianGroup(*orders)
        order_text = ",".join(map(str, orders))
        name = f"AbelianGroup({order_text})"
        super()._init_from_group(group, field, lift, name)


class DihedralGroup(Group):
    """Dihedral group: symmetries of a regular polygon with a given number of sides."""

    def __init__(self, sides: int) -> None:
        super()._init_from_group(comb.named_groups.DihedralGroup(sides))


class AlternatingGroup(Group):
    """Alternating group: even permutations of a set with a given number of elements."""

    def __init__(self, degree: int) -> None:
        super()._init_from_group(comb.named_groups.AlternatingGroup(degree))


class SymmetricGroup(Group):
    """Symmetric group: all permutations of a given number of symbols."""

    def __init__(self, symbols: int) -> None:
        super()._init_from_group(comb.named_groups.SymmetricGroup(symbols))


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

        def integer_lift(member: int) -> npt.NDArray[np.int_]:
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
            return sign * np.block(blocks).T % 3

        group = Group.from_table(table, field=3, integer_lift=integer_lift)
        super()._init_from_group(group, name=QuaternionGroup.__name__)


class SmallGroup(Group):
    """Group indexed by the GAP computer algebra system."""

    index: int

    def __init__(self, order: int, index: int) -> None:
        assert order > 0
        num_groups = SmallGroup.number(order)
        if not 1 <= index <= num_groups:
            raise ValueError(
                f"Index for SmallGroup of order {order} must be between 1 and {num_groups}"
                + f" (provided: {index})"
            )

        name = f"SmallGroup({order},{index})"
        super()._init_from_group(Group.from_name(name))
        self.index = index

    @functools.cached_property
    def structure(self) -> str:
        """A description of the structure of this group."""
        return self.get_structure(self.order, self.index)

    @classmethod
    def number(cls, order: int) -> int:
        """The number of groups of a given order."""
        return external.groups.get_small_group_number(order)

    @classmethod
    def generator(cls, order: int) -> Iterator[SmallGroup]:
        """Iterator over all groups of a given order."""
        for ii in range(SmallGroup.number(order)):
            yield SmallGroup(order, ii + 1)

    @classmethod
    def get_structure(cls, order: int, index: int) -> str:
        """Retrieve a description of the structure of a group."""
        return external.groups.get_small_group_structure(order, index)


################################################################################
# special linear (SL) and projective special linear (PSL) groups


class SpecialLinearGroup(Group):
    """Special linear group (SL): square matrices with determinant 1."""

    _dimension: int

    def __init__(self, dimension: int, field: int | None = None, linear_rep: bool = True) -> None:
        self._name = f"SL({dimension},{field})"
        self._dimension = dimension
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)

        if linear_rep:
            # Construct a linear representation of this group, in which group elements permute
            # elements of the vector space that the generating matrices act on.

            # identify the target space that group members (as matrices) act on
            target_elements = itertools.product(range(self.field.order), repeat=self.dimension)
            next(target_elements)  # skip the all-0 element
            target_space = [self.field(vec).tobytes() for vec in target_elements]
            target_space_size = self.field.order**self.dimension - 1

            # identify how the generators permute elements of the target space
            generators = []
            for member in self.get_generator_mats():
                perm = np.empty(target_space_size, dtype=int)
                for index, vec_bytes in enumerate(target_space):
                    next_vec = member @ self.field(np.frombuffer(vec_bytes, dtype=np.uint8))
                    next_index = target_space.index(next_vec.tobytes())
                    perm[index] = next_index
                generators.append(GroupMember(perm))

            def lift(member: GroupMember) -> npt.NDArray[np.int_]:
                """Lift a group member to a square matrix.

                Each column of the matrix is nominally determined by how the matrix acts on a
                standard basis vector.  We then take the transpose to make the matrix left-acting.
                """
                cols = []
                for entry in range(self.dimension):
                    inp_vec = np.zeros(self.dimension, dtype=np.uint8)
                    inp_vec[entry] = 1
                    inp_idx = target_space.index(inp_vec.tobytes())
                    out_idx = member(inp_idx)
                    out_vec = np.frombuffer(target_space[out_idx], dtype=np.uint8)
                    cols.append(out_vec)
                return np.vstack(cols, dtype=int).T

            super()._init_from_group(comb.PermutationGroup(generators), field, lift)

        else:
            # represent group members by how they permute elements of the group
            group = self.from_generating_mats(*self.get_generator_mats())
            super()._init_from_group(group)

    @property
    def dimension(self) -> int:
        """Dimension of the elements of this group."""
        return self._dimension

    def get_generator_mats(self) -> tuple[galois.FieldArray, ...]:
        """Generator matrices for this group, based on arXiv:2201.09155."""
        gen_w = -self.field(np.diag(np.ones(self.dimension - 1, dtype=int), k=-1))
        gen_w[0, -1] = 1
        gen_x = self.field.Identity(self._dimension)
        if self.field.order <= 3:
            gen_x[0, 1] = 1
        else:
            gen_x[0, 0] = self.field.primitive_element
            gen_x[1, 1] = self.field.primitive_element**-1
            gen_w[0, 0] = -1 * self.field(1)
        return gen_x, gen_w

    @classmethod
    def iter_mats(cls, dimension: int, field: int | None = None) -> Iterator[galois.FieldArray]:
        """Iterate over all elements of SL(dimension, field)."""
        base_field = galois.GF(field or DEFAULT_FIELD_ORDER)
        for vec in itertools.product(base_field.elements, repeat=dimension**2):
            mat = base_field(np.reshape(vec, (dimension, dimension)))
            if np.linalg.det(mat) == 1:
                yield mat


class ProjectiveSpecialLinearGroup(Group):
    """Projective variant of the special linear group (PSL)."""

    # TODO: support linear representation, similarly to SL.
    # This should straightforward to implement (with a similar construction to that in SL) if we
    # find a representation of generating matrices of dimension > 2.

    _dimension: int

    def __init__(self, dimension: int, field: int | None = None) -> None:
        self._name = f"PSL({dimension},{field})"
        self._dimension = dimension
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        group: Group
        if self.field.order == 2:
            group = SpecialLinearGroup(dimension, 2)
        elif dimension == 2:
            group = Group.from_generating_mats(*self.get_generator_mats())
        else:
            raise ValueError(
                "Projective special linear groups with both dimension and field greater than 2 are"
                " not yet supported"
            )
        super()._init_from_group(group)

    @property
    def dimension(self) -> int:
        """Dimension of the elements of this group."""
        return self._dimension

    def get_generator_mats(self) -> tuple[galois.FieldArray, ...]:
        """Expanding generator matrices for this group, based on arXiv:1807.03879."""
        minus_one = -self.field(1)
        return (
            self.field([[1, 1], [0, 1]]),
            self.field([[1, minus_one], [0, 1]]),
            self.field([[1, 0], [1, 1]]),
            self.field([[1, 0], [minus_one, 1]]),
        )

    @classmethod
    def iter_mats(cls, dimension: int, field: int | None = None) -> Iterator[galois.FieldArray]:
        """Iterate over all elements of PSL(dimension, field)."""
        for mat in SpecialLinearGroup.iter_mats(dimension, field):
            vec = mat.ravel()
            # to quotient SL(d,q) by -I, force the first non-zero entry to be <= q/2
            if vec[(vec != 0).argmax()] <= type(mat).order // 2:
                yield mat


SL = SpecialLinearGroup
PSL = ProjectiveSpecialLinearGroup
