"""Methods for generation of random codes, classical and quantum.

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

# import itertools
# import time
# import galois

from collections.abc import Sequence

import numpy as np

from qldpc.abstract import CyclicGroup, Group, GroupMember, SpecialLinearGroup
from qldpc.codes import ClassicalCode, QTCode


def random_cyclicgens(
    order: int | Sequence[int], degree: int
) -> tuple[Group, set[GroupMember], set[GroupMember]]:
    """Generates a pair of random subsets of a cyclic group or a product of cyclic groups.
    Order: Can be an integer k --> Z_k or a tuple,
            (k_1,k_2, ... , k_r) --> Z_{k_1} x ... x Z_{k_r}
    degree: The size of the subsets (both equal to degree)
    """
    cyclegroup: CyclicGroup | Group
    if isinstance(order, int):
        cyclegroup = CyclicGroup(order)
    else:
        cyclegroup = CyclicGroup(order[0])
        for i in order[1:]:
            cyclegroup = cyclegroup @ CyclicGroup(i)
    subset_a = cyclegroup.random_symmetric_subset(degree)
    subset_b = cyclegroup.random_symmetric_subset(degree)
    print(f"Quantum Tanner Code over Cyclic group of order {order} with {degree} generators")
    if isinstance(order, int):
        print("Generators")
        print([p(0) for p in subset_a])
        print([p(0) for p in subset_b])
    return cyclegroup, subset_a, subset_b


def random_lineargens(
    sl_field: int, degree: int, dimension: int = 2
) -> tuple[SpecialLinearGroup, set[GroupMember], set[GroupMember]]:
    """Generates a pair of random subsets of SL(dimension, sl_field) of size degree."""
    lineargroup = SpecialLinearGroup(sl_field, dimension)
    subset_a = lineargroup.random_symmetric_subset(degree)
    subset_b = lineargroup.random_symmetric_subset(degree)
    print(f"Quantum Tanner Code over SL({sl_field}, {dimension}) with {degree} generators ")
    return lineargroup, subset_a, subset_b


def random_basecodes(
    blocklength: int,
    field: int = 2,
    hamming: int | None = None,
    save: int | None = None,
) -> tuple[ClassicalCode, ClassicalCode]:
    """Outputs a pair of codes C_A, C_B such that
    dim(C_A) + dim(C_B) = blocklength
    """
    if hamming is not None:
        assert blocklength == 2**hamming - 1
        print(f"Inner Code is Hamming and its dual of rank {hamming}")
        code_a = ClassicalCode.hamming(hamming, field)
        code_b = ~code_a
    else:
        rate = 0.2
        print("Inner Code is random linear and its dual")
        code_a = ClassicalCode.random(blocklength, int(rate * blocklength), field)
        code_b = ~code_a
    print("Inner code params:")
    print(code_a.get_code_params())
    print(code_b.get_code_params())
    return code_a, code_b


def random_cyclicQTcode(
    order: int | Sequence[int],
    field: int = 2,
    hamming: int | None = None,
    save: bool = False,
) -> QTCode:
    """Constructs a Quantum Tanner Code over Cyclic group of given order
    with random generators.
    """
    if isinstance(order, int):
        size = order
    else:
        size = np.prod(np.array(order))

    if hamming is None:
        deg = int(2 * np.log2(size))
    else:
        deg = 2**hamming - 1
        assert deg <= size

    _, subset_a, subset_b = random_cyclicgens(order, deg)
    code_a, code_b = random_basecodes(deg, field, hamming=hamming, save=save)
    tannercode = QTCode(subset_a, subset_b, code_a, code_b, bipartite=False)
    params = [
        tannercode.num_qubits,
        tannercode.dimension,
        tannercode.get_distance(upper=5, ensure_nontrivial=False),
        tannercode.get_weight(),
    ]
    print("Final code params:", params)
    return tannercode


def random_linearQTcode(
    sl_field: int,
    field: int = 2,
    dimension: int = 2,
    hamming: int | None = None,
    save: bool = False,
) -> QTCode:
    """Constructs a Quantum Tanner Code over SpecialLinear group of given order
    with random generators.
    """
    if hamming:
        deg = 2**hamming - 1
    else:
        deg = 9
    _, subset_a, subset_b = random_lineargens(sl_field, deg, dimension)
    code_a, code_b = random_basecodes(deg, field, hamming=hamming, save=save)
    tannercode = QTCode(subset_a, subset_b, code_a, code_b)
    params = [
        tannercode.num_qubits,
        tannercode.dimension,
        tannercode.get_distance(upper=5, ensure_nontrivial=False),
        tannercode.get_weight(),
    ]
    print("Final code params:", params)
    return tannercode


np.set_printoptions(linewidth=200)

blocklength = 18
field = 2
sl_field = 4
random_cyclicQTcode(blocklength, field, hamming=3)
# random_linearQTcode(sl_field, hamming=3)
# if tannercode.get_distance(upper=10, ensure_nontrivial=False) > 20:
#    np.save
# print(np.any(tannercode.matrix))
""" Experiment with cyclic codes upto like 20?
Fix base codes to be Hamming[7,4] and its dual [7,3]
"""
