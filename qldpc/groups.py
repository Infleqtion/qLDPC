"""Methods for generation of random codes, classical and quantum

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

import itertools
import time

import galois
import numpy as np
from sympy.combinatorics import Permutation, PermutationGroup
from qldpc.abstract import CyclicGroup, Group
from qldpc.codes import QTCode, ClassicalCode


def random_cyclicQTcode(
    blocklength: int,
    field: int = 2,
) -> QTCode :
    deg = int(2 * int(np.log2(blocklength)))
    cyclegroup = CyclicGroup(blocklength)
    subset_a = cyclegroup.random_subset(deg)
    subset_b = cyclegroup.random_subset(deg)
    # print(len(subset_a))
    # print(deg)
    # print(len(subset_b))
    code_a = ClassicalCode.random(deg, int(0.4*deg), field)
    code_b = ClassicalCode.random(deg, deg-int(0.4*deg),field)
    return QTCode(subset_a, subset_b, code_a, code_b)

# def random_robustcode(
#     blocklength: int,
#     dimension: int,
#     field: int = 2,
#     robustness: float | None = None,
# ) -> ClassicalCode :
#     return


def construct_linear_all(field: int, dimension: int):
    gf = galois.GF(field)
    vectors = itertools.product(gf.elements, repeat=dimension**2)
    psl = []
    sl = []
    for M in vectors:
        a = np.reshape(gf(M), (dimension, dimension))
        if np.linalg.det(a) == 1:
            sl.append(a)
            if (
                M[(gf(M) != 0).argmax()] <= field // 2
            ):  # we force the first non-zero entry to be < p/2 to quotient by -I
                psl.append(a)
    return (psl, sl)


def special_linear_gen(field: int, dimension: int) -> galois.FieldArray:
    """
    Construct generators for SL(field, dimension) based on https://arxiv.org/abs/2201.09155
    """
    gf = galois.GF(field)
    sl = []
    W = gf.Zeros((dimension, dimension))
    W[0, -1] = 1
    for index in range(dimension - 1):
        W[index + 1, index] = -1 * gf(1)
    if field <= 3:
        A = gf.Identity(dimension)
        A[0, 1] = 1
        sl.append(A)
        sl.append(W)
    else:
        W[0, 0] = -1 * gf(1)
        sl.append(W)
        prim = gf.primitive_element
        A = gf.Identity(dimension)
        A[0, 0] = prim
        A[1, 1] = prim**-1
        sl.append(A)
    return sl


def proj_linear_gen(field: int, dimension: int) -> galois.FieldArray:
    """
    Generators for PSL(2) from https://math.stackexchange.com/questions/580607/generating-pair-for-psl2-q
    I doubt if it is correct. Looks the same as that for SL(2,q)!
    """
    gf = galois.GF(field)
    psl = []
    if dimension == 2:
        prim = gf.primitive_element
        minus = -1 * gf(1)
        A = gf([[minus, 1], [minus, 0]])
        W = gf([[prim, 0], [0, prim**-1]])
        psl.append(A)
        psl.append(W)
    else:
        return NotImplemented
    return psl


def psl2_expanding_generators(field: int):
    """
    Expanding generators for PSL2/SL2 from https://arxiv.org/abs/1807.03879
    """
    gf = galois.GF(field)
    A = gf([[1, -1 * gf(1)], [0, 1]])
    B = gf([[1, 1], [0, 1]])
    C = gf([[1, 0], [-1 * gf(1), 1]])
    D = gf([[1, 0], [1, 1]])
    return [A, B, C, D]


def construct_projspace(field: int, dimension: int):
    gf = galois.GF(field)
    proj_space = []
    vectors = itertools.product(gf.elements, repeat=dimension)
    for v in vectors:
        if v[(gf(v) != 0).argmax()] == 1:
            proj_space.append(gf(v).tobytes())
    return proj_space


# def construct_linspace(field: int, dimension: int):
#     gf = galois.GF(field)
#     lin_space = []
#     vectors = itertools.product(gf.elements, repeat=dimension)
#     for v in vectors:
#         if gf(v).any():
#             lin_space.append(gf(v).tobytes())
#     return lin_space


# def group_to_permutation(field: int, space, group) -> PermutationGroup:
#     """
#     Constructs a sympy PermutationGroup using these permutation matrices
#     """
#     gf = galois.GF(field)
#     perm_group = []
#     for M in group:
#         # print(M)
#         perm_string = list(range(len(space)))
#         for index in range(len(space)):
#             current_vector = gf(np.frombuffer(space[index], dtype=np.uint8))
#             next_vector = M @ current_vector
#             next_index = space.index(next_vector.tobytes())
#             perm_string[index] = next_index
#         perm_group.append(Permutation(perm_string))
#         # print(perm_string)
#     return PermutationGroup(perm_group)


# gf = galois.GF(field)
# lin_space = gf(list(itertools.product(gf.elements, repeat=dimension))[1:])
# proj_space = construct_projspace(field, dimension)

# dimension = 2
# field = 37
# lin_space = construct_linspace(field, dimension)
# init = time.time()
# slg = special_linear_gen(field, dimension)
# sl_group2 = group_to_permutation(field, lin_space, slg)
# final = time.time()
# print("Method 2 executed in", final - init)
# print(sl_group2.order())

# init = time.time()
# psl, sl = construct_linear_all(field, dimension)
# sl_group = group_to_permutation(field,proj_space,sl)
# final = time.time()
# print("Method 1 executed in", final - init)
# print(sl_group.order())


# sl_group = group_to_permutation(2,proj_space,sl)
# print(psl_group.random())


np.set_printoptions(linewidth=200)

blocklength = 29
field = 2
#cyclegroup = CyclicGroup(blocklength)
#subset_a = cyclegroup.random_subset(4)
tannercode = random_cyclicQTcode(blocklength, field)
print(tannercode.num_qubits)
print(tannercode.dimension)
print(tannercode.get_distance(upper=10))
# print(np.any(tannercode.matrix))