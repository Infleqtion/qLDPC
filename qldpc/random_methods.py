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

# import itertools
# import time
# import galois

import numpy as np
from qldpc.abstract import CyclicGroup, SpecialLinearGroup
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
    code_a = ClassicalCode.random(deg, int(0.35*deg), field)
    code_b = ~code_a
    #code_b = ClassicalCode.random(deg, deg-int(0.4*deg),field)
    return QTCode(subset_a, subset_b, code_a, code_b)

def random_cyclicHammingQTcode(
    blocklength: int,
    field: int = 2,
) -> QTCode :
    deg = 7
    cyclegroup = CyclicGroup(blocklength)
    subset_a = cyclegroup.random_subset(deg-1)
    subset_b = cyclegroup.random_subset(deg-1)
    identity = cyclegroup.identity
    subset_a.add(identity)
    subset_b.add(identity)

    code_a = ClassicalCode.hamming(3,field)
    print(code_a.get_weight())
    code_b = ~code_a
    #code_b = ClassicalCode.random(deg, deg-int(0.4*deg),field)
    return QTCode(subset_a, subset_b, code_a, code_b)

def random_linearQTcode(
    sl_field:int,
    field: int = 2,
    dimension: int = 2,
) -> QTCode :
    deg = 7
    group = SpecialLinearGroup(sl_field, dimension)
    subset_a = group.random_subset(deg-1)
    subset_b = group.random_subset(deg-1)
    identity = group.identity
    subset_a.add(identity)
    subset_b.add(identity)
    code_a = ClassicalCode.hamming(3,field)
    #print(code_a.get_weight())
    code_b = ~code_a
    # print(len(subset_a))
    # print(deg)
    # print(len(subset_b))
    # code_a = ClassicalCode.random(deg, int(0.4*deg), field)
    # code_b = ~code_a
    #code_b = ClassicalCode.random(deg, deg-int(0.4*deg),field)
    return QTCode(subset_a, subset_b, code_a, code_b)


# def random_robustcode(
#     blocklength: int,
#     dimension: int,
#     field: int = 2,
#     robustness: float | None = None,
# ) -> ClassicalCode :
#     return


np.set_printoptions(linewidth=200)

blocklength = 12
field = 2
sl_field = 5
#cyclegroup = CyclicGroup(blocklength)
#subset_a = cyclegroup.random_subset(4)
#tannercode = random_cyclicHammingQTcode(blocklength, field)
tannercode = random_linearQTcode(sl_field, field)
print(tannercode.num_qubits)
print(tannercode.dimension)
print(tannercode.get_distance(upper=10))
# print(np.any(tannercode.matrix))

