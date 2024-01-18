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
import warnings

import cvxpy
import ldpc
import numpy as np
import numpy.typing as npt
import networkx as nx
import galois
import itertools

import qldpc
from qldpc import codes
from qldpc.codes import ClassicalCode, QuditCode, QTCode
from qldpc.abstract import Group, GroupMember, PermutationGroup

"""
def random_cyclicQTcode(
    blocklength: int,
    dimension: int,
    field: int = 2,
) -> ClassicalCode :
    deg = int(20 * np.log2(blocklength))
    code_a = ClassicalCode.random(deg, int(0.2*deg), field)
    code_b = ClassicalCode.random(deg, int(0.2*deg),field)
    
    pass
"""

def construct_psl2(field:int):
    gf = galois.GF(field)
    proj_line = [gf([1,0])]
    for p in gf.elements[1:]: 
        proj_line.append(gf([p,1]))
    vectors = itertools.product(gf.elements, repeat=4)
    psl = []
    sl = []
    for M in vectors:
        a = np.reshape(gf(M), (2,2))
        if np.linalg.det(a) == 1:
            sl.append(a)
            if M[(gf(M)!=0).argmax()] <= field//2: #we force the first non-zero entry to be < p/2 to quotient by -I 
                psl.append(a)
    return (psl,sl)

def construct_linearmat(field:int, dimension:int):
    gf = galois.GF(field)
    vectors = itertools.product(gf.elements, repeat=dimension**2)
    psl = []
    sl = []
    for M in vectors:
        a = np.reshape(gf(M), (dimension,dimension))
        if np.linalg.det(a) == 1:
            sl.append(a)
            if M[(gf(M)!=0).argmax()] <= field//2: #we force the first non-zero entry to be < p/2 to quotient by -I 
                psl.append(a)
    return (psl,sl)
    

def construct_projspace(field:int, dimension:int):
    gf = galois.GF(field)
    proj_space = []
    vectors = itertools.product(gf.elements, repeat=dimension)
    for v in vectors:
        if v[(gf(v)!=0).argmax()] == 1: 
            proj_space.append(v)
    return proj_space

def construct_lineargroup(field:int, dimension:int):
    """
    Construct a hashdict of the projspace
    multiply these elements by the PSL/SL  matrices and generate the permutation matrices by lookup
    Construct sympy group using these permutation matrices
    """
    return NotImplemented