#!/usr/bin/env python3
import time

import numpy as np
from numba_distance import get_distance_quantum_64

from qldpc import codes
from qldpc.objects import Pauli

num_trials = 100
code = codes.ToricCode(6)

stabilizers = code.get_code(Pauli.X).canonicalized().matrix.view(np.ndarray).astype(np.uint8)
logical_ops = code.get_logical_ops(Pauli.X).view(np.ndarray).astype(np.uint8)

start = time.time()
for _ in range(num_trials):
    codes.common.get_distance_quantum(logical_ops, stabilizers, homogeneous=True)
print((time.time() - start) / num_trials, "sec/trial")


# run once to jit everything, which shouldn't count against the average function evaluation time
get_distance_quantum_64(logical_ops, stabilizers)

start = time.time()
for _ in range(num_trials):
    get_distance_quantum_64(logical_ops, stabilizers)
print((time.time() - start) / num_trials, "sec/trial")
