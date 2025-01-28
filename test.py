import time

import numpy as np

import qldpc
import test

code = qldpc.codes.SurfaceCode(8)
code._exact_distance_x = None
code._exact_distance_z = None

start = time.time()

print()
for stabilizers, pauli in [
    (code.matrix_x, qldpc.objects.Pauli.X),
    (code.matrix_z, qldpc.objects.Pauli.Z),
]:
    stabilizers = stabilizers.row_reduce()
    logical_ops = code.get_logical_ops(pauli).reshape(-1, 2, len(code))[:, pauli, :]
    dist = test.get_css_sector_distance_64_2(stabilizers, logical_ops)
    print(pauli, dist)

print(time.time() - start)
