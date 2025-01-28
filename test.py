import numpy as np

import qldpc
import test

code = qldpc.codes.SurfaceCode(6)

# for stabilizers, pauli in [
#     (code.matrix_x, qldpc.objects.Pauli.X),
#     (code.matrix_z, qldpc.objects.Pauli.Z),
# ]:
#     stabilizers = stabilizers.row_reduce()
#     logical_ops = code.get_logical_ops(pauli).reshape(-1, 2, len(code))[:, pauli, :]
#     print()
#     print(stabilizers)
#     print()
#     print(logical_ops)
#     print()
#     print(test.get_sector_distance(stabilizers, logical_ops))

matrix_x = code.matrix_x
matrix_z = code.matrix_z

logs_x = code.get_logical_ops(qldpc.objects.Pauli.X).reshape(-1, 2, len(code))[
    :, qldpc.objects.Pauli.X, :
]
logs_z = code.get_logical_ops(qldpc.objects.Pauli.Z).reshape(-1, 2, len(code))[
    :, qldpc.objects.Pauli.Z, :
]

import time

start = time.time()

for _ in range(1000):
    for stabilizers, logical_ops in [(matrix_x, logs_x), (matrix_z, logs_z)]:
        dist = test.get_sector_distance(stabilizers, logical_ops)
print()
print(dist)
print(time.time() - start)
