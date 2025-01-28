import time

import numpy as np

import qldpc
import test

code = qldpc.codes.ToricCode(6)
code._exact_distance_x = None
code._exact_distance_z = None

stabilizers_x = code.code_x.canonicalized().matrix
logical_ops = code.get_logical_ops()

pauli = qldpc.objects.Pauli.X
logical_ops_x = logical_ops.reshape(2, -1, 2, len(code))[pauli, :, pauli, :]


stabilizers_x = stabilizers_x.view(np.ndarray)
logical_ops_x = logical_ops_x.view(np.ndarray)

print(stabilizers_x.shape)
print(logical_ops_x.shape)
print()
print()
print()

trials = 1000

print()
start = time.time()
for _ in range(trials):
    test.get_css_sector_distance(stabilizers_x, logical_ops_x)
print("-----------------")
print(time.time() - start)
print("-----------------")
print(test.get_css_sector_distance(stabilizers_x, logical_ops_x))


# print()
# start = time.time()
# for _ in range(trials):
#     test._get_css_sector_distance_large(stabilizers_x, logical_ops_x)
# print("-----------------")
# print(time.time() - start)
# print("-----------------")
# print(test._get_css_sector_distance_large(stabilizers_x, logical_ops_x))


# start = time.time()
# for _ in range(1):
#     test.get_css_sector_distance(stabilizers, logical_ops)
# print(time.time() - start)
