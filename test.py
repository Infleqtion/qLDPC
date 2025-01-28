import time

import qldpc

# code = qldpc.codes.ToricCode(5, 6, rotated=False)
code = qldpc.codes.ToricCode(6)
code._exact_distance_x = None
code._exact_distance_z = None

pauli = qldpc.objects.Pauli.X
print()

print(len(code))

trials = 1

print()
start = time.time()
for _ in range(trials):
    code.get_distance_exact(pauli)
print("-----------------")
print(time.time() - start)
print("-----------------")
print(code.get_distance_exact(pauli))
