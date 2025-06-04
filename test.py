import galois
import numpy as np

from qldpc import abstract, codes

np.set_printoptions(linewidth=200)

# pick your favorite group
group = abstract.CyclicGroup(3)
# group = abstract.AbelianGroup(2, 3)
# group = abstract.AbelianGroup(3, 3)
# group = abstract.AbelianGroup(3, 4)
# group = abstract.DihedralGroup(3)

zero = abstract.RingMember.zero(group)
one = abstract.RingMember.one(group)

# matrix = abstract.RingArray.from_field_array(group, group.field.Random((2, 4, group.order)))
# vector = abstract.RingArray.from_field_array(group, group.field.Random((2, group.order)))
# solution = matrix.linalg_solve(vector)
# print()
# print(matrix.lift())
# print()
# print(vector.to_field_vector())
# if solution is not None:
#     print()
#     print(solution.to_field_vector())
#     print()
#     print(np.array_equal(matrix @ solution, vector))
#     print()
#     print(np.array_equal(matrix.lift() @ solution.to_field_vector(), vector.to_field_vector()))
# exit()


def get_ring_members():
    for vec in codes.HammingCode(group.order).matrix.T:
        yield abstract.RingMember.from_vector(group, vec)


def to_lifted(ops: abstract.RingArray) -> galois.FieldArray:
    new_ops = ops.lift().row_reduce()
    return new_ops[np.any(new_ops, axis=1)]


for seed in range(5, 100):
    # print("seed:", seed)

    # construct a random RingArray
    coefficients = group.field.Random((2, 5, group.order), seed=seed)
    matrix = abstract.RingArray.from_field_array(group, coefficients)

    # identify the null space of the RingArray
    null_space = matrix.null_space()
    assert not np.any(matrix @ null_space.T)

    pivots = abstract.RingArray.build(group, np.eye(null_space.shape[1], dtype=int))

    left_inverse_exists = all(one in row for row in null_space) and len(null_space)
    # if left_inverse_exists:
    #     print("REDUCED PIVOTS!")
    #     pivots = abstract.RingArray.build(group, np.zeros(null_space.shape, dtype=int))
    #     for row, vector in enumerate(null_space):
    #         col = next(col for col, entry in enumerate(vector) if entry == one)
    #         pivots[row, col] = one

    logical_ops_x = to_lifted(np.kron(pivots, null_space))
    logical_ops_z = to_lifted(np.kron(null_space, pivots))
    rank = np.linalg.matrix_rank(logical_ops_x @ logical_ops_z.T)

    code = codes.SLPCode(matrix)

    print(seed, rank, code.dimension)
    # if rank < code.dimension or not left_inverse_exists:
    if rank < code.dimension:
        print()
        print(code.dimension % group.lift_dim)
        print()
        print(matrix.lift())
        print()
        print(matrix.row_reduce().lift())
        print()
        print(null_space.lift())
        print()
        print(pivots.lift())
        exit()

print("done")
exit()

print()
print("null vectors:", len(null_space))
print(null_space.lift())
print()
print("group order:", group.order)
print("code.dimension:", code.dimension)
print("code.gauge_dimension:", code.gauge_dimension)
