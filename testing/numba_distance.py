import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit
def hamming_weight(num: numba.uint64) -> numba.uint64:
    """Hamming weight of an integer."""
    count: numba.uint64 = 0
    while num:
        count += num & 1
        num >>= 1
    return count


@numba.njit
def count_trailing_zeros(x: numba.uint64) -> numba.uint64:
    """Count the number of trailing zeros in a 64-bit integer."""
    count: numba.uint64 = 0
    while (x & 1) == 0:
        x >>= 1
        count += 1
    return count


@numba.njit
def rows_to_uint64(binary_array: NDArray[np.uint8]) -> NDArray[np.uint64]:
    """Convert the rows of a binary array into a vector of unsigned 64-bit integers."""
    num_rows: numba.uint64 = binary_array.shape[0]
    bin_cols: numba.uint64 = binary_array.shape[1]
    int_cols: numba.uint64 = (bin_cols + 63) // 64
    int_array: NDArray[np.uint64] = np.zeros((num_rows, int_cols), dtype=np.uint64)

    for row in range(num_rows):
        for c_int in range(int_cols):
            value: numba.uint64 = numba.uint64(0)
            for c_bin in range(64 * c_int, min(64 * (c_int + 1), bin_cols)):
                value = (value << 1) | binary_array[row, c_bin]
            int_array[row, c_int] = value
    return int_array


@numba.njit
def get_distance_quantum_64(
    logical_ops: NDArray[np.uint8], stabilizers: NDArray[np.uint8]
) -> numba.uint64:
    """Distance of a binary quantum code."""
    num_bits: numba.uint64 = logical_ops.shape[1]
    num_logical_ops: numba.uint64 = logical_ops.shape[0]
    num_stabilizers: numba.uint64 = stabilizers.shape[0]

    assert num_logical_ops < 64 and num_stabilizers < 64  # the Gray code breaks otherwise
    assert num_bits <= 64

    # Convert each Pauli string into integer form
    int_logical_ops: NDArray[np.uint64] = rows_to_uint64(logical_ops).ravel()
    int_stabilizers: NDArray[np.uint64] = rows_to_uint64(stabilizers).ravel()

    # Initialize a minimum weight and logical operator
    min_weight: numba.uint64 = num_bits
    logical_op: numba.uint64 = numba.uint64(0)

    # Iterate over all products of logical operators and stabilizers
    ll: numba.uint64
    ss: numba.uint64
    for counter_ll in range(1, 1 << num_logical_ops):
        ll = count_trailing_zeros(counter_ll)
        logical_op ^= int_logical_ops[ll]
        min_weight = min(min_weight, hamming_weight(logical_op))
        for counter_ss in range(1, 1 << num_stabilizers):
            ss = count_trailing_zeros(counter_ll)
            logical_op ^= int_stabilizers[ss]
            min_weight = min(min_weight, hamming_weight(logical_op))

    return min_weight
