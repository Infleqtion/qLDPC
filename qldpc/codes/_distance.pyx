from libc.stdint cimport uint64_t

cdef extern from "stdint.h":
    int __builtin_popcountl(uint64_t nn)  # hamming weight of a unit64
    int __builtin_ctzl(uint64_t nn)  # count trailing zeroes in a unit64


# disable deprecated numpy API
cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

from typing import Iterator
import numpy as np
cimport numpy as cnp


def _weight(uint64_t nn) -> uint64_t:
    """Hamming weight of an integer."""
    return __builtin_popcountl(nn)


def _gray_code_flips(uint64_t nn) -> Iterator[uint64_t]:
    """Iterate over the bits to flip in a Gray code over a bitstring of the given length."""
    for counter in range(1, 1 << nn):
        yield __builtin_ctzl(counter)


def _rows_to_uint64(cnp.ndarray[cnp.uint8_t, ndim=2] binary_array) -> cnp.ndarray[cnp.uint64_t]:
    """Convert the rows of a binary array into integers."""
    cdef uint64_t num_rows = binary_array.shape[0]
    cdef uint64_t num_cols = binary_array.shape[1]
    cdef cnp.ndarray[cnp.uint64_t] int_array = np.zeros(num_rows, dtype=np.uint64)
    cdef uint64_t value
    for row in range(num_rows):
        value = 0
        for col in range(num_cols):
            value = (value << 1) | binary_array[row, col]
        int_array[row] = value
    return int_array


def get_subcode_distance_64(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops, cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers
) -> int:
    """Distance of one (X or Z) sector of a CSS code."""
    cdef uint64_t num_qubits = logical_ops.shape[1]
    if num_qubits > 64:
        raise ValueError("Fast distance calculation not supported for > 64 qubits")

    cdef uint64_t num_logical_ops = logical_ops.shape[0]
    cdef uint64_t num_stabilizers = stabilizers.shape[0]

    # convert each Pauli string into an integer
    cdef cnp.ndarray[cnp.uint64_t] int_logical_ops = _rows_to_uint64(logical_ops)
    cdef cnp.ndarray[cnp.uint64_t] int_stabilizers = _rows_to_uint64(stabilizers)

    # iterate over all products of logical operators and stabilizers
    cdef uint64_t logical_op = 0
    cdef uint64_t ll, ss
    cdef uint64_t min_weight = num_qubits
    for ll in _gray_code_flips(num_logical_ops):
        logical_op ^= int_logical_ops[ll]
        min_weight = min(_weight(logical_op), min_weight)
        for ss in _gray_code_flips(num_stabilizers):
            logical_op ^= int_stabilizers[ss]
            min_weight = min(_weight(logical_op), min_weight)
    return min_weight


def get_classical_distance_64(cnp.ndarray[cnp.uint8_t, ndim=2] generator) -> int:
    """Distance of a classical code with the given generator matrix."""
    cdef uint64_t num_bits = generator.shape[1]
    null_matrix =  np.zeros((0, num_bits), dtype=np.uint8)
    return get_subcode_distance_64(generator, null_matrix)
