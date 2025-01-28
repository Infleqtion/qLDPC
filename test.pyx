from typing import Iterator

import numpy as np

cimport numpy as cnp
from libc.stdint cimport uint64_t


cdef extern from "stdint.h":
    int __builtin_popcountl(uint64_t nn)  # hamming weight of a unit64
    int __builtin_ctzl(uint64_t nn)  # count trailing zeroes in a unit64


def _weight(uint64_t nn) -> uint64_t:
    """Hamming weight of an integer."""
    return __builtin_popcountl(nn)


def _gray_code_flips(uint64_t nn) -> Iterator[uint64_t]:
    """Iterate over the bits to flip in a Gray code over a bitstring of the given length."""
    for counter in range(1, 1 << nn):
        yield __builtin_ctzl(counter)


def get_css_sector_distance(
    cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers, cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops
) -> int:
    """Distance one (X or Z) sector of a CSS code."""
    cdef uint64_t num_qubits = logical_ops.shape[1]
    assert num_qubits <= 128
    if num_qubits <= 64:
        return _get_css_sector_distance_64(stabilizers, logical_ops)
    return _get_css_sector_distance_64_2(stabilizers, logical_ops)


def get_classical_distance(cnp.ndarray[cnp.uint8_t, ndim=2] generator) -> int:
    """Distance of a classical code with the given generator matrix."""
    cdef uint64_t num_bits = generator.shape[1]
    null_matrix =  np.zeros((0, num_bits), dtype=np.uint8)
    return get_css_sector_distance(null_matrix, generator)


def _get_css_sector_distance_64(
    cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers, cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops
) -> int:
    """Distance one (X or Z) sector of a CSS code with <= 64 qubits."""
    cdef uint64_t num_qubits = logical_ops.shape[1]
    cdef uint64_t num_stabilizers = stabilizers.shape[0]
    cdef uint64_t num_logical_ops = logical_ops.shape[0]

    # convert each Pauli string into an integer
    cdef cnp.ndarray[cnp.uint64_t] int_stabilizers = _rows_to_uint64(stabilizers)
    cdef cnp.ndarray[cnp.uint64_t] int_logical_ops = _rows_to_uint64(logical_ops)

    # iterate over all products of logical operators and stabilizers
    cdef uint64_t logical_op = 0
    cdef uint64_t min_weight = num_qubits
    cdef uint64_t log_idx, stab_idx
    for log_idx in _gray_code_flips(num_logical_ops):
        logical_op ^= int_logical_ops[log_idx]
        min_weight = min(_weight(logical_op), min_weight)
        for stab_idx in _gray_code_flips(num_stabilizers):
            logical_op ^= int_stabilizers[stab_idx]
            min_weight = min(_weight(logical_op), min_weight)
    return min_weight


def _get_css_sector_distance_64_2(
    cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers, cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops
) -> int:
    """Distance one (X or Z) sector of a CSS code with <= 128 qubits."""
    cdef uint64_t num_qubits = logical_ops.shape[1]
    cdef uint64_t num_stabilizers = stabilizers.shape[0]
    cdef uint64_t num_logical_ops = logical_ops.shape[0]

    # convert each Pauli string into an integer
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] int_stabilizers = _rows_to_uint64_2(stabilizers)
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] int_logical_ops = _rows_to_uint64_2(logical_ops)

    # iterate over all products of logical operators and stabilizers
    cdef uint64_t logical_op_0 = 0
    cdef uint64_t logical_op_1 = 0
    cdef uint64_t min_weight = num_qubits
    cdef uint64_t log_idx, stab_idx
    for log_idx in _gray_code_flips(num_logical_ops):
        logical_op_0 ^= int_logical_ops[log_idx, 0]
        logical_op_1 ^= int_logical_ops[log_idx, 1]
        min_weight = min(_weight(logical_op_0) + _weight(logical_op_1), min_weight)
        for stab_idx in _gray_code_flips(num_stabilizers):
            logical_op_0 ^= int_stabilizers[log_idx, 0]
            logical_op_1 ^= int_stabilizers[log_idx, 1]
            min_weight = min(_weight(logical_op_0) + _weight(logical_op_1), min_weight)
    return min_weight


def _rows_to_uint64(cnp.ndarray[cnp.uint8_t, ndim=2] binary_array):
    """Convert the rows of a binary array into integers."""
    cdef uint64_t num_rows = binary_array.shape[0]
    cdef uint64_t num_cols = binary_array.shape[1]
    cdef cnp.ndarray[cnp.uint64_t, ndim=1] int_array = np.zeros(num_rows, dtype=np.uint64)
    cdef uint64_t row_value
    for rr in range(num_rows):
        row_value = 0
        for cc in range(num_cols):
            row_value = (row_value << 1) | binary_array[rr, cc]
        int_array[rr] = row_value
    return int_array


def _rows_to_uint64_2(
    cnp.ndarray[cnp.uint8_t, ndim=2] binary_array
) -> cnp.ndarray[cnp.uint64_t]:
    """Convert the rows of a binary array into two integers."""
    cdef uint64_t num_rows = binary_array.shape[0]
    cdef uint64_t num_cols = binary_array.shape[1]
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] int_array = np.zeros((num_rows, 2), dtype=np.uint64)
    cdef uint64_t row_value
    for rr in range(num_rows):
        row_value = 0
        for cc in range(min(num_cols, 64)):
            row_value = (row_value << 1) | binary_array[rr, cc]
        int_array[rr, 1] = row_value
        row_value = 0
        for cc in range(64, num_cols):
            row_value = (row_value << 1) | binary_array[rr, cc]
        int_array[rr, 0] = row_value
    return int_array
