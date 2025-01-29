from libc.stdint cimport uint32_t, uint64_t

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


cdef uint64_t _hamming_weight(uint64_t num):
    """Hamming weight of an integer."""
    return __builtin_popcountl(num)


cdef uint64_t _symplectic_weight(uint64_t num):
    """Symplectic weight of an integer."""
    cdef uint32_t first_bits = num >> <uint64_t>32
    cdef uint32_t last_bits = num & 0xFFFFFFFF
    return _hamming_weight(first_bits | last_bits)


def _gray_code_flips(uint64_t num) -> Iterator[uint64_t]:
    """Iterate over the bits to flip in a Gray code for bitstrings of the given length."""
    for counter in range(<uint64_t>1, <uint64_t>1 << num):
        yield __builtin_ctzl(counter)


cdef cnp.ndarray[cnp.uint64_t] _rows_to_uint64(cnp.ndarray[cnp.uint8_t, ndim=2] binary_array):
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


def get_distance_classical_64(
    cnp.ndarray[cnp.uint8_t, ndim=2] generator, *, bint _symplectic=False
) -> int:
    """Distance of a classical code with the given generator matrix."""
    cdef uint64_t num_bits = generator.shape[1]
    if num_bits > 64:
        raise ValueError("Fast distance calculation not supported for ClassicalCodes on >64 bits")

    # decide whether to use the hamming or symplectic weight of bitstrings
    cdef uint64_t (*weight)(uint64_t)
    if not _symplectic:
        weight = _hamming_weight
    else:
        assert num_bits % 2 == 0
        weight = _symplectic_weight

    cdef uint64_t num_words = generator.shape[0]
    cdef cnp.ndarray[cnp.uint64_t] int_words = _rows_to_uint64(generator)

    # iterate over code words and minimize over their weight
    cdef uint64_t ww
    cdef uint64_t word = 0
    cdef uint64_t min_weight = num_bits
    for ww in _gray_code_flips(num_words):
        word ^= int_words[ww]
        min_hamming_weight = min(weight(word), min_weight)
    return min_weight


def get_distance_quantum_32(cnp.ndarray[cnp.uint8_t, ndim=2] generator) -> int:
    """Distance of a quantum code with the given symplectic generator matrix."""
    if generator.shape[1] > 64:
        raise ValueError("Fast distance calculation not supported for QuditCodes on >32 qubits")
    return get_distance_classical_64(generator, _symplectic=True)


def get_distance_subcode_64(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops, cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers
) -> int:
    """Distance of one (X or Z) sector of a quantum CSS code."""
    cdef uint64_t num_qubits = logical_ops.shape[1]
    if num_qubits > 64:
        raise ValueError("Fast distance calculation not supported for CSSCodes on >64 qubits")

    cdef uint64_t num_logical_ops = logical_ops.shape[0]
    cdef uint64_t num_stabilizers = stabilizers.shape[0]

    # convert each Pauli string into an integer
    cdef cnp.ndarray[cnp.uint64_t] int_logical_ops = _rows_to_uint64(logical_ops)
    cdef cnp.ndarray[cnp.uint64_t] int_stabilizers = _rows_to_uint64(stabilizers)

    # iterate over all products of logical operators and stabilizers
    cdef uint64_t ll, ss
    cdef uint64_t logical_op = 0
    cdef uint64_t min_hamming_weight = num_qubits
    for ll in _gray_code_flips(num_logical_ops):
        logical_op ^= int_logical_ops[ll]
        min_hamming_weight = min(_hamming_weight(logical_op), min_hamming_weight)
        for ss in _gray_code_flips(num_stabilizers):
            logical_op ^= int_stabilizers[ss]
            min_hamming_weight = min(_hamming_weight(logical_op), min_hamming_weight)
    return min_hamming_weight
