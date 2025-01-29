from libc.stdint cimport uint32_t, uint64_t

cdef extern from "stdint.h":
    uint64_t __builtin_popcountl(uint64_t nn)  # hamming weight of a unit64
    uint64_t __builtin_ctzll(uint64_t nn)  # count trailing zeroes in a unit64


# disable deprecated numpy API
cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

from typing import Iterator
import numpy as np
cimport numpy as cnp

cdef uint64_t hamming_weight(uint64_t num):
    """Hamming weight of an integer."""
    return __builtin_popcountl(num)


cdef uint64_t symplectic_weight(uint64_t num):
    """Symplectic weight of an integer."""
    cdef uint32_t first_bits = num >> <uint64_t>32
    cdef uint32_t last_bits = num & <uint64_t>0xFFFFFFFF
    return hamming_weight(first_bits | last_bits)


def gray_code_flips(uint64_t num) -> Iterator[uint64_t]:
    """Iterate over the bits to flip in a Gray code for bitstrings of the given length."""
    for counter in range(<uint64_t>1, <uint64_t>1 << num):
        yield __builtin_ctzll(counter)


cdef cnp.ndarray[cnp.uint64_t, ndim=2] rows_to_uint64(
    cnp.ndarray[cnp.uint8_t, ndim=2] binary_array
):
    """Convert the rows of a binary array into a vector of unsigned 64-bit integers."""
    cdef uint64_t num_rows = binary_array.shape[0]
    cdef uint64_t bin_cols = binary_array.shape[1]
    cdef uint64_t int_cols = (bin_cols + <uint64_t>63) // <uint64_t>64
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] int_array = np.zeros(
        (num_rows, int_cols), dtype=np.uint64
    )
    cdef uint64_t value
    for row in range(num_rows):
        for c_int in range(int_cols):
            value = 0
            for c_bin in range(64 * c_int, min(64 * (c_int + 1), bin_cols)):
                value = (value << 1) | binary_array[row, c_bin]
            int_array[row, c_int] = value
    return int_array


def get_distance_classical_64(
    cnp.ndarray[cnp.uint8_t, ndim=2] generator, *, bint symplectic=False
) -> int:
    """Distance of a classical code with the given generator matrix."""
    cdef uint64_t num_bits = generator.shape[1]
    if num_bits > 64:
        raise ValueError("Fast distance calculation not supported for ClassicalCodes on >64 bits")

    # decide whether to use the hamming or symplectic weight of bitstrings
    cdef uint64_t (*weight)(uint64_t)
    if not symplectic:
        weight = hamming_weight
    else:
        assert num_bits % 2 == 0
        weight = symplectic_weight

    cdef uint64_t num_words = generator.shape[0]
    cdef cnp.ndarray[cnp.uint64_t] int_words = rows_to_uint64(generator)[:, 0]

    # iterate over code words and minimize over their weight
    cdef uint64_t ww
    cdef uint64_t word = 0
    cdef uint64_t min_weight = num_bits
    for ww in gray_code_flips(num_words):
        word ^= int_words[ww]
        min_weight = min(weight(word), min_weight)
    return min_weight


def get_distance_quantum_32(cnp.ndarray[cnp.uint8_t, ndim=2] generator) -> int:
    """Distance of a quantum code with the given symplectic generator matrix."""
    if generator.shape[1] > 64:
        raise ValueError("Fast distance calculation not supported for QuditCodes on >32 qubits")
    return get_distance_classical_64(generator, symplectic=True)


def get_distance_sector_xz_64(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops, cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers
) -> int:
    """Distance of one (X or Z) sector of a quantum CSS code."""
    cdef uint64_t num_qubits = logical_ops.shape[1]
    if num_qubits > 64:
        raise ValueError("Fast distance calculation not supported for CSSCodes on >64 qubits")

    cdef uint64_t num_logical_ops = logical_ops.shape[0]
    cdef uint64_t num_stabilizers = stabilizers.shape[0]

    # convert each Pauli string into an integer
    cdef cnp.ndarray[cnp.uint64_t] int_logical_ops = rows_to_uint64(logical_ops)[:, 0]
    cdef cnp.ndarray[cnp.uint64_t] int_stabilizers = rows_to_uint64(stabilizers)[:, 0]

    # iterate over all products of logical operators and stabilizers
    cdef uint64_t ll, ss
    cdef uint64_t logical_op = 0
    cdef uint64_t minhamming_weight = num_qubits
    for ll in gray_code_flips(num_logical_ops):
        logical_op ^= int_logical_ops[ll]
        minhamming_weight = min(hamming_weight(logical_op), minhamming_weight)
        for ss in gray_code_flips(num_stabilizers):
            logical_op ^= int_stabilizers[ss]
            minhamming_weight = min(hamming_weight(logical_op), minhamming_weight)
    return minhamming_weight
