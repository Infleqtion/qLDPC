# C imports
from libc.stdint cimport uint64_t

cdef extern from "stdint.h":
    uint64_t __builtin_popcountl(uint64_t nn)  # hamming weight of a unit64
    uint64_t __builtin_ctzll(uint64_t nn)  # count trailing zeroes in a unit64

# disable deprecated numpy API
cdef extern from *:
    """
    #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
    """

# Python imports
from typing import Iterator
import numpy as np
cimport numpy as cnp

####################################################################################################
# common utility functions


cdef uint64_t hamming_weight(uint64_t num):
    """Hamming weight of an integer."""
    return __builtin_popcountl(num)


cdef uint64_t symplectic_weight(uint64_t num, uint64_t half_length):
    """Symplectic weight of an integer."""
    cdef uint64_t first_bits = num >> half_length
    cdef uint64_t last_bits = num & ((<uint64_t>1 << half_length) - <uint64_t>1)
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


####################################################################################################
# classical code distance


def get_distance_classical(cnp.ndarray[cnp.uint8_t, ndim=2] generator) -> int:
    """Distance of a classical linear binary code with the given generator matrix."""
    cdef uint64_t num_bits = generator.shape[1]
    if num_bits <= 64:
        return get_distance_classical_64(generator)
    return get_distance_classical_large(generator)


cdef uint64_t get_distance_classical_64(cnp.ndarray[cnp.uint8_t, ndim=2] generator):
    """Distance of a classical linear binary code with block length <= 64."""
    cdef uint64_t num_bits = generator.shape[1]
    cdef uint64_t num_basis_words = generator.shape[0]
    assert num_bits <= 64
    assert num_basis_words < 64  # the Gray code breaks otherwise

    # convert each generating word into integer form
    cdef cnp.ndarray[cnp.uint64_t] int_words = rows_to_uint64(generator)[:, 0]

    # iterate over all code words and minimize over their Hamming weight
    cdef uint64_t ww
    cdef uint64_t word = 0
    cdef uint64_t min_weight = num_bits
    for ww in gray_code_flips(num_basis_words):
        word ^= int_words[ww]
        min_weight = min(min_weight, hamming_weight(word))
    return min_weight


cdef uint64_t get_distance_classical_large(cnp.ndarray[cnp.uint8_t, ndim=2] generator):
    """Distance of a classical linear binary code with arbitrary block length."""
    cdef uint64_t num_bits = generator.shape[1]
    cdef uint64_t num_basis_words = generator.shape[0]
    assert num_basis_words < 64  # the Gray code breaks otherwise

    # convert each generating word into integer form
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] int_words = rows_to_uint64(generator)
    cdef uint64_t num_int_parts = int_words.shape[1]

    # iterate over all code words and minimize over their Hamming weight
    cdef uint64_t ww, pp, weight
    cdef cnp.ndarray[cnp.uint64_t] word = np.zeros(num_int_parts, dtype=np.uint64)
    cdef uint64_t min_weight = num_bits
    for ww in gray_code_flips(num_basis_words):
        weight = 0
        for pp in range(num_int_parts):
            word[pp] ^= int_words[ww, pp]
            weight += hamming_weight(word[pp])
        min_weight = min(min_weight, weight)
    return min_weight


####################################################################################################
# quantum CSS code distance


def get_distance_sector_xz(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops, cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers
) -> int:
    """Distance of one (X or Z) sector of a quantum binary CSS code."""
    cdef uint64_t num_qubits = logical_ops.shape[1]
    if num_qubits <= 64:
        return get_distance_sector_xz_64(logical_ops, stabilizers)
    return get_distance_sector_xz_large(logical_ops, stabilizers)


cdef uint64_t get_distance_sector_xz_64(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops, cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers
):
    """Distance of one (X or Z) sector of a quantum binary CSS code with block length <= 64."""
    cdef uint64_t num_qubits = logical_ops.shape[1]
    cdef uint64_t num_logical_ops = logical_ops.shape[0]
    cdef uint64_t num_stabilizers = stabilizers.shape[0]
    assert num_qubits <= 64
    assert num_logical_ops < 64 and num_stabilizers < 64  # the Gray code breaks otherwise

    # convert each Pauli string into integer form
    cdef cnp.ndarray[cnp.uint64_t] int_logical_ops = rows_to_uint64(logical_ops)[:, 0]
    cdef cnp.ndarray[cnp.uint64_t] int_stabilizers = rows_to_uint64(stabilizers)[:, 0]

    # iterate over all products of logical operators and stabilizers
    cdef uint64_t ll, ss
    cdef uint64_t logical_op = 0
    cdef uint64_t min_weight = num_qubits
    for ll in gray_code_flips(num_logical_ops):
        logical_op ^= int_logical_ops[ll]
        min_weight = min(min_weight, hamming_weight(logical_op))
        for ss in gray_code_flips(num_stabilizers):
            logical_op ^= int_stabilizers[ss]
            min_weight = min(min_weight, hamming_weight(logical_op))
    return min_weight


cdef uint64_t get_distance_sector_xz_large(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops, cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers
):
    """Distance of one (X or Z) sector of a quantum binary CSS code with arbitrary block length."""
    cdef uint64_t num_qubits = logical_ops.shape[1]
    cdef uint64_t num_logical_ops = logical_ops.shape[0]
    cdef uint64_t num_stabilizers = stabilizers.shape[0]
    assert num_logical_ops < 64 and num_stabilizers < 64  # the Gray code breaks otherwise

    # convert each Pauli string into integer form
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] int_logical_ops = rows_to_uint64(logical_ops)
    cdef cnp.ndarray[cnp.uint64_t, ndim=2] int_stabilizers = rows_to_uint64(stabilizers)
    cdef uint64_t num_int_parts = int_logical_ops.shape[1]

    # iterate over all products of logical operators and stabilizers
    cdef uint64_t ll, ss, pp_ll, pp_ss, weight
    cdef cnp.ndarray[cnp.uint64_t] logical_op = np.zeros(num_int_parts, dtype=np.uint64)
    cdef uint64_t min_weight = num_qubits
    for ll in gray_code_flips(num_logical_ops):
        weight = 0
        for pp_ll in range(num_int_parts):
            logical_op[pp_ll] ^= int_logical_ops[ll, pp_ll]
            weight += hamming_weight(logical_op[pp_ll])
        min_weight = min(min_weight, weight)
        for ss in gray_code_flips(num_stabilizers):
            weight = 0
            for pp_ss in range(num_int_parts):
                logical_op[pp_ss] ^= int_stabilizers[ss, pp_ss]
                weight += hamming_weight(logical_op[pp_ss])
            min_weight = min(min_weight, weight)
    return min_weight


####################################################################################################
# quantum non-CSS code distance


def get_distance_quantum_32(cnp.ndarray[cnp.uint8_t, ndim=2] generator) -> int:
    """Distance of a binary quantum code with block length <= 32."""
    cdef uint64_t num_qubits = generator.shape[1] // 2
    cdef uint64_t num_basis_words = generator.shape[0]
    assert num_qubits <= 32
    assert num_basis_words < 64  # the Gray code breaks otherwise

    # convert each generating word into integer form
    cdef cnp.ndarray[cnp.uint64_t] int_words = rows_to_uint64(generator)[:, 0]

    # iterate over all code words and minimize over their symplectic weight
    cdef uint64_t ww
    cdef uint64_t word = 0
    cdef uint64_t min_weight = num_qubits
    for ww in gray_code_flips(num_basis_words):
        word ^= int_words[ww]
        min_weight = min(min_weight, symplectic_weight(word, num_qubits))
    return min_weight
