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


cdef uint64_t ODD_BITS_MASK = 0x5555555555555555
cdef uint64_t EVEN_BITS_MASK = 0xAAAAAAAAAAAAAAAA


cdef uint64_t hamming_weight(uint64_t num):
    """Hamming weight of an integer."""
    return __builtin_popcountl(num)


cdef uint64_t symplectic_weight(uint64_t num):
    """Symplectic weight of an integer."""
    return hamming_weight((num & ODD_BITS_MASK) | (num & EVEN_BITS_MASK))


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
    """Distance of a classical linear binary code."""
    cdef uint64_t num_bits = generator.shape[1]
    if num_bits <= 64:
        return _get_distance_classical_64(generator)
    return _get_distance_classical_long(generator)


cdef uint64_t _get_distance_classical_64(cnp.ndarray[cnp.uint8_t, ndim=2] generator):
    """Distance of a classical linear binary code."""
    cdef uint64_t num_bits = generator.shape[1]
    cdef uint64_t num_basis_words = generator.shape[0]
    assert num_basis_words < 64  # the Gray code breaks otherwise

    # convert each generating word into integer form
    cdef cnp.uint64_t[:] int_words = rows_to_uint64(generator).ravel()

    # initialize a minimum weight and code word
    cdef uint64_t min_weight = num_bits
    cdef uint64_t word = 0

    # iterate over all code words
    cdef uint64_t ww
    for ww in gray_code_flips(num_basis_words):
        word ^= int_words[ww]
        min_weight = min(min_weight, hamming_weight(word))
    return min_weight


cdef uint64_t _get_distance_classical_long(cnp.ndarray[cnp.uint8_t, ndim=2] generator):
    """Distance of a classical linear binary code."""
    cdef uint64_t num_bits = generator.shape[1]
    cdef uint64_t num_basis_words = generator.shape[0]
    assert num_basis_words < 64  # the Gray code breaks otherwise

    # convert each generating word into integer form
    cdef cnp.uint64_t[:, :] int_words = rows_to_uint64(generator)

    # initialize a minimum weight and code word
    cdef uint64_t min_weight = num_bits
    cdef uint64_t num_int_parts = int_words.shape[1]
    cdef cnp.uint64_t[:] word = np.zeros(num_int_parts, dtype=np.uint64)

    # iterate over all code words and minimize over their Hamming weight
    cdef uint64_t ww, pp, weight
    for ww in gray_code_flips(num_basis_words):
        weight = 0
        for pp in range(num_int_parts):
            word[pp] ^= int_words[ww, pp]
            weight += hamming_weight(word[pp])
        min_weight = min(min_weight, weight)
    return min_weight


####################################################################################################
# quantum code distance


def get_distance_sector_xz(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops,
    cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers,
) -> int:
    """Distance of one (X or Z) sector of a quantum binary CSS code."""
    cdef uint64_t num_qubits = logical_ops.shape[1]
    if num_qubits <= 64:
        return _get_distance_quantum_64(logical_ops, stabilizers, symplectic=False)
    if num_qubits <= 128:
        return _get_distance_quantum_64_2(logical_ops, stabilizers, symplectic=False)
    return _get_distance_quantum_long(logical_ops, stabilizers, symplectic=False)


def get_distance_quantum(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops,
    cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers,
) -> int:
    """Distance of a binary quantum code."""
    cdef uint64_t num_qubits = logical_ops.shape[1] // 2
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] riffled_logical_ops = (
        logical_ops.reshape(-1, 2, num_qubits).transpose(0, 2, 1).reshape(-1, 2 * num_qubits)
    )
    cdef cnp.ndarray[cnp.uint8_t, ndim=2] riffled_stabilizers = (
        stabilizers.reshape(-1, 2, num_qubits).transpose(0, 2, 1).reshape(-1, 2 * num_qubits)
    )
    if num_qubits <= 32:
        return _get_distance_quantum_64(logical_ops, stabilizers, symplectic=True)
    if num_qubits <= 64:
        return _get_distance_quantum_64_2(logical_ops, stabilizers, symplectic=True)
    return _get_distance_quantum_long(logical_ops, stabilizers, symplectic=True)


cdef uint64_t _get_distance_quantum_64(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops,
    cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers,
    bint symplectic,
):
    """Distance of one (X or Z) sector of a quantum binary CSS code."""
    cdef uint64_t num_bits = logical_ops.shape[1]
    cdef uint64_t num_logical_ops = logical_ops.shape[0]
    cdef uint64_t num_stabilizers = stabilizers.shape[0]
    assert num_logical_ops < 64 and num_stabilizers < 64  # the Gray code breaks otherwise
    assert num_bits <= 64

    # decide which weight function to use
    cdef uint64_t (*weight_func)(uint64_t)
    if not symplectic:
        weight_func = hamming_weight
    else:
        weight_func = symplectic_weight

    # convert each Pauli string into integer form
    cdef cnp.uint64_t[:] int_logical_ops = rows_to_uint64(logical_ops).ravel()
    cdef cnp.uint64_t[:] int_stabilizers = rows_to_uint64(stabilizers).ravel()

    # initialize a minimum weight and logical operator
    cdef uint64_t min_weight = num_bits
    cdef uint64_t logical_op = 0

    # iterate over all products of logical operators and stabilizers
    cdef uint64_t ll, ss
    for ll in gray_code_flips(num_logical_ops):
        logical_op ^= int_logical_ops[ll]
        min_weight = min(min_weight, weight_func(logical_op))
        for ss in gray_code_flips(num_stabilizers):
            logical_op ^= int_stabilizers[ss]
            min_weight = min(min_weight, weight_func(logical_op))
    return min_weight


cdef uint64_t _get_distance_quantum_64_2(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops,
    cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers,
    bint symplectic,
):
    """Distance of one (X or Z) sector of a quantum binary CSS code."""
    cdef uint64_t num_bits = logical_ops.shape[1]
    cdef uint64_t num_logical_ops = logical_ops.shape[0]
    cdef uint64_t num_stabilizers = stabilizers.shape[0]
    assert num_logical_ops < 64 and num_stabilizers < 64  # the Gray code breaks otherwise
    assert 64 < num_bits <= 128

    # decide which weight function to use
    cdef uint64_t (*weight_func)(uint64_t)
    if not symplectic:
        weight_func = hamming_weight
    else:
        assert num_bits % 2 == 0
        weight_func = symplectic_weight

    # convert each Pauli string into integer form
    cdef cnp.uint64_t[:, :] int_logical_ops = rows_to_uint64(logical_ops)
    cdef cnp.uint64_t[:, :] int_stabilizers = rows_to_uint64(stabilizers)
    cdef cnp.uint64_t[:] int_logical_ops_0 = int_logical_ops[:, 0]
    cdef cnp.uint64_t[:] int_stabilizers_0 = int_stabilizers[:, 0]
    cdef cnp.uint64_t[:] int_logical_ops_1 = int_logical_ops[:, 1]
    cdef cnp.uint64_t[:] int_stabilizers_1 = int_stabilizers[:, 1]

    # initialize a minimum weight and logical operator
    cdef uint64_t min_weight = num_bits
    cdef uint64_t logical_op_0 = 0
    cdef uint64_t logical_op_1 = 1

    # iterate over all products of logical operators and stabilizers
    cdef uint64_t ll, ss
    for ll in gray_code_flips(num_logical_ops):
        logical_op_0 ^= int_logical_ops_0[ll]
        logical_op_1 ^= int_logical_ops_1[ll]
        min_weight = min(
            min_weight, weight_func(logical_op_0) + weight_func(logical_op_1)
        )
        for ss in gray_code_flips(num_stabilizers):
            logical_op_0 ^= int_stabilizers_0[ll]
            logical_op_1 ^= int_stabilizers_1[ll]
            min_weight = min(
                min_weight, weight_func(logical_op_0) + weight_func(logical_op_1)
            )
    return min_weight


cdef uint64_t _get_distance_quantum_long(
    cnp.ndarray[cnp.uint8_t, ndim=2] logical_ops,
    cnp.ndarray[cnp.uint8_t, ndim=2] stabilizers,
    bint symplectic,
):
    """Distance of one (X or Z) sector of a quantum binary CSS code."""
    cdef uint64_t num_bits = logical_ops.shape[1]
    cdef uint64_t num_logical_ops = logical_ops.shape[0]
    cdef uint64_t num_stabilizers = stabilizers.shape[0]
    assert num_logical_ops < 64 and num_stabilizers < 64  # the Gray code breaks otherwise

    # decide which weight function to use
    cdef uint64_t (*weight_func)(uint64_t)
    if not symplectic:
        weight_func = hamming_weight
    else:
        assert num_bits % 2 == 0
        weight_func = symplectic_weight

    # convert each Pauli string into integer form
    cdef cnp.uint64_t[:, :] int_logical_ops = rows_to_uint64(logical_ops)
    cdef cnp.uint64_t[:, :] int_stabilizers = rows_to_uint64(stabilizers)

    # initialize a minimum weight and logical operator
    cdef uint64_t min_weight = num_bits
    cdef uint64_t num_int_parts = int_logical_ops.shape[1]
    cdef cnp.uint64_t[:] logical_op = np.zeros(num_int_parts, dtype=np.uint64)

    # iterate over all products of logical operators and stabilizers
    cdef uint64_t ll, ss, ll_pp, ss_pp, weight
    for ll in gray_code_flips(num_logical_ops):
        weight = 0
        for ll_pp in range(num_int_parts):
            logical_op[ll_pp] ^= int_logical_ops[ll, ll_pp]
            weight += weight_func(logical_op[ll_pp])
        min_weight = min(min_weight, weight)
        for ss in gray_code_flips(num_stabilizers):
            weight = 0
            for ss_pp in range(num_int_parts):
                logical_op[ss_pp] ^= int_stabilizers[ss, ss_pp]
                weight += weight_func(logical_op[ss_pp])
            min_weight = min(min_weight, weight)
    return min_weight
