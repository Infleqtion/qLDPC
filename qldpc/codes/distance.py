from __future__ import annotations

import math
from typing import SupportsInt

import numpy as np
import numpy.typing as npt


def _hamming_weight(
    arr: npt.NDArray[np.uint],
    buf: npt.NDArray[np.uint] | None = None,
    out: npt.NDArray[np.uint] | None = None,
) -> npt.NDArray[np.uint]:
    """Somewhat efficient (vectorized) Hamming weight calculation. Assumes 64-bit (u)ints.

    This is the weak point in the python/numpy implementation (unfortunately numpy<2.0.0 doesn't
    expose processors' `popcnt` instruction).
    """
    out = np.right_shift(arr, 1, out=out)
    out &= 0x5555555555555555
    out = np.subtract(arr, out, out=out)

    buf = np.right_shift(out, 2, out=buf)
    buf &= 0x3333333333333333
    out &= 0x3333333333333333
    out += buf

    buf = np.right_shift(out, 4, out=buf)
    out += buf
    out &= 0x0F0F0F0F0F0F0F0F

    out *= 0x0101010101010101
    out >>= 56
    return out


def _symplectic_weight(
    arr: npt.NDArray[np.uint],
    buf: npt.NDArray[np.uint] | None = None,
    out: npt.NDArray[np.uint] | None = None,
) -> npt.NDArray[np.uint]:
    """Somewhat efficient (vectorized) symplectic weight calculation. Assumes 64-bit (u)ints.

    This function is equivalent to `_hamming_weight((arr | (arr >> 1)) & 0x5555555555555555).
    """
    out = np.right_shift(arr, 1, out=out)
    out |= arr
    out &= 0x5555555555555555

    buf = np.right_shift(out, 2, out=buf)
    buf &= 0x3333333333333333
    out &= 0x3333333333333333
    out += buf

    buf = np.right_shift(out, 4, out=buf)
    out += buf
    out &= 0x0F0F0F0F0F0F0F0F

    out *= 0x0101010101010101
    out >>= 56
    return out


if getattr(np, "bitwise_count", None):
    np_bitwise_count = getattr(np, "bitwise_count")

    def hamming_weight(
        arr: npt.NDArray[np.uint],
        buf: npt.NDArray[np.uint] | None = None,
        out: npt.NDArray[np.uint] | None = None,
    ) -> npt.NDArray[np.uint]:
        return np_bitwise_count(arr, out=out)

    def symplectic_weight(
        arr: npt.NDArray[np.uint],
        buf: npt.NDArray[np.uint] | None = None,
        out: npt.NDArray[np.uint] | None = None,
    ) -> npt.NDArray[np.uint]:
        """Symplectic weight of an integer."""
        buf = np.right_shift(arr, 1, out=buf)
        buf |= arr
        buf &= 0x5555555555555555
        return np_bitwise_count(buf, out=out)

else:
    hamming_weight = _hamming_weight
    symplectic_weight = _symplectic_weight
    np_bitwise_count = None


def count_trailing_zeros(num: SupportsInt) -> int:
    """Returns the position of the least significant 1 in the binary representation of `num`."""
    num = int(num)
    num = num ^ (num - 1)
    num ^= num >> 1
    return int(math.log2(num))


def rows_to_ints(
    array: npt.ArrayLike, dtype: npt.DTypeLike = np.uint, axis: int = -1,
) -> npt.NDArray[np.uint]:
    """Pack rows of a binary array into rows of the given integral type."""
    array = np.asarray(array, dtype=dtype)
    tsize = array.itemsize * 8

    if array.size == 0:
        return array

    def _to_int(bits: npt.NDArray[np.uint]) -> npt.NDArray[np.uint]:
        """Pack `bits` into a single integer (of type `dtype`)."""
        return (bits << np.arange(len(bits) - 1, -1, -1, dtype=dtype)).sum(dtype=dtype)

    def _to_ints(bits: npt.NDArray[np.uint]) -> list[npt.NDArray[np.uint]]:
        """Pack a single row of bits into a row of integers."""
        return [_to_int(bits[i : i + tsize]) for i in range(0, np.shape(bits)[-1], tsize)]

    return np.apply_along_axis(_to_ints, axis, array)


def rows_to_ints_symplectic(
    array: npt.ArrayLike, dtype: npt.DTypeLike = np.uint
) -> npt.NDArray[np.uint]:
    """Pack rows of a binary array into rows of the given integral type."""
    array = np.asarray(array, dtype=dtype)
    num_bits = array.shape[-1]
    assert num_bits % 2 == 0

    # "riffle" Pauli strings, putting the X and Z support bits for each qubit next to each other
    array = array.reshape(-1, 2, num_bits // 2).transpose(0, 2, 1).reshape(-1, num_bits)
    return rows_to_ints(array, dtype=dtype)


def get_distance_classical(generators: npt.ArrayLike, block_size: int = 15) -> int:
    """Distance of a classical linear binary code."""
    num_bits = np.shape(generators)[-1]
    int_generators = rows_to_ints(generators, dtype=np.uint)

    # Number of generators to include in the operational array. Most calculations will then be
    # vectorized over ``2**block_size`` values
    num_blocked_ops = min(block_size + 1 - int_generators.shape[-1], len(int_generators))

    # Precompute all combinations of first `num_blocked_ops` generators
    array = np.zeros((1, int_generators.shape[-1]), dtype=np.uint)
    for op in int_generators[:num_blocked_ops]:
        array = np.vstack([array, array ^ op])

    int_generators = int_generators[num_blocked_ops:]

    out = np.empty_like(array)
    buf = np.empty_like(array) if np_bitwise_count is None else None

    # Initially array[0] is all zeros, so check array[1:]
    weights = hamming_weight(array[1:], buf=None if buf is None else buf[1:], out=out[1:]).sum(-1)
    min_weight = weights.min(initial=num_bits)

    for i in range(1, 2 ** len(int_generators)):
        array ^= int_generators[count_trailing_zeros(i)]
        weights = hamming_weight(array, buf=buf, out=out).sum(-1)
        min_weight = weights.min(initial=min_weight)

    return int(min_weight)


def get_distance_quantum(
    logical_ops: npt.ArrayLike,
    stabilizers: npt.ArrayLike,
    block_size: int = 15,
    homogeneous: bool = False,
) -> int:
    """Distance of a binary quantum code."""
    num_bits = np.shape(logical_ops)[-1]

    if homogeneous:
        int_logical_ops = rows_to_ints(logical_ops, dtype=np.uint)
        int_stabilizers = rows_to_ints(stabilizers, dtype=np.uint)
        weight_func = hamming_weight
    else:
        int_logical_ops = rows_to_ints_symplectic(logical_ops, dtype=np.uint)
        int_stabilizers = rows_to_ints_symplectic(stabilizers, dtype=np.uint)
        weight_func = symplectic_weight

    num_stabilizers = len(int_stabilizers)

    # Number of generators to include in the operational array. Most calculations will then be
    # vectorized over ``2**block_size`` values
    num_blocked_ops = min(
        block_size + 1 - int_logical_ops.shape[-1],
        len(int_logical_ops) + len(int_stabilizers),
    )

    # Precompute all combinations of first `num_blocked_ops` stabilizers
    array = np.zeros((1, int_stabilizers.shape[-1]), dtype=np.uint)
    for op in int_stabilizers[:num_blocked_ops]:
        array = np.vstack([array, array ^ op])

    if num_blocked_ops > num_stabilizers:
        # fill out block with products of some logical ops
        for op in int_logical_ops[: num_blocked_ops - num_stabilizers]:
            array = np.vstack([array, array ^ op])

        int_logical_ops = int_logical_ops[num_blocked_ops - num_stabilizers :]

    int_stabilizers = int_stabilizers[num_blocked_ops:]

    out = np.empty_like(array)
    if homogeneous and np_bitwise_count is not None:
        buf = None
    else:
        buf = np.empty_like(array)

    # Min weight of the part containing logical ops
    weights = weight_func(array[2**num_stabilizers :]).sum(-1)
    min_weight = weights.min(initial=num_bits)

    for li in range(1, 2 ** len(int_logical_ops)):
        array ^= int_logical_ops[count_trailing_zeros(li)]
        weights = weight_func(array, buf=buf, out=out).sum(-1)
        min_weight = weights.min(initial=min_weight)

        for si in range(1, 2 ** len(int_stabilizers)):
            array ^= int_stabilizers[count_trailing_zeros(si)]
            weights = weight_func(array, buf=buf, out=out).sum(-1, out=weights)
            min_weight = weights.min(initial=min_weight)

    return int(min_weight)
