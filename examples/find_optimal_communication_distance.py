#!/usr/bin/env python3
"""Find minimal real-space qudit communication distances for various bivariate bicycle codes.

The qudit placement strategy is as described in arXiv:2404.18809.
"""

import functools
import itertools
import math

import numpy as np
import numpy.typing as npt
import scipy.optimize
from sympy.abc import x, y

import qldpc


def get_optimal_layout_params(
    code: qldpc.codes.BBCode, folded_layout: bool, *, verbose: bool = False, cheat: bool = False
) -> tuple[qldpc.codes.BBCode, float]:
    """Get an optimal toric variant of a code, and its maximum communication distance."""
    optimal_distance = 2 * math.sqrt(sum(xx**2 for xx in code.orders))

    if cheat:
        # return parameters that we know are pretty good
        code_params = len(code), code.dimension
        if code_params == (72, 12):
            vecs_l = ((1, 1), (1, 2))
            vecs_r = ((-1, -1), (-1, -2))
            shift_r = (2, 1)
        elif code_params == (90, 8):
            vecs_l = ((0, 1), (2, 0))
            vecs_r = ((0, 1), (2, 0))
            shift_r = (0, 0)
        elif code_params == (108, 8):
            vecs_l = ((3, 1), (4, 4))
            vecs_r = ((-3, -1), (-4, -4))
            shift_r = (1, 2)
        elif code_params == (144, 12):
            vecs_l = ((0, 1), (1, 0))
            vecs_r = ((0, 1), (1, 0))
            shift_r = (1, 2)
        elif code_params == (288, 12):
            vecs_l = ((0, 5), (1, 0))
            vecs_r = ((0, 5), (1, 0))
            shift_r = (1, 9)
        else:
            raise ValueError(f"Optima unknown for code with parameters {code_params}")
        optimal_distance = get_minimal_communication_distance(
            code, folded_layout, vecs_l, vecs_r, shift_r, optimal_distance
        )
        return vecs_l, vecs_r, shift_r, optimal_distance

    # construct the set of pairs of lattice vector that span a torus with shape code.orders
    lattice_vectors = [
        (vec_a, vec_b)
        for vec_a, vec_b in itertools.combinations(np.ndindex(code.orders), 2)
        if code.vectors_span_torus(vec_a, vec_b)
    ]

    # iterate over all lattice vectors used to place qubits in the "left" partition
    for vecs_l in lattice_vectors:
        # iterate over a restricted set of lattice vectors for the "right" partition
        (aa, bb), (cc, dd) = vecs_l
        for vecs_r in [
            ((aa, bb), (cc, dd)),
            ((-aa, -bb), (-cc, -dd)),
        ]:
            # iterate over all relative shifts between the left and right partitions
            for shift_r in np.ndindex(code.orders):
                min_distance = get_minimal_communication_distance(
                    code, folded_layout, vecs_l, vecs_r, shift_r, optimal_distance, validate=False
                )
                if min_distance < optimal_distance:
                    optimal_vecs_l = vecs_l
                    optimal_vecs_r = vecs_r
                    optimal_shift_r = shift_r
                    optimal_distance = min_distance
                    if verbose:
                        print()
                        print("new best found:", min_distance)
                        print("vecs_l:", vecs_l)
                        print("vecs_r:", vecs_r)
                        print("shift_r:", shift_r)

    return optimal_vecs_l, optimal_vecs_r, optimal_shift_r, optimal_distance


def get_minimal_communication_distance(
    code: qldpc.codes.BBCode,
    folded_layout: bool,
    vecs_l: tuple[tuple[int, int], tuple[int, int]],
    vecs_r: tuple[tuple[int, int], tuple[int, int]],
    shift_r: tuple[int, int],
    cutoff: float,
    *,
    digits: int = 1,
    validate: bool = True,
) -> float:
    """Fix check qubit locations, and minimize the maximum communication distance for the code.

    If the minimum is greater than some cutoff, quit early and return a loose upper bound.
    """
    placement_matrix = get_placement_matrix(
        code, folded_layout, vecs_l, vecs_r, shift_r, validate=validate
    )

    precision = 10**-digits / 2
    low, high = 0.0, 2 * cutoff + precision
    while True:
        mid = (low + high) / 2
        if has_perfect_matching(placement_matrix <= int(mid**2)):
            high = mid
            if high - low < precision:
                return round(mid, digits)
        else:
            low = mid
            if high - low < precision or low > cutoff:
                return round(high, digits)


def get_placement_matrix(
    code: qldpc.codes.BBCode,
    folded_layout: bool,
    vecs_l: tuple[tuple[int, int], tuple[int, int]],
    vecs_r: tuple[tuple[int, int], tuple[int, int]],
    shift_r: tuple[int, int],
    *,
    validate: bool = True,
) -> npt.NDArray[np.int_]:
    """Construct a placement matrix of squared maximum communication distances.

    Rows and columns of the placement matrix are indexed by check qubits (nodes) and candidate
    locations (locs) for these nodes, such that the value at placement_matrix[node_index][loc_index]
    is the answer to the question: when the given node is placed at the given location, what is that
    node's maximum squared distance to any of its neighbors in the Tanner graph of the code?
    """
    num_plaquettes = len(code) // 2

    # precompute plaquette mappings
    plaquette_map_l = get_plaquette_map(code, *vecs_l)
    plaquette_map_r = get_plaquette_map(code, *vecs_r)
    orders_l = [code.get_order(vec) for vec in vecs_l]
    orders_r = [code.get_order(vec) for vec in vecs_r]

    @functools.cache
    def get_qubit_pos(qubit_index: int, *, is_data: str) -> tuple[int, int]:
        """Get the default position of the given qubit/node."""
        sector_l = qubit_index < num_plaquettes
        return code.get_qubit_pos(
            qldpc.objects.Node(qubit_index, is_data=is_data),
            folded_layout,
            shift=(0, 0) if sector_l else shift_r,
            plaquette_map=plaquette_map_l if sector_l else plaquette_map_r,
            orders=orders_l if sector_l else orders_r,
        )

    # identify all candidate locations for check qubit placement
    locs = [get_qubit_pos(qubit_index, is_data=False) for qubit_index in range(code.num_checks)]

    # compute the (fixed) locations of all check qubits' neighbors
    neighbor_locs = [
        [get_qubit_pos(qubit_index, is_data=True) for qubit_index, *_ in zip(*np.where(stabilizer))]
        for stabilizer in itertools.chain(code.matrix_z, code.matrix_x)
    ]
    """
    Vectorized calculation of displacements, with shape = (len(nodes), len(locs), num_neighbors, 2).
    Here dispalacements[node_idx, loc_idx, neighbor_idx, :] is the displacement between a given node
    and a given neighbor when the node is placed at a given location.
    """
    displacements = (
        np.array(locs, dtype=int)[None, :, None, :]
        - np.array(neighbor_locs, dtype=int)[:, None, :, :]
    )

    # squared communication distances
    distances_squared = np.sum(displacements**2, axis=-1)

    # matrix of maximum squared communication distances
    return np.max(distances_squared, axis=-1)


def get_plaquette_map(
    code: qldpc.codes.BBCode,
    vec_a: tuple[int, int],
    vec_b: tuple[int, int],
) -> dict[tuple[int, int], tuple[int, int]]:
    order_a = code.get_order(vec_a)
    order_b = code.get_order(vec_b)
    return {
        (
            (aa * vec_a[0] + bb * vec_b[0]) % code.orders[0],
            (aa * vec_a[1] + bb * vec_b[1]) % code.orders[1],
        ): (aa, bb)
        for aa in range(order_a)
        for bb in range(order_b)
    }


def has_perfect_matching(biadjacency_matrix: npt.NDArray[np.bool_]) -> bool | np.bool_:
    """Does a bipartite graph with the given biadjacenty matrix have a perfect matching?"""
    # quit early if any vertex has no indicent edges <--> any row/column is all zeros
    if np.any(~np.any(biadjacency_matrix, axis=0)) or np.any(~np.any(biadjacency_matrix, axis=1)):
        return False
    rows, cols = scipy.optimize.linear_sum_assignment(biadjacency_matrix, maximize=True)
    return np.all(biadjacency_matrix[rows, cols])


if __name__ == "__main__":
    codes = [
        qldpc.codes.BBCode(
            {x: 6, y: 6},
            x**3 + y + y**2,
            y**3 + x + x**2,
        ),
        qldpc.codes.BBCode(
            {x: 15, y: 3},
            x**9 + y + y**2,
            1 + x**2 + x**7,
        ),
        qldpc.codes.BBCode(
            {x: 9, y: 6},
            x**3 + y + y**2,
            y**3 + x + x**2,
        ),
        qldpc.codes.BBCode(
            {x: 12, y: 6},
            x**3 + y + y**2,
            y**3 + x + x**2,
        ),
        qldpc.codes.BBCode(
            {x: 12, y: 12},
            x**3 + y**2 + y**7,
            y**3 + x + x**2,
        ),
    ]
    folded_layout = True

    for code in codes:
        print()
        print("(n, k):", (len(code), code.dimension))
        *_, min_distance = get_optimal_layout_params(code, folded_layout, verbose=True, cheat=True)
        print("min_distance:", min_distance)
