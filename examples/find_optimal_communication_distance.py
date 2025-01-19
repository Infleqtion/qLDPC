#!/usr/bin/env python3
"""Find minimal real-space qudit communication distances for various bivariate bicycle codes.

The qubit placement strategy is as described in arXiv:2404.18809.
"""

import math

import numpy as np
import numpy.typing as npt
import scipy.optimize
from sympy.abc import x, y

import qldpc


def get_optimal_code_variant(
    code: qldpc.codes.BBCode, folded_layout: bool
) -> tuple[qldpc.codes.BBCode, float]:
    """Get an optimal toric variant of a code, and its maximum communication distance."""
    optimal_variant = code
    optimal_distance = get_minimal_communication_distance(code, folded_layout)

    for orders, poly_a, poly_b in code.get_equivalent_toric_layout_code_data():
        variant = qldpc.codes.BBCode(orders, poly_a, poly_b)
        min_distance = get_minimal_communication_distance(variant, folded_layout)
        if min_distance < optimal_distance:
            optimal_variant = variant
            optimal_distance = min_distance

    return optimal_variant, optimal_distance


def get_minimal_communication_distance(
    code: qldpc.codes.BBCode, folded_layout: bool, *, digits: int = 1
) -> float:
    """Fix check qubit locations, and find the minimum communication distance for the given code."""
    placement_matrix = get_placement_matrix(code, folded_layout)

    precision = 10**-digits
    low, high = 0.0, math.sqrt(sum((2 * xx) ** 2 for xx in code.orders))
    while high - low > precision:
        mid = (low + high) / 2
        if has_perfect_matching(placement_matrix <= mid**2):
            high = mid
        else:
            low = mid
    return round(high, digits)


def get_placement_matrix(code: qldpc.codes.BBCode, folded_layout: bool) -> npt.NDArray[np.int_]:
    """Construct a placment matrix of squared maximum communication distances.

    Rows and columns of the placement matrix are indexed by check qubits (nodes) and candidate
    locations (locs) for these nodes, such that the value at placement_matrix[node_index][loc_index]
    is the answer to the question: when the given node is placed at the given location, what is that
    node's maximum squared distance to any of its neighbors in the Tanner graph of the code?
    """
    graph = code.graph.to_undirected()

    # identify all node that need to be placed, and candidate locations for placement
    nodes = [node for node in graph.nodes() if not node.is_data]
    locs = [code.get_qubit_pos(node, folded_layout) for node in nodes]

    # compute the locations of all nodes' neighbors (which have fixed locations)
    neighbor_locs = [
        [code.get_qubit_pos(neighbor, folded_layout) for neighbor in graph.neighbors(node)]
        for node in nodes
    ]

    """
    Vectorized calculation of displacements, with shape = (len(nodes), len(locs), num_neighbors, 2).
    Here dispalacements[node_idx, loc_idx, neighbor_idx, :] is the displacement between a given node
    and a given neighbor when placed at a given location.
    """
    displacements = (
        np.array(locs, dtype=int)[None, :, None, :]
        - np.array(neighbor_locs, dtype=int)[:, None, :, :]
    )

    # squared communication distances
    distances_squared = np.sum(displacements**2, axis=-1)

    # matrix of maximum squared communication distances
    return np.max(distances_squared, axis=-1)


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
        variant, min_distance = get_optimal_code_variant(code, folded_layout)
        nn, kk = len(variant), variant.dimension
        orders = {xx: oo for xx, oo in zip(code.symbols, code.orders)}
        print()
        print("(n, k):", (nn, kk))
        print("orders:", orders)
        print("poly_a:", variant.poly_a.as_expr())
        print("poly_b:", variant.poly_b.as_expr())
        print("min_distance:", min_distance)
