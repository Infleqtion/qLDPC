#!/usr/bin/env python3
"""Find minimal real-space qudit communication distances for various bivariate bicycle codes.

The qubit placement strategy is as described in arXiv:2404.18809.
"""

import math
from collections.abc import Sequence

import networkx as nx
import numpy as np
import numpy.typing as npt
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
    nodes, locs, placement_matrix = get_placement_data(code, folded_layout)

    precision = 10**-digits
    low, high = 0.0, math.sqrt(sum((2 * xx) ** 2 for xx in code.orders))
    while high - low > precision:
        mid = (low + high) / 2
        if get_perfect_matching(nodes, locs, placement_matrix <= mid**2):
            high = mid
        else:
            low = mid
    return round(high, digits)


def get_placement_data(
    code: qldpc.codes.BBCode, folded_layout: bool
) -> tuple[list[qldpc.objects.Node], list[tuple[int, int]], npt.NDArray[np.int_]]:
    """Check qubits, their candidate locations, and a placement matrix.

    Rows and columns of the placement matrix are indexed by check qubits (nodes) and candidate
    locations (locs) for these nodes, such that the value at placement_matrix[node_index][loc_index]
    is the maximum squared distance between a node and its neighbors in the Tanner graph of the code
    when the node is placed at a given location.
    """
    graph = code.graph.to_undirected()

    # identify all node that need to be placed, and candidate locations for placement
    nodes = [node for node in graph.nodes() if not node.is_data]
    locs = [code.get_qubit_pos(node, folded_layout) for node in nodes]

    # precompute the locations of all nodes' neighbors (which have fixed locations)
    neighbor_locs = [
        [code.get_qubit_pos(neighbor, folded_layout) for neighbor in graph.neighbors(node)]
        for node in nodes
    ]

    # vectorized displacement calculation; shape = (len(nodes), len(locs), num_neighbors, 2)
    displacements = (
        np.array(locs, dtype=int)[None, :, None, :]
        - np.array(neighbor_locs, dtype=int)[:, None, :, :]
    )

    distances_squared = np.sum(displacements**2, axis=-1)
    placement_matrix = np.max(distances_squared, axis=-1)

    return nodes, locs, placement_matrix


def get_perfect_matching(
    vertices_a: Sequence[object],
    vertices_b: Sequence[object],
    biadjacency_matrix: npt.NDArray[np.bool_],
) -> set[tuple[qldpc.objects.Node, tuple[int, int]]] | None:
    """Find a perfect matching of a bipartite graph.  If no matching exists, return None."""
    # quit early if any vertex has no indicent edges
    if np.any(~np.any(biadjacency_matrix, axis=0)) or np.any(~np.any(biadjacency_matrix, axis=1)):
        return None

    # build the graph
    graph = nx.Graph()
    for idx_a, idx_b in zip(*np.where(biadjacency_matrix)):
        graph.add_edge(vertices_a[idx_a], vertices_b[idx_b])

    # find a perfect matching of the graph
    matching = nx.bipartite.maximum_matching(graph, top_nodes=vertices_a)
    return matching if nx.is_perfect_matching(graph, matching) else None


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
