#!/usr/bin/env python3
"""Find minimal real-space qudit communication distances for various bivariate bicycle codes.

The qubit placement strategy is as described in arXiv:2404.18809.
"""

import math

import networkx as nx
import numpy as np
from sympy.abc import x, y

import qldpc


def get_optimal_code_variant(
    code: qldpc.codes.BBCode, folded_layout: bool
) -> tuple[qldpc.codes.BBCode, float]:
    """Get an optimal toric variant of a code, and its minimal maximum communication distance."""
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
    precision = 10**-digits
    low, high = 0.0, get_max_communication_distance(code)
    while high - low > precision:
        mid = (low + high) / 2
        if get_qubit_assignment(code, folded_layout, mid):
            high = mid
        else:
            low = mid
    return round(high, digits)


def get_max_communication_distance(code: qldpc.codes.BBCode) -> float:
    """Get the maximum distance between any pair of qubits in a code."""
    return math.sqrt(sum((2 * xx) ** 2 for xx in code.orders))


def get_qubit_assignment(
    code: qldpc.codes.BBCode, folded_layout: bool, max_comm_dist: float
) -> set[tuple[qldpc.objects.Node, tuple[int, int]]] | None:
    """Find an assignment of data qubits to candidate locations, under a max_comm_dist constraint.

    If no such assignment exists, return None.
    """
    graph = build_placement_graph(code, folded_layout, max_comm_dist)
    if graph is None:
        return None
    if nx.is_connected(graph):
        matching = nx.bipartite.maximum_matching(graph)
    else:
        matching = nx.max_weight_matching(graph, maxcardinality=True)
    return matching if nx.is_perfect_matching(graph, matching) else None


def build_placement_graph(
    code: qldpc.codes.BBCode, folded_layout: bool, max_comm_dist: float
) -> nx.Graph | None:
    """Build a check qubit placement graph.  If some check qubit cannot be placed, return None.

    The check qubit placement graph consists of two vertex sets:
        (a) check qubits, and
        (b) candidate locations.
    The graph draws an edge between qubit qq and location ll if qq's neighbors are at most
    max_comm_dist away from ll.
    """
    max_comm_dist_squared = max_comm_dist**2

    # identify all node that need to be placed, and precompute candidate locations for placement
    nodes = [node for node in code.graph.nodes() if not node.is_data]
    candidate_locs = np.array([code.get_qubit_pos(node, folded_layout) for node in nodes])

    # precompute the locations of all neighbors (which have fixed locations)
    node_neighbors = {
        node: set(code.graph.successors(node)).union(code.graph.predecessors(node))
        for node in nodes
    }
    neighbor_positions = {
        node: np.array(
            [code.get_qubit_pos(neighbor, folded_layout) for neighbor in neighbors],
        )
        for node, neighbors in node_neighbors.items()
    }

    # precompute valid locations for each node based on max_comm_dist
    valid_locations = {}
    for node, neighbor_locs in neighbor_positions.items():
        if len(neighbor_locs) == 0:
            valid_locations[node] = candidate_locs
            continue

        # vectorized distance calculation; shape = (len(node_locs), len(neighbor_locs), 2)
        diff = candidate_locs[:, None, :] - neighbor_locs[None, :, :]
        distances_squared = np.sum(diff**2, axis=-1)
        max_dist_squared = np.max(distances_squared, axis=-1)

        # filter locations satisfying max_comm_dist
        valid_locs = candidate_locs[max_dist_squared <= max_comm_dist_squared]
        if len(valid_locs) == 0:
            return None
        valid_locations[node] = valid_locs

    # build the graph using precomputed valid locations
    graph = nx.Graph()
    for node, locs in valid_locations.items():
        graph.add_edges_from((node, tuple(loc)) for loc in locs)

    return graph


if __name__ == "__main__":
    codes = [
        qldpc.codes.BBCode(
            {x: 6, y: 6},
            x**3 + y + y**2,
            y**3 + x + x**2,
        ),
        # qldpc.codes.BBCode(
        #     {x: 15, y: 3},
        #     x**9 + y + y**2,
        #     1 + x**2 + x**7,
        # ),
        # qldpc.codes.BBCode(
        #     {x: 9, y: 6},
        #     x**3 + y + y**2,
        #     y**3 + x + x**2,
        # ),
        # qldpc.codes.BBCode(
        #     {x: 12, y: 6},
        #     x**3 + y + y**2,
        #     y**3 + x + x**2,
        # ),
        # qldpc.codes.BBCode(
        #     {x: 12, y: 12},
        #     x**3 + y**2 + y**7,
        #     y**3 + x + x**2,
        # ),
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
