#!/usr/bin/env python3
"""Find minimal real-space qudit communication distances for various bivariate bicycle codes.

The qubit placement strategy is as described in arXiv:2404.18809.
"""

import functools
import math

import networkx as nx
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
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    return matching if nx.is_perfect_matching(graph, matching) else None


def build_placement_graph(
    code: qldpc.codes.BBCode, folded_layout: bool, max_comm_dist: float
) -> nx.Graph | None:
    """Build a check qubit placement graph.  If some check qubit cannot be placed, return None.

    The check qubit placement graph consists of two vertex sets:
        (a) check qubits, and
        (b) candidate locations.
    The graph draws an edge betweeen qubit qq and location ll if qq's neighbors are at most
    max_comm_dist away from ll.
    """
    nodes = [node for node in code.graph.nodes() if not node.is_data]
    node_locs = [code.get_qubit_pos(node, folded_layout) for node in nodes]

    def satisfies_max_comm_dist(node: qldpc.objects.Node, loc: tuple[int, int]) -> bool:
        """Does placing a node at the given location satisfy the max_comm_dist constraint?"""
        neighbors = set(code.graph.successors(node)).union(code.graph.predecessors(node))
        return not any(
            get_dist(loc, code.get_qubit_pos(neighbor, folded_layout)) > max_comm_dist
            for neighbor in neighbors
        )

    graph = nx.Graph()
    for node in nodes:
        edges = [(node, loc) for loc in node_locs if satisfies_max_comm_dist(node, loc)]
        if not edges:
            return None
        graph.add_edges_from(edges)

    return graph


@functools.cache
def get_dist(loc_a: tuple[int, ...], loc_b: tuple[int, ...]) -> float:
    """Euclidean (L2) distance between two locations."""
    return math.sqrt(sum((aa - bb) ** 2 for aa, bb in zip(loc_a, loc_b)))


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
