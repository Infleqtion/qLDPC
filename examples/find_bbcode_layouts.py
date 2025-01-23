#!/usr/bin/env python3
"""Find BBCode qudit layouts with minimal real-space qudit communication distances.

The qudit placement strategy is described in arXiv:2404.18809.
"""

import functools
import itertools
import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import numpy.typing as npt
import scipy.optimize
from sympy.abc import x, y

import qldpc

Basis2D = tuple[tuple[int, int], tuple[int, int]]
LayoutParams = tuple[bool, Basis2D, Basis2D, tuple[int, int]]


def get_best_known_layout_params(
    code: qldpc.codes.BBCode, folded_layout: bool
) -> tuple[LayoutParams, float]:
    """Retrieve the best known layout parameters for a bivariate bicycle code.

    This function can be used to identify optimized data qubit layout parameters with:
    ```
    layout_params, max_distance = get_best_known_layout_params(code, folded_layout)
    ```
    Here max_distance is the maximum communication distance between neighboring qubits in the Tanner
    graph of the code after additionally optimizing over check qubit locations.  The location of the
    data qubits in the retrieved layout are given by:
    ```
    get_data_qubit_pos = get_data_qubit_pos_func(code, layout_params)
    for qubit in range(len(code)):
        print(qubit, get_data_qubit_pos(qubit))
    ```
    Optimized check qubit locations can similarly be found with
    ```
    check_qubit_locations = get_check_qubit_locations(code, layout_params, min_max_distance)
    for qubit, location in check_qubit_locations.items():
        print(qubit, location)
    ```
    """
    code_params = len(code), code.dimension
    if code_params == (72, 12) and folded_layout:
        vecs_l = ((1, 1), (1, 2))
        vecs_r = ((-1, -1), (-1, -2))
        shift_r = (2, 1)
    elif code_params == (90, 8) and folded_layout:
        vecs_l = ((0, 1), (2, 0))
        vecs_r = ((0, 1), (2, 0))
        shift_r = (0, 0)
    elif code_params == (108, 8) and folded_layout:
        vecs_l = ((3, 1), (4, 4))
        vecs_r = ((-3, -1), (-4, -4))
        shift_r = (1, 2)
    elif code_params == (144, 12) and folded_layout:
        vecs_l = ((0, 1), (1, 0))
        vecs_r = ((0, 1), (1, 0))
        shift_r = (1, 2)
    elif code_params == (288, 12) and folded_layout:
        vecs_l = ((0, 5), (1, 0))
        vecs_r = ((0, 5), (1, 0))
        shift_r = (1, 9)
    else:
        raise ValueError(
            f"Layout parameters unknown for BBCode with parameters {code_params}"
            f" and folded_layout={folded_layout}"
        )
    layout_params = (folded_layout, vecs_l, vecs_r, shift_r)
    max_distance = get_min_max_communication_distance(code, layout_params)
    return layout_params, max_distance


def find_layout_params(
    code: qldpc.codes.BBCode, folded_layout: bool, *, verbose: bool = True
) -> tuple[LayoutParams, float]:
    """Opitmize BBCode layout parameters, as described in arXiv:2404.18809.

    This function can be used to identify optimized data qubit layout parameters with:
    ```
    layout_params, max_distance = find_layout_params(code, folded_layout)
    ```
    Here max_distance is the maximum communication distance between neighboring qubits in the Tanner
    graph of the code after additionally optimizing over check qubit locations.  The location of the
    data qubits in the retrieved layout are given by:
    ```
    get_data_qubit_pos = get_data_qubit_pos_func(code, layout_params)
    for qubit in range(len(code)):
        print(qubit, get_data_qubit_pos(qubit))
    ```
    Optimized check qubit locations can similarly be found with
    ```
    check_qubit_locations = get_check_qubit_locations(code, layout_params, min_max_distance)
    for qubit, location in check_qubit_locations.items():
        print(qubit, location)
    ```
    """
    # initialize the optimized (min-max) communication distance to an upper bound
    optimal_distance = get_max_distance(code)

    # precompute the support of parity checks
    check_supports = get_check_supports(code)

    # construct the set of pairs of lattice vector that span a torus with shape code.orders
    sites = np.ndindex(code.orders)
    pairs: Iterable[Basis2D] = itertools.combinations(sites, 2)  # type:ignore[assignment]
    lattice_vectors = [
        (vec_a, vec_b) for vec_a, vec_b in pairs if code.is_valid_basis(vec_a, vec_b)
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
            shift_r: tuple[int, int]
            for shift_r in np.ndindex(code.orders):  # type:ignore[assignment]
                min_max_distance = get_min_max_communication_distance(
                    code,
                    (folded_layout, vecs_l, vecs_r, shift_r),
                    distance_cutoff=optimal_distance,
                    check_supports=check_supports,
                    validate=False,
                )
                if min_max_distance < optimal_distance:
                    optimal_vecs_l = vecs_l
                    optimal_vecs_r = vecs_r
                    optimal_shift_r = shift_r
                    optimal_distance = min_max_distance
                    if verbose:
                        print()
                        print("new best found:", min_max_distance)
                        print("vecs_l:", vecs_l)
                        print("vecs_r:", vecs_r)
                        print("shift_r:", shift_r)

    return (folded_layout, optimal_vecs_l, optimal_vecs_r, optimal_shift_r), optimal_distance


def get_max_distance(code: qldpc.codes.BBCode) -> float:
    """Get the maximum distance between two qubits in the given code."""
    return 2 * math.sqrt(sum(xx**2 for xx in code.orders))


def get_check_supports(code: qldpc.codes.BBCode) -> Sequence[npt.NDArray[np.int_]]:
    """Identify the support of the parity checks of the given code."""
    return [np.where(stabilizer)[0] for stabilizer in itertools.chain(code.matrix_z, code.matrix_x)]


def get_min_max_communication_distance(
    code: qldpc.codes.BBCode,
    layout_params: LayoutParams,
    *,
    distance_cutoff: float | None = None,
    check_supports: Sequence[npt.NDArray[np.int_]] | None = None,
    digits: int = 1,
    validate: bool = True,
) -> float:
    """Fix check qubit locations, and minimize the maximum communication distance for the code.

    The distance_cutoff argument is used for early stopping: if the minimum is greater than the
    distance_cutoff, then quit early and return a number greater than the distance_cutoff.
    """
    distance_cutoff = distance_cutoff or get_max_distance(code)
    check_supports = check_supports or get_check_supports(code)
    placement_matrix = get_placement_matrix(
        code,
        layout_params,
        check_supports=check_supports,
        validate=validate,
    )

    precision = 10**-digits / 2
    low, high = 0.0, 2 * distance_cutoff + precision
    while True:
        mid = (low + high) / 2
        if has_perfect_matching(placement_matrix <= int(mid**2)):
            high = mid
            if high - low < precision:
                return round(mid, digits)
        else:
            low = mid
            if high - low < precision or low > distance_cutoff:
                return round(high, digits)


def get_placement_matrix(
    code: qldpc.codes.BBCode,
    layout_params: LayoutParams,
    *,
    check_supports: Sequence[npt.NDArray[np.int_]] | None = None,
    validate: bool = True,
) -> npt.NDArray[np.int_]:
    """Construct a placement matrix of squared maximum communication distances.

    Rows and columns of the placement matrix are indexed by check qubits (nodes) and candidate
    locations (locs) for these nodes, such that the value at placement_matrix[node_index][loc_index]
    is the answer to the question: when the given node is placed at the given location, what is that
    node's maximum squared distance to any of its neighbors in the Tanner graph of the code?
    """
    check_supports = check_supports or get_check_supports(code)

    # identify all candidate locations for check qubit placement
    folded_layout, (vec_a, vec_b) = layout_params[:2]
    orders = code.get_order(vec_a), code.get_order(vec_b)
    candidate_locs = get_candidate_check_qubit_locs(folded_layout, orders)

    # compute the (fixed) locations of all check qubits' neighbors
    get_data_qubit_pos = get_data_qubit_pos_func(code, layout_params, validate=validate)
    neighbor_locs = [
        [get_data_qubit_pos(qubit_index) for qubit_index in support] for support in check_supports
    ]

    """
    Vectorized calculation of displacements, with shape = (len(nodes), len(locs), num_neighbors, 2).
    Here dispalacements[node_idx, loc_idx, neighbor_idx, :] is the displacement between a given node
    and a given neighbor when the node is placed at a given location.
    """
    displacements = (
        candidate_locs[None, :, None, :] - np.array(neighbor_locs, dtype=int)[:, None, :, :]
    )

    # squared communication distances
    distances_squared = np.einsum("...i,...i->...", displacements, displacements)

    # matrix of maximum squared communication distances
    return np.max(distances_squared, axis=-1)


@functools.cache
def get_candidate_check_qubit_locs(
    folded_layout: bool, shape: tuple[int, int]
) -> npt.NDArray[np.int_]:
    """Identify all candidate locations for check qubit placement."""
    num_checks = 2 * shape[0] * shape[1]
    nodes = [qldpc.objects.Node(index, is_data=False) for index in range(num_checks)]
    locs = [
        qldpc.codes.BBCode.get_qubit_pos_from_orders(node, folded_layout, shape) for node in nodes
    ]
    return np.array(locs, dtype=int)


def get_data_qubit_pos_func(
    code: qldpc.codes.BBCode, layout_params: LayoutParams, *, validate: bool = True
) -> Callable[[int], tuple[int, int]]:
    """Construct a function that gives qubit positions in particular layout of a BBCode."""
    folded_layout, vecs_l, vecs_r, shift_r = layout_params

    # precompute plaquette mappings
    plaquette_map_l = get_plaquette_map(code, vecs_l, validate=validate)
    plaquette_map_r = get_plaquette_map(code, vecs_r, validate=validate)
    orders_l = (code.get_order(vecs_l[0]), code.get_order(vecs_l[1]))
    orders_r = (code.get_order(vecs_r[0]), code.get_order(vecs_r[1]))
    num_plaquettes = len(code) // 2

    @functools.cache
    def get_data_qubit_pos(qubit_index: int) -> tuple[int, int]:
        """Get the default position of the given qubit/node."""
        plaquette_index = qubit_index % num_plaquettes
        if qubit_index < num_plaquettes:
            sector = "L"
            aa, bb = plaquette_map_l[
                plaquette_index // code.orders[1],
                plaquette_index % code.orders[1],
            ]
            orders = orders_l
        else:
            sector = "R"
            aa, bb = plaquette_map_r[
                (plaquette_index // code.orders[1] + shift_r[0]) % code.orders[0],
                (plaquette_index % code.orders[1] + shift_r[1]) % code.orders[1],
            ]
            orders = orders_r
        return code.get_qubit_pos_from_orders((sector, aa, bb), folded_layout, orders)

    return get_data_qubit_pos


def get_plaquette_map(
    code: qldpc.codes.BBCode, basis: Basis2D, *, validate: bool = True
) -> dict[tuple[int, int], tuple[int, int]]:
    """Construct a map that re-labels plaquettes according to a new basis.

    If the old label of a plaquette was (x, y), the new label is the coefficients (a, b) for which
    (x, y) = a * basis[0] + b * basis[1].  Here (x, y) is taken modulo code.orders, and (a, b) is
    taken modulo the order of the basis vectors on a torus with dimensions code.orders.
    """
    if validate:
        assert code.is_valid_basis(*basis)
    vec_a, vec_b = basis
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


def get_check_qubit_locations(
    code: qldpc.codes.BBCode, layout_params: LayoutParams, max_comm_distance: float
) -> dict[int, tuple[int, int]] | None:
    """If possible, find check qubit locations satisfying a max_comm_distance constraint."""
    folded_layout, (vec_a, vec_b) = layout_params[:2]
    orders = code.get_order(vec_a), code.get_order(vec_b)
    candidate_locs = get_candidate_check_qubit_locs(folded_layout, orders)

    placement_matrix = get_placement_matrix(code, layout_params)
    biadjacency_matrix = placement_matrix <= max_comm_distance**2
    rows, cols = scipy.optimize.linear_sum_assignment(biadjacency_matrix, maximize=True)

    return {
        qubit_index: tuple(candidate_locs[loc_index]) for qubit_index, loc_index in zip(rows, cols)
    }


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
        # layout_params, min_max_distance = find_layout_params(code, folded_layout)
        layout_params, min_max_distance = get_best_known_layout_params(code, folded_layout)
        print("min_max_distance:", min_max_distance)
