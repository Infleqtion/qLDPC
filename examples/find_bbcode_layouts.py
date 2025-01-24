#!/usr/bin/env python3
"""Find BBCode qudit layouts with minimal real-space qudit communication distances.

The qudit placement strategy is described in arXiv:2404.18809.
"""

import functools
import itertools
import math
from collections.abc import Callable, Iterable

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
    graph of the code after additionally optimizing over check qubit locations.  The locations of the
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
    """Optimize BBCode layout parameters, as described in arXiv:2404.18809.

    This function can be used to identify optimized data qubit layout parameters with:
    ```
    layout_params, max_distance = find_layout_params(code, folded_layout)
    ```
    Here max_distance is the maximum communication distance between neighboring qubits in the Tanner
    graph of the code after additionally optimizing over check qubit locations.  The locations of the
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


def get_check_supports(code: qldpc.codes.BBCode) -> npt.NDArray[np.int_]:
    """Identify the support of the parity checks of the given code."""
    return np.array(
        [np.where(stabilizer)[0] for stabilizer in itertools.chain(code.matrix_z, code.matrix_x)],
        dtype=int,
    )


def get_min_max_communication_distance(
    code: qldpc.codes.BBCode,
    layout_params: LayoutParams,
    *,
    distance_cutoff: float | None = None,
    check_supports: npt.NDArray[np.int_] | None = None,
    digits: int = 1,
    validate: bool = True,
) -> float:
    """Fix check qubit locations, and minimize the maximum communication distance for the code.

    The distance_cutoff argument is used for early stopping: if the minimum is greater than the
    distance_cutoff, then quit early and return a number greater than the distance_cutoff.
    """
    distance_cutoff = distance_cutoff or get_max_distance(code)
    check_supports = check_supports if check_supports is not None else get_check_supports(code)
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
    check_supports: npt.NDArray[np.int_] | None = None,
    validate: bool = True,
) -> npt.NDArray[np.int_]:
    """Construct a placement matrix of squared maximum communication distances.

    Rows and columns of the placement matrix are indexed by check qubits (nodes) and candidate
    locations (locs) for these nodes, such that the value at placement_matrix[node_index][loc_index]
    is the answer to the question: when the given node is placed at the given location, what is that
    node's maximum squared distance to any of its neighbors in the Tanner graph of the code?
    """
    # identify how data qubits are permuted relative to the default data qubit layout
    layout_data_locs = get_data_qubit_locs(code, layout_params, validate=validate)
    layout_perm = np.lexsort(layout_data_locs.T)
    inverse_perm = np.empty_like(layout_perm)
    inverse_perm[layout_perm] = np.arange(layout_perm.size, dtype=int)

    # identify matrix of squared distances between all pairs of (check, data) qubit locations
    folded_layout, (vec_a, vec_b) = layout_params[:2]
    orders = code.get_order(vec_a), code.get_order(vec_b)
    squared_distance_matrix = get_squared_distance_matrix(folded_layout, orders)[:, inverse_perm]

    """
    Compute a tensor of squared distances, with shape (num_sites, num_checks, num_neighbors), for
    which squared_distance_tensor[loc_index, check_index, neighbor_index] is the squared distance
    between a check qubit and one of its neighbors when the check qubit in a given location.
    """
    check_supports = check_supports if check_supports is not None else get_check_supports(code)
    squared_distance_tensor = squared_distance_matrix[:, check_supports]

    # matrix of maximum squared communication distances
    return np.max(squared_distance_tensor, axis=-1).T


def get_data_qubit_locs(
    code: qldpc.codes.BBCode, layout_params: LayoutParams, *, validate: bool = True
) -> npt.NDArray[np.int_]:
    """Get the locations of data qubits in particular layout of a BBCode."""
    get_data_qubit_pos = get_data_qubit_pos_func(code, layout_params, validate=validate)
    return np.array([get_data_qubit_pos(index) for index in range(len(code))], dtype=int)


def get_data_qubit_pos_func(
    code: qldpc.codes.BBCode, layout_params: LayoutParams, *, validate: bool = True
) -> Callable[[int], tuple[int, int]]:
    """Construct a function that gives positions of data qubits in particular layout of a BBCode."""
    folded_layout, vecs_l, vecs_r, shift_r = layout_params
    orders = (code.get_order(vecs_l[0]), code.get_order(vecs_l[1]))
    if validate:
        assert orders == (code.get_order(vecs_r[0]), code.get_order(vecs_r[1]))

    # precompute plaquette mappings
    num_plaquettes = len(code) // 2
    plaquette_map_l = get_plaquette_map(code, vecs_l, validate=validate)
    plaquette_map_r = get_plaquette_map(code, vecs_r, validate=validate)

    @functools.cache
    def get_data_qubit_pos(qubit_index: int) -> tuple[int, int]:
        """Get the position of a data qubit in a BBCode."""
        plaquette_index = qubit_index % num_plaquettes
        if qubit_index < num_plaquettes:
            sector = "L"
            aa, bb = plaquette_map_l[
                plaquette_index // code.orders[1],
                plaquette_index % code.orders[1],
            ]
        else:
            sector = "R"
            aa, bb = plaquette_map_r[
                (plaquette_index // code.orders[1] + shift_r[0]) % code.orders[0],
                (plaquette_index % code.orders[1] + shift_r[1]) % code.orders[1],
            ]
        return code.get_qubit_pos_from_orders((sector, aa, bb), folded_layout, orders)

    return get_data_qubit_pos


def get_plaquette_map(
    code: qldpc.codes.BBCode, basis: Basis2D, *, validate: bool = True
) -> dict[tuple[int, int], tuple[int, int]]:
    """Construct a map that re-labels plaquettes by coefficients in a basis that spans the torus.

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


@functools.cache
def get_squared_distance_matrix(
    folded_layout: bool, torus_shape: tuple[int, int]
) -> npt.NDArray[np.int_]:
    """Compute a matrix of squared distances between default check and data qubit locations.

    Here torus_shape is the dimensions of the grid of qubit plaquettes that the qubits live on.
    """
    # identify locations of data and check qubits
    data_locs = get_default_qubit_locs(folded_layout, torus_shape, data=True)
    check_locs = get_default_qubit_locs(folded_layout, torus_shape, data=False)

    # construct a tensor of all pair-wise displacements
    displacements = check_locs[:, None, :] - data_locs[None, :, :]

    # square displacements to get squared distances
    return np.einsum("...i,...i->...", displacements, displacements)


@functools.cache
def get_default_qubit_locs(
    folded_layout: bool, torus_shape: tuple[int, int], *, data: bool
) -> npt.NDArray[np.int_]:
    """Identify the default location of a qubit when placed on a torus with the given dimensions."""
    num_checks = 2 * torus_shape[0] * torus_shape[1]
    nodes = [qldpc.objects.Node(index, is_data=data) for index in range(num_checks)]
    locs = [
        qldpc.codes.BBCode.get_qubit_pos_from_orders(node, folded_layout, torus_shape)
        for node in nodes
    ]
    return with_sorted_rows(np.array(locs, dtype=int))


def with_sorted_rows(array: npt.NDArray[np.int_], axis: int = 0) -> npt.NDArray[np.int_]:
    """Sort the rows of a 2D numpy array."""
    return array[np.lexsort(array.T)]


def has_perfect_matching(biadjacency_matrix: npt.NDArray[np.bool_]) -> bool | np.bool_:
    """Does a bipartite graph with the given biadjacenty matrix have a perfect matching?"""
    # quit early if any vertex has no indicent edges <--> any row/column is all zeros
    if np.any(~np.any(biadjacency_matrix, axis=0)) or np.any(~np.any(biadjacency_matrix, axis=1)):
        return False
    rows, cols = scipy.optimize.linear_sum_assignment(biadjacency_matrix, maximize=True)
    return np.all(biadjacency_matrix[rows, cols])


def get_check_qubit_locations(
    code: qldpc.codes.BBCode, layout_params: LayoutParams, max_comm_distance: float
) -> dict[int, tuple[int, int]]:
    """Find check qubit locations satisfying a max_comm_distance constraint."""
    # identify candidate check qubit locations
    folded_layout, (vec_a, vec_b) = layout_params[:2]
    orders = code.get_order(vec_a), code.get_order(vec_b)
    candidate_locs = get_default_qubit_locs(folded_layout, orders, data=False)

    # assign check qubits to locations
    placement_matrix = get_placement_matrix(code, layout_params)
    biadjacency_matrix = placement_matrix <= max_comm_distance**2
    qubit_indices, loc_indices = scipy.optimize.linear_sum_assignment(
        biadjacency_matrix, maximize=True
    )
    return {
        qubit_index: tuple(candidate_locs[loc_index])
        for qubit_index, loc_index in zip(qubit_indices, loc_indices)
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
