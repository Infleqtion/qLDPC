#!/usr/bin/env python3
"""Quantum Tanner code search experiment."""
import concurrent.futures
import itertools
import os

import numpy as np
from sympy.abc import x, y

import qldpc
import qldpc.cache


def get_quasi_cyclic_code(
    dim_x: int, dim_y: int, exponents: tuple[int, int, int, int]
) -> qldpc.codes.QCCode:
    """Construct a quasi-cyclic code."""
    dims = (dim_x, dim_y)
    exp_ax, exp_ay, exp_bx, exp_by = exponents
    poly_a = 1 + x + x**exp_ax * y**exp_ay
    poly_b = 1 + y + x**exp_bx * y**exp_by
    return qldpc.codes.QCCode(dims, poly_a, poly_b)


@qldpc.cache.use_disk_cache("qldpc_quasi_cyclic")
def get_communication_distance(
    dim_x: int, dim_y: int, exponents: tuple[int, int, int, int]
) -> float:
    """Get communication distance required for a toric layout of a quasi-cyclic code."""
    code = get_quasi_cyclic_code(dim_x, dim_y, exponents)
    if code.dimension == 0:
        return np.inf

    # identify maximum Euclidean distance between check/data qubits required for each toric layout
    max_distances = []
    toric_mappings = code.get_toric_mappings()
    for plaquette_map, torus_shape in toric_mappings:
        shifts_x, shifts_z = code.get_check_shifts(plaquette_map, torus_shape, open_boundaries=True)
        distances = set(np.sqrt(xx**2 + yy**2) for xx, yy in shifts_x | shifts_z)
        max_distances.append(max(distances))

    # minimize distance requirement over possible toric layouts
    return min(max_distances)


@qldpc.cache.use_disk_cache("qldpc_quasi_cyclic")
def get_code_params(
    dim_x: int, dim_y: int, exponents: tuple[int, int, int, int], num_trials: int
) -> tuple[int, int, int]:
    """Get the code distance of a quasi-cyclic code."""
    code = get_quasi_cyclic_code(dim_x, dim_y, exponents)
    distance = code.get_distance_bound(num_trials=num_trials)
    if not isinstance(distance, int):
        distance = -1
    return code.num_qubits, code.dimension, distance


def compute_distances(
    dim_x: int,
    dim_y: int,
    exponents: tuple[int, int, int, int],
    num_trials: int,
    *,
    communication_distance_cutoff: int | float = np.inf,
    silent: bool = False,
) -> None:
    """Compute communication and code distances."""
    communication_distance = get_communication_distance(dim_x, dim_y, exponents)
    if communication_distance >= communication_distance_cutoff:
        return None

    code_params = get_code_params(dim_x, dim_y, exponents, num_trials)
    if not silent:
        print(dim_x, dim_y, exponents, "f{communication_distance:.1f}", code_params)


if __name__ == "__main__":
    min_order, max_order = 3, 20
    communication_distance_cutoff = 15
    num_trials = 1000

    max_concurrent_jobs = num_cpus // 2 if (num_cpus := os.cpu_count()) else 1
    save_dir = os.path.join(os.path.dirname(__file__), "quasi_cyclic_codes")

    # run multiple jobs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_jobs) as executor:

        for dim_x in range(min_order, max_order + 1):
            for dim_y in range(min_order, dim_x + 1):
                for exponents in itertools.product(
                    range(dim_x), range(dim_y), range(dim_x), range(dim_y)
                ):
                    # submit this job to the job queue
                    executor.submit(
                        compute_distances,
                        dim_x,
                        dim_y,
                        exponents,
                        num_trials,
                        communication_distance_cutoff=communication_distance_cutoff,
                    )
