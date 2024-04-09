#!/usr/bin/env python3
"""Script to perform a brute-force search for quasi-cyclic codes

   Copyright 2023 The qLDPC Authors and Infleqtion Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import concurrent.futures
import itertools
import os
import sys

import diskcache
import numpy as np
from sympy.abc import x, y

import qldpc
import qldpc.cache

NUM_TRIALS = 1000  # for code distance calculations
MAX_COMMUNICATION_DISTANCE = 12
MIN_ORDER = 3  # minimum cyclic group order

CACHE_DIR = os.path.dirname(__file__)
CACHE_NAME = ".code_cache"


def get_quasi_cyclic_code_params(
    dims: tuple[int, int], exponents: tuple[int, int, int, int], num_trials: int
) -> tuple[int, int, int | None, float] | None:
    """Compute communication distance and code distance for a quasi-cyclic code.

    If the code is trivial or the communication distance is beyond the cutoff, return None.
    """
    # construct the code itself
    ax, ay, bx, by = exponents
    poly_a = 1 + x + x**ax * y**ay
    poly_b = 1 + y + x**bx * y**by
    code = qldpc.codes.QCCode(dims, poly_a, poly_b)

    if code.dimension == 0:
        return None

    # identify maximum Euclidean distance between check/data qubits required for each toric layout
    max_distances = []
    for plaquette_map, torus_shape in code.toric_mappings:
        shifts_x, shifts_z = code.get_check_shifts(plaquette_map, torus_shape, open_boundaries=True)
        distances = set(np.sqrt(xx**2 + yy**2) for xx, yy in shifts_x | shifts_z)
        max_distances.append(max(distances))

    # minimize distance requirement over possible toric layouts
    comm_distance = min(max_distances)

    if comm_distance > MAX_COMMUNICATION_DISTANCE:
        return code.num_qubits, code.dimension, None, comm_distance

    distance = code.get_distance_bound(num_trials=num_trials)
    assert isinstance(distance, int)

    return code.num_qubits, code.dimension, distance, comm_distance


def run_and_save(
    dims: tuple[int, int],
    exponents: tuple[int, int, int, int],
    num_trials: int,
    cache: diskcache.Cache,
    *,
    silent: bool = False,
) -> None:
    """Compute and save quasi-cyclic code parameters."""
    if not silent and exponents == (0, 0, 0, 0):
        print(dims)

    params = get_quasi_cyclic_code_params(dims, exponents, num_trials)
    if params is not None:
        nn, kk, dd, comm_dist = params
        cache[dims, exponents, num_trials] = (nn, kk, dd, comm_dist)
        if not silent and dd is not None:
            merit = kk * dd**2 / nn
            print(dims, exponents, (nn, kk, dd), f"{comm_dist:.2f}", f"{merit:.2f}")


def redundant(dims: tuple[int, int], exponents: tuple[int, int, int, int]) -> bool:
    """Are the given code parameters redundant?"""
    dim_x, dim_y = dims
    ax, ay, bx, by = exponents
    return (
        (dim_x, ax, ay) < (dim_y, by, bx)  # enforce torus width >= height
        or (ax, bx) > ((1 - ax) % dim_x, -bx % dim_x)  # reflection about x axis
        or (by, ay) > ((1 - by) % dim_y, -ay % dim_y)  # reflection about y axis
    )


if __name__ == "__main__":
    dim_x = int(sys.argv[1])

    max_concurrent_jobs = num_cpus - 2 if (num_cpus := os.cpu_count()) else 1
    cache = qldpc.cache.get_disk_cache(CACHE_NAME, cache_dir=CACHE_DIR)

    # run multiple jobs in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_jobs) as executor:

        for dim_y in range(MIN_ORDER, dim_x + 1):
            dims = (dim_x, dim_y)
            for exponents in itertools.product(
                range(dim_x), range(dim_y), range(dim_x), range(dim_y)
            ):
                if not redundant(dims, exponents):
                    executor.submit(run_and_save, dims, exponents, NUM_TRIALS, cache)
