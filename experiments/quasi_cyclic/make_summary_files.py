#!/usr/bin/env python3
"""Script to save quasi-cyclic code search results to data files

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
import os

import numpy as np
from run_search import CACHE_DIR, CACHE_NAME, MAX_COMMUNICATION_DISTANCE, NUM_TRIALS

import qldpc.cache

# Euclidean communication distance "cutoffs" by which to organize results
comm_cutoffs = [5, 8, 10]

# directory in which to save results
save_dir = os.path.join(os.path.dirname(__file__), "codes")

# data file headers
headers = [
    "AUTHOR: Michael A. Perlin, 2024",
    "quasi-cyclic codes of arXiv:2308.07915, with generating polynomials",
    "    A = 1 + x + x**ax * y**ay",
    "    B = 1 + y + x**bx * y**by",
    "here x and y are generators of cyclic groups with orders Rx and Ry",
    "code parameters [[n, k, d]] indicate",
    "    n = number of physical qubits",
    "    k = number of logical qubits",
    "    d = code distance (minimal weight of a nontrivial logical operator)",
    "code distance is estimated by the method of arXiv:2308.07915,"
    + f" minimizing over {NUM_TRIALS} trials",
    "also included:",
    "    D = (Euclidean) communication distance required for a 'folded toric layout' of the code",
    "    r = k d^2 / n",
    "topological 2D codes such as the toric code strictly satisfy r <= 1",
    "we only keep track of codes with r > 1",
    "",
    "Rx, Ry, ax, ay, bx, by, n, k, d, D, r",
]

# data format
fmt = "%d, %d, %d, %d, %d, %d, %d, %d, %d, %.3f, %.3f"

##################################################

comm_cutoffs = sorted([cutoff for cutoff in comm_cutoffs if cutoff <= MAX_COMMUNICATION_DISTANCE])
data_groups: list[list[tuple[int | float, ...]]] = [[] for _ in range(len(comm_cutoffs))]

# iterate over all entries in the cache
cache = qldpc.cache.get_disk_cache(CACHE_NAME, cache_dir=CACHE_DIR)
for key in cache.iterkeys():
    # identify cyclic group orders and polynomial exponents
    dims, exponents, num_trials = key
    if num_trials != NUM_TRIALS:
        continue

    # retrieve code parameters
    nn, kk, dd, comm_dist = cache[key]

    if comm_dist > comm_cutoffs[-1] or dd is None:
        # we don't care about this code
        continue

    # figure of merit relative to the surface code
    merit = kk * dd**2 / nn
    if merit <= 1:
        # this code doesn't even beat the surface code, so we don't care about it
        continue

    # add a summary of this code to the appropriate group of data
    code = (*dims, *exponents, nn, kk, dd, comm_dist, merit)
    for cutoff, data in zip(comm_cutoffs, data_groups):
        if comm_dist <= cutoff:
            data.append(code)
            break

##################################################

os.makedirs(save_dir, exist_ok=True)
header = "\n".join(headers)

# save data groups to files
for last_comm, comm_dist, data in zip([0] + comm_cutoffs, comm_cutoffs, data_groups):
    file = f"codes_D{last_comm}-{comm_dist}.csv"
    path = os.path.join(save_dir, file)
    np.savetxt(path, data, header=header, fmt=fmt)
    last_comm = comm_dist

# save all data
path = os.path.join(save_dir, "codes_all.csv")
data_all = [code for data_group in data_groups for code in data_group]
np.savetxt(path, data_all, header=header, fmt=fmt)
