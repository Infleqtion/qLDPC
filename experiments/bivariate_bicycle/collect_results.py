#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Export bivariate bicycle code search results to a data file.

This script reads the disk cache written by run_search.py, and saves every code whose figure of
merit r = k d^2 / n exceeds 1 (the bound satisfied by 2D topological codes such as the toric code)
to codes.csv in this directory.  Only results computed with --num-trials distance trials are
exported.

Example:
    python experiments/bivariate_bicycle/collect_results.py
"""

from __future__ import annotations

import argparse
import pathlib

import diskcache
import numpy as np
from run_search import CACHE_DIR, CACHE_NAME, NUM_TRIALS

import qldpc.cache

SAVE_PATH = pathlib.Path(__file__).parent / "codes.csv"  # path of the exported data file

# format of each row in the exported data file
ROW_FORMAT = "%d, %d, %d, %d, %d, %d, %d, %d, %d, %.3f"

####################################################################################################
# command-line interface


def main() -> None:
    """Export bivariate bicycle codes that outperform the toric code to a data file."""
    args = parse_args()
    cache = qldpc.cache.get_disk_cache(CACHE_NAME, cache_dir=CACHE_DIR)
    data = get_code_data(cache, args.num_trials)
    np.savetxt(SAVE_PATH, data, header=get_header(args.num_trials), fmt=ROW_FORMAT)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=NUM_TRIALS,
        help="number of distance trials used by the search results to export"
        " (default: %(default)s)",
    )
    return parser.parse_args()


####################################################################################################
# data collection


def get_code_data(cache: diskcache.Cache, num_trials: int) -> list[tuple[int | float, ...]]:
    """Collect the parameters of cached bivariate bicycle codes with figure of merit r > 1.

    Args:
        cache: The cache written by run_search.py.
        num_trials: Only collect codes whose distance was bounded with this many trials.

    Returns:
        One row (Rx, Ry, ax, ay, bx, by, n, k, d, r) per code, in the format of the data file.
    """
    data: list[tuple[int | float, ...]] = []
    last_print_val = -1

    for key in cache.iterkeys():
        # identify cyclic group orders and polynomial exponents
        dims, exponents, key_num_trials = key
        if key_num_trials != num_trials:
            continue

        # report progress whenever the first polynomial exponent changes
        if (print_val := exponents[0]) != last_print_val:
            last_print_val = print_val
            print(dims, exponents)

        # retrieve code parameters, skipping codes whose distance was not computed
        nn, kk, dd = cache[key]
        if dd is None:
            continue

        # only report codes that outperform the surface code
        merit = kk * dd**2 / nn
        if merit > 1:
            data.append((*dims, *exponents, nn, kk, dd, merit))

    return data


def get_header(num_trials: int) -> str:
    """Get the header of the exported data file.

    Args:
        num_trials: The number of trials used to bound code distances.

    Returns:
        A multi-line header that describes the data file.
    """
    return "\n".join(
        [
            "AUTHOR: Michael A. Perlin, 2024",
            "bivariate bicycle codes of arXiv:2308.07915, with generating polynomials",
            "    A = 1 + x + x**ax * y**ay",
            "    B = 1 + y + x**bx * y**by",
            "here x and y are generators of cyclic groups with orders Rx and Ry",
            "code parameters [[n, k, d]] indicate",
            "    n = number of physical qubits",
            "    k = number of logical qubits",
            "    d = code distance (minimal weight of a nontrivial logical operator)",
            "code distance is estimated by the method of arXiv:2308.07915,"
            + f" minimizing over {num_trials} trials",
            "the last column reports r = k d^2 / n",
            "topological 2D codes such as the toric code strictly satisfy r <= 1",
            "we only keep track of codes with r > 1",
            "",
            "Rx, Ry, ax, ay, bx, by, n, k, d, r",
        ]
    )


if __name__ == "__main__":
    main()
