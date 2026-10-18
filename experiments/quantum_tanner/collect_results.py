#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Collect quantum Tanner code search results into a local database.

This script reads the files of codes saved by run_randomized_search.py, and stores their parameters
in a disk cache (by default, .code_cache in this directory).  Each cache entry is keyed by
(group_order, group_index, base_code_id), and holds a list with one tuple
(n, k, d, num_trials, stabilizer_weight) per saved code.  Codes with no logical qubits are skipped.

Example:
    python experiments/quantum_tanner/collect_results.py
"""

from __future__ import annotations

import argparse
import collections
import pathlib

import run_randomized_search as search

import qldpc.cache

CACHE_DIR = pathlib.Path(__file__).parent  # default directory of the database
CACHE_NAME = ".code_cache"  # name of the database

####################################################################################################
# command-line interface


def main() -> None:
    """Collect the parameters of saved quantum Tanner codes into a database."""
    args = parse_args()
    cache = qldpc.cache.get_disk_cache(CACHE_NAME, cache_dir=args.cache_dir)
    for key, data in get_code_data(args.save_dir).items():
        cache[key] = data


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    search.add_save_dir_argument(parser)
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=CACHE_DIR,
        help=f"directory of the database {CACHE_NAME} (default: the directory of this script)",
    )
    return parser.parse_args()


####################################################################################################
# data collection


def get_code_data(
    save_dir: pathlib.Path,
) -> dict[tuple[int, int, str], list[tuple[int | float, ...]]]:
    """Collect the parameters of saved quantum Tanner codes, grouped by group and base code.

    Args:
        save_dir: The directory of saved codes.

    Returns:
        A dictionary that maps (group_order, group_index, base_code_id) to a list of tuples
        (n, k, d, num_trials, stabilizer_weight), with one tuple per saved code that has k > 0.
    """
    data = collections.defaultdict(list)
    for path in sorted(save_dir.glob("qtcode_*.txt")):
        group_order, group_index, base_code_id, _ = search.parse_code_path(path)
        num_trials, code_params, weight = search.read_code_header(path)

        # this code has no logical qubits, so we can ignore it
        if code_params[1] == 0:
            continue

        data[group_order, group_index, base_code_id].append((*code_params, num_trials, weight))
    return dict(data)


if __name__ == "__main__":
    main()
