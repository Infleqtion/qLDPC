#!/usr/bin/env python3
"""Quantum Tanner code search experiment."""

import hashlib
import itertools
import os
from collections.abc import Hashable, Iterator

from qldpc import abstract, codes


def get_deterministic_hash(*inputs: Hashable, bytes: int = 4):
    """Get a deterministic hash from the given inputs."""
    input_bytes = repr(inputs).encode("utf-8")
    hash_bytes = hashlib.sha256(input_bytes).digest()
    return int.from_bytes(hash_bytes[:bytes], byteorder="big", signed=False)


def get_cordaro_wagner_code(length: int) -> codes.ClassicalCode:
    """Cordaro Wagner code with a given block length."""
    return codes.ClassicalCode.from_name(f"CordaroWagnerCode({length})")


def get_mittal_code(length: int) -> codes.ClassicalCode:
    """Modified Hammming codes of a given block length."""
    name = "MittalCode"
    base_code = codes.HammingCode(3)
    if length == 4:
        code = base_code.shorten(2, 3).puncture(4)
    elif length == 5:
        code = base_code.shorten(2, 3)
    elif length == 6:
        code = base_code.shorten(3)
    else:
        raise ValueError(f"Unrecognized length for {name}: {length}")
    setattr(code, "_name", name)
    return code


def get_groups(max_order: int = 20) -> Iterator[abstract.SmallGroup]:
    """Iterator over all finite groups up to a given order."""
    for order in range(2, max_order + 1):
        for index in range(1, abstract.SmallGroup.number(order) + 1):
            yield abstract.SmallGroup(order, index)


def get_codes_with_tags() -> Iterator[codes.ClassicalCode]:
    """Iterator over several small classical codes and their identifier tags."""
    yield from itertools.chain(
        ((codes.HammingCode(rr), f"hamming-{rr}") for rr in [2, 3]),
        ((get_cordaro_wagner_code(nn), f"CW-{nn}") for nn in [3, 4, 5, 6]),
        ((get_mittal_code(nn), f"M-{nn}") for nn in [4, 5, 6]),
    )


if __name__ == "__main__":
    num_samples = 100
    num_trials = 1000
    save_dir = os.path.join(os.path.dirname(__file__), "quantum_tanner_codes")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for group, (base_code, base_tag) in itertools.product(get_groups(), get_codes_with_tags()):
        if group.order < base_code.num_bits:
            # no subset of the group has as many elements as the code block length
            continue

        for sample in range(num_samples):
            seed = get_deterministic_hash(group.order, group.index, base_code.matrix.tobytes())
            code = codes.QTCode.random(
                group, base_code, bipartite=False, one_subset=False, seed=seed
            )

            code_params = code.get_code_params(bound=num_trials)
            headers = [
                f"group: {group}",
                f"base code: {base_code.name}",
                f"distance trials: {num_trials}",
                f"code parameters: {code_params}",
            ]

            file = f"qtcode_group-{group.order}-{group.index}_{base_tag}_s{seed}.txt"
            path = os.path.join(save_dir, file)
            code.save(path, *headers)
            exit()
