#!/usr/bin/env python3
"""Quantum Tanner code search experiment."""

import concurrent.futures
import hashlib
import os
from collections.abc import Hashable, Iterator

from qldpc import abstract, codes


def get_deterministic_hash(*inputs: Hashable, num_bytes: int = 4) -> int:
    """Get a deterministic hash from the given inputs."""
    input_bytes = repr(inputs).encode("utf-8")
    hash_bytes = hashlib.sha256(input_bytes).digest()
    return int.from_bytes(hash_bytes[:num_bytes], byteorder="big", signed=False)


def get_small_groups(max_order: int = 20) -> Iterator[tuple[int, int]]:
    """Finite groups by order and index."""
    for order in range(3, max_order + 1):
        for index in range(1, abstract.SmallGroup.number(order) + 1):
            yield order, index


def get_cordaro_wagner_code(length: int) -> codes.ClassicalCode:
    """Cordaro Wagner code with a given block length."""
    return codes.ClassicalCode.from_name(f"CordaroWagnerCode({length})")


def get_mittal_code(length: int) -> codes.ClassicalCode:
    """Modified Hammming code of a given block length."""
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


def get_base_codes() -> Iterator[tuple[codes.ClassicalCode, str]]:
    """Iterator over classical codes and their identifiers."""
    for rr in [2, 3]:
        yield codes.HammingCode(rr), f"Hamming-{rr}"
    for nn in [3, 4, 5, 6]:
        yield get_cordaro_wagner_code(nn), f"CordaroWagner-{nn}"
    for nn in [4, 5, 6]:
        yield get_mittal_code(nn), f"Mittal-{nn}"


def run_and_save(
    group_order: int,
    group_index: int,
    base_code: codes.ClassicalCode,
    base_code_id: str,
    sample: int,
    num_samples: int,
    num_trials: int,
    *,
    identify_completion_text: bool = False,
    override_existing_data: bool = False,
    silent: bool = False,
) -> None:
    """Make a random quantum Tanner code, compute its distance, and save it to a text file.

    The multiprocessing module is unable to properly handle SymPy PermutationGroup objects, so we
    have to construct groups here from their identifying data.  ClassicalCode objects seem to be
    fine, though.
    """
    group = abstract.SmallGroup(group_order, group_index)
    group_id = f"SmallGroup-{group_order}-{group_index}"

    if group_order < base_code.num_bits:
        # the base code is too large for this group
        return None

    seed = get_deterministic_hash(group_order, group_index, base_code.matrix.tobytes(), sample)
    file = f"qtcode_{group_id}_{base_code_id}_s{seed}.txt"
    path = os.path.join(save_dir, file)

    if os.path.isfile(path) and not override_existing_data:
        # we already have the data for this code, so there is nothing to do
        return None

    if not silent:
        job_id = f"{group_id} {base_code_id} {sample}/{num_samples}"
        print(job_id)

    # construct code and compue its parameters
    code = codes.QTCode.random(group, base_code, seed=seed)
    code_params = code.get_code_params(bound=num_trials)

    if not silent:
        completion_text = ""
        if identify_completion_text:
            completion_text += f" ({job_id})"
        completion_text += f" code parameters: {code_params}"
        print(completion_text)

    headers = [
        f"distance trials: {num_trials}",
        f"code parameters: {code_params}",
    ]
    code.save(path, *headers)


if __name__ == "__main__":
    num_samples = 100  # per choice of group and subcode
    num_trials = 1000  # for code distance calculations

    max_concurrent_tasks = os.cpu_count() or 1
    save_dir = os.path.join(os.path.dirname(__file__), "quantum_tanner_codes")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_concurrent_tasks) as executor:

        for group_order, group_index in get_small_groups():
            for base_code, base_code_id in get_base_codes():
                for sample in range(num_samples):
                    executor.submit(
                        run_and_save,
                        group_order,
                        group_index,
                        base_code,
                        base_code_id,
                        sample,
                        num_samples,
                        num_trials,
                        identify_completion_text=max_concurrent_tasks > 1,
                    )

                    if base_code.num_bits == group_order:
                        # there is only one possible instance of this random code
                        break
