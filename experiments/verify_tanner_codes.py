#!/usr/bin/env python3
"""Script to verify the reproducibility of saved quantum Tanner codes

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
import functools
import glob
import os

import quantum_tanner_search as qts

from qldpc import abstract, codes

file_dir = os.path.dirname(__file__)
save_dir = os.path.join(file_dir, "quantum_tanner_codes")
paths = glob.glob(os.path.join(save_dir, "*.txt"))
num_paths = len(paths)


@functools.cache
def get_code(code_name: str, code_param: int) -> codes.ClassicalCode:
    if code_name == "Hamming":
        return codes.HammingCode(int(code_param))
    elif code_name == "CordaroWagner":
        return qts.get_cordaro_wagner_code(int(code_param))
    elif code_name == "Mittal":
        return qts.get_mittal_code(int(code_param))
    raise ValueError(f"Code not recognized: {code_name}")


@functools.cache
def get_group(group_order: int, group_index: int) -> abstract.SmallGroup:
    return abstract.SmallGroup(group_order, group_index)


for idx, path in enumerate(paths):
    print(f"{idx}/{num_paths}")
    parts = path.split("_")
    group_order, group_index = parts[-3].split("-")[-2:]
    code_name, code_param = parts[-2].split("-")
    seed = int(parts[-1].strip(".txt")[1:])

    group_id = f"SmallGroup-{group_order}-{group_index}"
    base_code_id = parts[-2]
    assert os.path.basename(path) == f"qtcode_{group_id}_{base_code_id}_s{seed}.txt"
    old_code = codes.QTCode.load(path)

    base_code = get_code(code_name, int(code_param))
    group = get_group(int(group_order), int(group_index))
    new_code = codes.QTCode.random(group, base_code, seed=seed)
    assert old_code == new_code

print("done")
