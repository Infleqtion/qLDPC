"""Module for loading error-correcting codes from the GAP computer algebra system

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

import ast
import re

from qldpc.small_groups import gap_is_installed, get_gap_result


def get_code(name: str) -> tuple[list[list[int]], int | None]:
    """Retrieve a group from GAP."""

    if not gap_is_installed():
        raise ValueError("GAP 4 is not installed")

    # run GAP commands
    commands = [
        'LoadPackage("guava");',
        f"code := {name};",
        "mat := CheckMat(code);",
        r'for vec in mat do Print(List(vec, x -> Int(x)), "\n"); od;',
        r'Print(LeftActingDomain(code), "\n");',
    ]
    result = get_gap_result(commands)

    if "guava package is not available" in result.stdout:
        raise ValueError("GAP package GUAVA not available")

    if not result.stdout.strip():
        raise ValueError(f"Code not recognized by the GAP package GUAVA: {name}")

    # retrieve checks row by row
    checks = []
    field = None
    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        match = re.search(r"GF\(([0-9]+(\^[0-9]+)?)\)", line)
        if match:
            base, exponent, *_ = map(int, (match.group(1) + "^1").split("^"))
            field = base**exponent
        else:
            checks.append(ast.literal_eval(line))

    return checks, field
