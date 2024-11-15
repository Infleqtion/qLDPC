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

from __future__ import annotations

import ast
import re

import qldpc.cache
import qldpc.external.gap

CACHE_NAME = "qldpc_codes"


@qldpc.cache.use_disk_cache(CACHE_NAME)
def get_code(code: str) -> tuple[list[list[int]], int | None]:
    """Retrieve a group from GAP."""

    # run GAP commands
    if not qldpc.external.gap.is_installed():
        raise ValueError("GAP 4 is not installed")
    commands = [
        'LoadPackage("guava");',
        f"code := {code};",
        "mat := CheckMat(code);",
        r'Print(LeftActingDomain(code), "\n");',
        r'for vec in mat do Print(List(vec, x -> Int(x)), "\n"); od;',
    ]
    result = qldpc.external.gap.get_result(*commands)

    if "guava package is not available" in result.stdout:
        raise ValueError("GAP package GUAVA not available")

    if not result.stdout.strip():
        raise ValueError(f"Code not recognized by the GAP package GUAVA: {code}")

    # identify base field and retrieve parity checks
    field: int | None = None
    checks = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        if field is None and (match := re.search(r"GF\(([0-9]+(\^[0-9]+)?)\)", line)):
            base, exponent, *_ = (match.group(1) + "^1").split("^")
            field = int(base) ** int(exponent)
        else:
            checks.append(ast.literal_eval(line))

    if not checks:
        raise ValueError(f"Code exists, but has no parity checks: {code}")

    return checks, field
