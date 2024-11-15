"""Module for communicating with the GAP computer algebra system

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

import re
import subprocess
from collections.abc import Sequence


def is_installed() -> bool:
    """Is GAP 4 installed?"""
    commands = ["gap", "-q", "-c", r'Print(GAPInfo.Version, "\n"); QUIT;']
    try:
        result = subprocess.run(commands, capture_output=True, text=True)
        return bool(re.match(r"\n4\.[0-9]+\.[0-9]+$", result.stdout))
    except Exception:
        return False


def sanitize_commands(commands: Sequence[str]) -> tuple[str, ...]:
    """Sanitize GAP commands: don't format Print statements, and quit at the end."""
    stream = "__stream__"
    prefix = [
        f"{stream} := OutputTextUser();",
        f"SetPrintFormattingStatus({stream}, false);",
    ]
    suffix = ["QUIT;"]
    commands = [cmd.replace("Print(", f"PrintTo({stream}, ") for cmd in commands]
    return tuple(prefix + commands + suffix)


def get_result(*commands: str) -> subprocess.CompletedProcess[str]:
    """Get the output from the given GAP commands."""
    commands = sanitize_commands(commands)
    shell_commands = ["gap", "-q", "--quitonbreak", "-c", " ".join(commands)]
    result = subprocess.run(shell_commands, capture_output=True, text=True)
    return result
