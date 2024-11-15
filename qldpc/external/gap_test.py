"""Unit tests for gap.py

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

import subprocess
import unittest.mock

from qldpc import external


def get_mock_process(stdout: str) -> subprocess.CompletedProcess[str]:
    """Fake process with the given stdout."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def test_is_installed() -> None:
    """Is GAP 4 installed?"""
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("\n4.12.1")):
        assert external.gap.is_installed()
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("")):
        assert not external.gap.is_installed()
    with unittest.mock.patch("subprocess.run", side_effect=Exception):
        assert not external.gap.is_installed()


def test_get_result() -> None:
    """Run GAP commands and retrieve the GAP output."""
    output = "test"
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process(output)):
        assert external.gap.get_result().stdout == output
