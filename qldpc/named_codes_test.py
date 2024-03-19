"""Unit tests for named_codes.py

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

import subprocess
import unittest.mock

import pytest

from qldpc import named_codes


def get_mock_process(stdout: str) -> subprocess.CompletedProcess[str]:
    """Fake process with the given stdout."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def test_get_parity_checks() -> None:
    """Retrive parity check matrix from GAP 4."""

    # GAP is not installed
    process = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
    with (
        pytest.raises(ValueError, match="not installed"),
        unittest.mock.patch("subprocess.run", return_value=process),
    ):
        named_codes.get_parity_checks("")

    name = "RepetitionCode(2)"
    check = [1, 1]
    mock_process = get_mock_process("\n" + str(check))
    with (
        unittest.mock.patch("qldpc.named_codes.gap_is_installed", return_value=True),
        unittest.mock.patch("subprocess.run", return_value=mock_process),
    ):
        assert named_codes.get_parity_checks(name) == [check]
