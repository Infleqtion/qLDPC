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

# strip cache wrapper
if hasattr(named_codes.get_code, "__wrapped__"):
    named_codes.get_code = named_codes.get_code.__wrapped__


def get_mock_process(stdout: str) -> subprocess.CompletedProcess[str]:
    """Fake process with the given stdout."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def test_get_code() -> None:
    """Retrive parity check matrix from GAP 4."""

    # GAP is not installed
    with (
        pytest.raises(ValueError, match="GAP 4 is not installed"),
        unittest.mock.patch("qldpc.named_codes.gap_is_installed", return_value=False),
    ):
        named_codes.get_code("")

    # GUAVA is not installed
    mock_process = get_mock_process("guava package is not available")
    with (
        pytest.raises(ValueError, match="GAP package GUAVA not available"),
        unittest.mock.patch("qldpc.named_codes.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.named_codes.get_gap_result", return_value=mock_process),
    ):
        named_codes.get_code("")

    # code not recognized by GUAVA
    mock_process = get_mock_process("\n")
    with (
        pytest.raises(ValueError, match="Code not recognized"),
        unittest.mock.patch("qldpc.named_codes.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.named_codes.get_gap_result", return_value=mock_process),
    ):
        named_codes.get_code("")

    check = [1, 1]
    mock_process = get_mock_process(f"\n{check}\nGF(3^3)")
    with (
        unittest.mock.patch("qldpc.named_codes.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.named_codes.get_gap_result", return_value=mock_process),
    ):
        assert named_codes.get_code("") == ([check], 27)
