# SPDX-License-Identifier: Apache-2.0

"""Check that the qLDPC package imports cleanly."""

import os
import pathlib
import subprocess
import sys


def test_import_in_fresh_interpreter() -> None:
    """qLDPC imports without errors or warnings in a fresh interpreter.

    The package imports every subpackage when it is first imported, so importing any qLDPC module
    runs this same import sequence.
    """
    source_dir = pathlib.Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, "-W", "error", "-c", "import qldpc"],
        capture_output=True,
        check=False,
        env=os.environ | {"PYTHONPATH": str(source_dir)},
        text=True,
    )
    assert result.returncode == 0, result.stderr
