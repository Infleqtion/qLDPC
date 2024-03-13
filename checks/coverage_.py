#!/usr/bin/env python3
import sys

import checks_superstaq

if __name__ == "__main__":
    if sys.argv[1:]:
        # check coverage for the provided arguments
        exit(checks_superstaq.coverage_.run(*sys.argv[1:]))

    # Check that each file is covered by its own data file.
    # Start by identifying files that should be covered.
    tracked_files = checks_superstaq.check_utils.get_tracked_files("*.py")
    coverage_files = checks_superstaq.check_utils.exclude_files(
        tracked_files, ["checks/*.py", "*__init__.py", "*_test.py"]
    )

    # run checks on individual files
    exit_codes = {}
    for file in coverage_files:
        exit_codes[file] = checks_superstaq.coverage_.run(file)

    # print warnings for files that are not covered
    for file, exit_code in exit_codes.items():
        if exit_code:
            checks_superstaq.check_utils.warning(f"Coverage failed for {file}.")

    exit(sum(exit_codes.values()))
