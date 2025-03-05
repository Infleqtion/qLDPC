#!/usr/bin/env python3
import sys

import checks_superstaq

if __name__ == "__main__":
    skip_and_ruff = ["--skip", "configs", "requirements", "build_docs", "--ruff"]
    exit(checks_superstaq.all_.run(*skip_and_ruff, *sys.argv[1:]))
