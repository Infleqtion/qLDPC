#!/usr/bin/env python3
import sys

import checks_superstaq

if __name__ == "__main__":
    skip_args = ["--ruff", "--skip", "requirements", "--"]
    exit(checks_superstaq.all_.run(*skip_args, *sys.argv[1:], sphinx_paths=["../qldpc"]))
