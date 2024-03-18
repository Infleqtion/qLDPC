#!/usr/bin/env python3
import sys

import checks_superstaq

if __name__ == "__main__":
    skip_args = ["--skip", "requirements", "build_docs"]
    exclude_args = ["--exclude", "qldpc/indexed_groups.py"]
    exit(checks_superstaq.all_.run(*skip_args, *exclude_args, "--", *sys.argv[1:]))
