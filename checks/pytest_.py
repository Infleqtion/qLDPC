#!/usr/bin/env python3
import sys

import checks_superstaq

if __name__ == "__main__":
    exclude = ["checks/*.py", "experiments/*.py", "*/__init__.py"]
    exit(checks_superstaq.pytest_.run(*sys.argv[1:], exclude=exclude))
