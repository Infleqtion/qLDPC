"""Module for loading groups indexed by the GAP computer algebra system

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

import functools
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Callable, Hashable
from typing import Any

import diskcache
import platformdirs

GENERATORS_LIST = list[list[tuple[int, ...]]]
GROUPNAMES_URL = "https://people.maths.bris.ac.uk/~matyd/GroupNames/"


def get_group_url(order: int, index: int) -> str | None:
    """Retrieve the webpage of an indexed GAP group on GroupNames.org."""

    try:
        # load index
        extra = "index500.html" if order > 60 else ""
        index_url = GROUPNAMES_URL + extra
        index_page = urllib.request.urlopen(index_url)
        index_page_html = index_page.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError):
        # we cannot access the webapage
        return None

    # extract section with the specified group
    loc = index_page_html.find(f"<td>{order},{index}</td>")
    if loc == -1:
        raise ValueError(f"Group {order},{index} not found at {index_url}")

    end = loc + index_page_html[loc:].find("\n")
    start = loc - index_page_html[:loc][::-1].find("\n")
    section = index_page_html[start:end]

    # extract first link from this section
    match = re.search(r'href="([^"]*)"', section)
    if match is None:
        raise ValueError(f"Webpage for group {order},{index} not found")

    # return url for the desired group
    return GROUPNAMES_URL + match.group(1)


def get_generators_from_groupnames(order: int, index: int) -> GENERATORS_LIST | None:
    """Retrieve GAP group generators from GroupNames.org."""

    # load web page for the specified group
    group_url = get_group_url(order, index)
    if group_url is None:
        # we cannot access the webapage
        return None
    group_page = urllib.request.urlopen(group_url)
    group_page_html = group_page.read().decode("utf-8")

    # extract section with the generators we are after
    loc = group_page_html.find("Permutation representations")
    end = group_page_html[loc:].find("copytext")  # go until the first copy-able text
    section = group_page_html[loc : loc + end]

    # isolate generator text
    section = section[section.find("<pre") :]
    match = re.search(r">((?:.|\n)*?)<\/pre>", section)
    if match is None:
        raise ValueError(f"Generators for group {order},{index} not found")
    gen_strings = match.group(1).split("<br>\n")

    # build generators
    generators = []
    for string in gen_strings:
        cycles_str = string[1:-1].split(")(")
        cycles = [tuple(map(int, cycle.split())) for cycle in cycles_str]

        # decrement integers in the cycle by 1 to account for 0-indexing
        cycles = [tuple(index - 1 for index in cycle) for cycle in cycles]

        generators.append(cycles)

    return generators


def gap_is_installed() -> bool:
    """Is GAP 4 installed?"""
    commands = ["script", "-c", "gap --version", os.devnull]
    result = subprocess.run(commands, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    return len(lines) == 2 and lines[1].startswith("GAP 4")


def sanitize_gap_commands(commands: list[str]) -> list[str]:
    """Sanitize GAP commands: don't format Print statements, and quit at the end."""
    stream = "__stream__"
    prefix = [
        f"{stream} := OutputTextUser();",
        f"SetPrintFormattingStatus({stream}, false);",
    ]
    suffix = ["QUIT;"]
    commands = [cmd.replace("Print(", f"PrintTo({stream}, ") for cmd in commands]
    return prefix + commands + suffix


def get_gap_result(commands: list[str]) -> subprocess.CompletedProcess[str]:
    """Get the output from the given GAP commands."""
    commands = sanitize_gap_commands(commands)
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".gap") as script:
        script.write("\n".join(commands))
        script_name = script.name
    shell_commands = ["gap", "-q", "--quitonbreak", script_name]
    result = subprocess.run(shell_commands, capture_output=True, text=True)
    os.remove(script_name)
    return result


def get_generators_with_gap(order: int, index: int) -> GENERATORS_LIST | None:
    """Retrieve GAP group generators from GAP directly."""

    if not gap_is_installed():
        return None

    # run GAP commands
    group = f"SmallGroup({order},{index})"
    commands = [
        f"G := {group};",
        "iso := IsomorphismPermGroup(G);",
        "permG := Image(iso, G);",
        "gens := GeneratorsOfGroup(permG);",
        r'for gen in gens do Print(gen, "\n"); od;',
    ]
    result = get_gap_result(commands)

    if not result.stdout.strip():
        raise ValueError(f"Group not recognized by GAP: {group}")

    # collect generators
    generators = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        # extract list of cycles, where each cycle is a tuple of integers
        cycles_str = line[1:-1].split(")(")
        try:
            cycles = [tuple(map(int, cycle.split(","))) for cycle in cycles_str]
        except ValueError:
            raise ValueError(f"Cannot extract cycles from string: {line}")

        # decrement integers in the cycle by 1 to account for 0-indexing
        cycles = [tuple(index - 1 for index in cycle) for cycle in cycles]
        generators.append(cycles)

    return generators


def use_disk_cache(
    cache_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Cache new results to disk, and retrieve existing results (if available) from the cache."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:

        @functools.wraps(func)
        def wrapper(*args: Hashable) -> Any:

            # retrieve results from cache, if available
            cache = diskcache.Cache(platformdirs.user_cache_dir(cache_name))
            generators = cache.get(args, None)
            if generators is not None:
                return generators

            # compute results and save to cache
            result = func(*args)
            cache[args] = result
            return result

        return wrapper

    return decorator


@use_disk_cache("qldpc_groups")
def get_generators(order: int, index: int) -> GENERATORS_LIST:
    """Retrieve GAP group generators."""
    for get_generators_func in [
        get_generators_with_gap,
        get_generators_from_groupnames,
    ]:
        generators = get_generators_func(order, index)
        if generators is not None:
            return generators

    # we could not find or retrieve the generators :(
    message = [
        "Cannot build GAP group:",
        "- local database does not contain the group",
        "- GAP 4 not installed",
        "- GroupNames.org is unreachable",
    ]
    raise ValueError("\n".join(message))
