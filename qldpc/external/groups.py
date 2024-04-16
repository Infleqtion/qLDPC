"""Module for loading groups from the GAP computer algebra system

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

import os
import re
import subprocess
import urllib.error
import urllib.request
from collections.abc import Sequence

import qldpc.cache

CACHE_NAME = "qldpc_groups"
GENERATORS_LIST = list[list[tuple[int, ...]]]
GROUPNAMES_URL = "https://people.maths.bris.ac.uk/~matyd/GroupNames/"


def maybe_get_webpage(order: int) -> str | None:
    """Try to retrieve the webpage listing all groups up to a given order."""
    try:
        url = GROUPNAMES_URL + ("index500.html" if order > 60 else "")
        page = urllib.request.urlopen(url)
        return page.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError):
        # we cannot access the webapage
        return None


def get_group_url(order: int, index: int) -> str | None:
    """Retrieve the webpage of an indexed GAP group on GroupNames.org."""

    # get the HTML for the page with all groups
    page_html = maybe_get_webpage(order)
    if page_html is None:
        # we cannot access the webapage
        return None

    # extract section with the specified group
    loc = page_html.find(f"<td>{order},{index}</td>")
    if loc == -1:
        raise ValueError(f"Group {order},{index} not found on GroupNames.org")

    end = loc + page_html[loc:].find("\n")
    start = loc - page_html[:loc][::-1].find("\n")
    section = page_html[start:end]

    # extract first link from this section
    match = re.search(r'href="([^"]*)"', section)
    if match is None:
        raise ValueError(f"Webpage for group {order},{index} not found")

    # return url for the desired group
    return GROUPNAMES_URL + match.group(1)


def get_generators_from_groupnames(group: str) -> GENERATORS_LIST | None:
    """Retrieve GAP group generators from GroupNames.org."""

    # extract order and index of a SmallGroup
    match = re.match(r"SmallGroup\(([0-9]+),([0-9]+)\)", group)
    if match:
        order, index = map(int, match.groups())
    else:
        # this group is not indexed in GroupNames.org
        return None

    # load web page for the specified group
    group_url = get_group_url(order, index)
    if group_url is None:
        # we cannot access the webapage
        return None
    group_page = urllib.request.urlopen(group_url)
    group_page_html = group_page.read().decode("utf-8")

    # extract section with the generators we are after
    loc = group_page_html.lower().find("permutation representation")
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


def sanitize_gap_commands(commands: Sequence[str]) -> tuple[str, ...]:
    """Sanitize GAP commands: don't format Print statements, and quit at the end."""
    stream = "__stream__"
    prefix = [
        f"{stream} := OutputTextUser();",
        f"SetPrintFormattingStatus({stream}, false);",
    ]
    suffix = ["QUIT;"]
    commands = [cmd.replace("Print(", f"PrintTo({stream}, ") for cmd in commands]
    return tuple(prefix + commands + suffix)


def get_gap_result(*commands: str) -> subprocess.CompletedProcess[str]:
    """Get the output from the given GAP commands."""
    commands = sanitize_gap_commands(commands)
    shell_commands = ["gap", "-q", "--quitonbreak", "-c", " ".join(commands)]
    result = subprocess.run(shell_commands, capture_output=True, text=True)
    return result


def get_generators_with_gap(group: str) -> GENERATORS_LIST | None:
    """Retrieve GAP group generators from GAP directly."""

    if not gap_is_installed():
        return None

    # run GAP commands
    commands = [
        f"G := {group};",
        "iso := IsomorphismPermGroup(G);",
        "permG := Image(iso, G);",
        "gens := GeneratorsOfGroup(permG);",
        r'for gen in gens do Print(gen, "\n"); od;',
    ]
    result = get_gap_result(*commands)

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


@qldpc.cache.use_disk_cache(CACHE_NAME)
def get_generators(group: str) -> GENERATORS_LIST:
    """Retrieve GAP group generators."""

    generators = get_generators_with_gap(group)
    if generators is not None:
        return generators

    generators = get_generators_from_groupnames(group)
    if generators is not None:
        return generators

    message = [
        "Cannot build GAP group:",
        "- local database does not contain the group",
        "- GAP 4 is not installed",
    ]
    if group.startswith("SmallGroup"):
        message.append("- GroupNames.org is unreachable")
    else:
        message.append("- group not indexed by GroupNames.org")
    raise ValueError("\n".join(message))


@qldpc.cache.use_disk_cache(CACHE_NAME)
def get_small_group_number(order: int) -> int:
    """Get the number of 'SmallGroup's of a given order."""
    if gap_is_installed():
        command = f"Print(NumberSmallGroups({order}));"
        return int(get_gap_result(command).stdout)

    # get the HTML for the page with all groups
    page_html = maybe_get_webpage(order)
    if page_html is None:
        # we cannot access the webapage
        raise ValueError("Cannot determine the number of small groups")

    matches = re.findall(rf"<td>{order},([0-9]+)</td>", page_html)
    return max(int(match) for match in matches)


def get_small_group_structure(order: int, index: int) -> str:
    """Get a description of the structure of a SmallGroup from GAP."""
    # if we have the structure cached, retrieve it
    key = (order, index)
    cache = qldpc.cache.get_disk_cache(CACHE_NAME)
    if structure := cache.get(key, None):
        return structure

    # try to retrieve the structure from GAP
    name = f"SmallGroup({order},{index})"
    if gap_is_installed():
        command = f"Print(StructureDescription({name}));"
        result = get_gap_result(command)
        structure = result.stdout.strip()

        if not structure:
            raise ValueError(f"Group not recognized by GAP: {name}")

        cache[key] = structure
        return structure

    # return the name of the group
    return name
