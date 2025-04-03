"""Module for loading groups from GroupNames or the GAP computer algebra system

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

from __future__ import annotations

import re
import urllib.error
import urllib.request

import qldpc.cache
import qldpc.external.gap

CACHE_NAME = "qldpc_groups"
GENERATORS_LIST = list[list[tuple[int, ...]]]
GROUPNAMES_URL = "https://people.maths.bris.ac.uk/~matyd/GroupNames/"


@qldpc.cache.use_disk_cache(CACHE_NAME)
def get_generators(group: str) -> GENERATORS_LIST:
    """Retrieve GAP group generators."""

    if generators := KNOWN_GROUPS.get(group):
        return generators

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
    if qldpc.external.gap.is_installed():
        command = f"Print(NumberSmallGroups({order}));"
        return int(qldpc.external.gap.get_result(command).stdout)

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
    if qldpc.external.gap.is_installed():
        command = f"Print(StructureDescription({name}));"
        result = qldpc.external.gap.get_result(command)
        structure = result.stdout.strip()

        if not structure:
            raise ValueError(f"Group not recognized by GAP: {name}")

        cache[key] = structure
        return structure

    # return the name of the group
    return name


def get_generators_with_gap(group: str) -> GENERATORS_LIST | None:
    """Retrieve GAP group generators from GAP directly."""

    if not qldpc.external.gap.is_installed():
        return None

    # run GAP commands
    commands = [
        'LoadPackage("guava");',
        f"G := {group};",
        "iso := IsomorphismPermGroup(G);",
        "permG := Image(iso, G);",
        "gens := GeneratorsOfGroup(permG);",
        r'for gen in gens do Print(gen, "\n"); od;',
    ]
    result = qldpc.external.gap.get_result(*commands)

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
            cycles = [tuple(map(int, cycle.split(","))) for cycle in cycles_str if cycle]
        except ValueError:
            raise ValueError(f"Cannot extract cycles from string: {line}")

        # decrement integers in the cycle by 1 to account for 0-indexing
        cycles = [tuple(index - 1 for index in cycle) for cycle in cycles]
        generators.append(cycles)

    return generators


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


def maybe_get_webpage(order: int) -> str | None:
    """Try to retrieve the webpage listing all groups up to a given order."""
    try:
        url = GROUPNAMES_URL + ("index500.html" if order > 60 else "")
        page = urllib.request.urlopen(url)
        return page.read().decode("utf-8")
    except (urllib.error.URLError, urllib.error.HTTPError):
        # we cannot access the webapage
        return None


KNOWN_GROUPS: dict[str, GENERATORS_LIST] = {
    "AutomorphismGroup(CheckMatCode([[1,0,0,0,1,1,1,0,1,1],[0,1,0,0,1,0,0,1,1,0],[0,0,1,0,1,1,1,0,0,0],[0,0,0,1,1,1,0,1,1,1]],GF(2)))": [
        [(3, 7), (4, 5), (8, 9)],
        [(1, 5), (2, 7), (3, 9), (6, 8)],
        [(2, 6), (3, 8), (4, 5), (7, 9)],
        [(0, 9, 2, 8), (1, 6), (3, 5, 4, 7)],
        [(0, 7, 3), (1, 9, 8), (2, 4, 5)],
    ],
    "AutomorphismGroup(CheckMatCode([[1,0,0,0,1,0,1,0,1,0],[0,1,0,0,1,0,1,1,1,1],[0,0,1,0,1,1,1,1,0,1],[0,0,0,1,1,1,0,1,0,0]],GF(2)))": [
        [(2, 5), (4, 6), (7, 9)],
        [(1, 9), (3, 5), (6, 8)],
        [(1, 8), (4, 7), (6, 9)],
        [(1, 4, 9, 8, 7, 6), (2, 5, 3)],
        [(1, 7), (2, 3), (4, 8)],
        [(0, 3, 1, 4, 9, 5), (2, 8, 6)],
        [(0, 6, 8), (1, 9, 2), (3, 5, 7)],
    ],
    "AutomorphismGroup(CheckMatCode([[1,0,0,0,1,1,1,0,1,1,0,1,0,1,0],[0,1,0,0,1,0,0,1,1,0,0,1,1,1,1],[0,0,1,0,1,1,1,0,0,0,1,1,1,0,1],[0,0,0,1,1,1,0,1,1,1,1,0,1,0,0]],GF(2)))": [
        [(0, 1), (5, 12), (6, 14), (7, 9)],
        [(2, 3), (6, 9), (7, 14), (8, 11)],
        [(1, 2), (5, 8), (6, 13), (7, 10)],
        [(1, 13), (4, 12), (7, 8), (11, 14)],
        [(3, 10), (4, 8), (5, 9), (7, 12)],
        [(2, 6), (4, 12), (5, 10), (11, 14)],
        [(2, 11), (3, 8), (6, 14), (7, 9)],
        [(3, 8), (4, 10), (5, 12), (7, 9)],
        [(3, 9), (4, 12), (5, 10), (7, 8)],
    ],
    "AutomorphismGroup(CheckMatCode([[1,1,1,1,0,0,0,0,1,1,1,1],[0,0,0,0,1,1,1,1,1,1,1,1]],GF(2)))": [
        [(4, 9), (5, 8, 6, 11), (7, 10)],
        [(4, 7, 6, 5), (9, 11, 10)],
        [(2, 3), (4, 10), (5, 11), (6, 9), (7, 8)],
        [(10, 11)],
        [(6, 7), (8, 11, 10)],
        [(5, 7, 6), (8, 11, 9)],
        [(9, 11, 10)],
        [(8, 9, 11, 10)],
        [(5, 6), (10, 11)],
        [(1, 3, 2), (4, 11), (5, 8), (6, 9), (7, 10)],
        [(1, 3, 2), (4, 6, 7), (9, 11, 10)],
        [(0, 4, 10, 1, 5, 9, 2, 7, 8), (3, 6, 11)],
        [(2, 3), (4, 7), (8, 11)],
        [(0, 10, 2, 11, 1, 9), (3, 8), (4, 7)],
    ],
}
