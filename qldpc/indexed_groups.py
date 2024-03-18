"""Module for loading indexed groups from the GAP computer algebra system

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

import re
import subprocess
import urllib.request

import sympy.combinatorics as comb

from qldpc import abstract


GROUPNAMES_URL = "https://people.maths.bris.ac.uk/~matyd/GroupNames/"


def get_groupnames_url(order: int, index: int) -> str:
    """Get the webpage for an indexed group on GroupNames.org."""

    # load index
    extra = "index500.html" if order > 60 else ""
    page = urllib.request.urlopen(GROUPNAMES_URL + extra)
    page_text = page.read().decode("utf-8")

    # extract section with the specified group
    loc = page_text.find(f"<td>{order},{index}</td>")
    if loc == -1:
        raise ValueError(f"Group {order},{index} not found")
    end = loc + page_text[loc:].find("\n")
    start = loc - page_text[:loc][::-1].find("\n")
    section = page_text[start:end]

    # extract first link from this section
    pattern = r'href="([^"]*)"'
    match = re.search(pattern, section)
    if match is None:
        raise ValueError(f"Webpage for group {order},{index} not found")

    return GROUPNAMES_URL + match.group(1)


def get_generators_from_groupnames(order: int, index: int) -> list[abstract.GroupMember]:
    """Get a finite group by its index on GroupNames.org."""

    # load web page for the specified group
    url = get_groupnames_url(order, index)
    page = urllib.request.urlopen(url)
    html = page.read().decode("utf-8")

    # extract section with the generators we are after
    loc = html.find("Permutation representations")
    end = html[loc:].find("copytext")  # go until the first copy-able text
    section = html[loc : loc + end]

    # isolate generator text
    section = section[section.find("<pre") :]
    pattern = r">((?:.|\n)*?)<\/pre>"
    match = re.search(pattern, section)
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

        # add generator
        generators.append(abstract.GroupMember(cycles))

    return generators


def get_generators_with_GAP(order: int, index: int) -> list[abstract.GroupMember]:
    """Get a finite group from the GAP computer algebra system."""
    # build GAP command
    gap_lines = [
        f"G := SmallGroup({order},{index});",
        "iso := IsomorphismPermGroup(G);",
        "permG := Image(iso,G);",
        "gens := GeneratorsOfGroup(permG);",
        r'for gen in gens do Print(gen, "\n"); od;',
        "QUIT;",
    ]
    gap_script = "".join(gap_lines)

    # run GAP
    result = subprocess.run(["gap", "-q", "-c", gap_script], capture_output=True, text=True)

    # collect generators
    generators = []
    for line in result.stdout.split("\n")[:-1]:
        cycles_str = line[1:-1].split(")(")

        try:
            cycles = [tuple(map(int, cycle.split(","))) for cycle in cycles_str]
        except ValueError:
            raise ValueError(f"Cannot extract cycle from string: {line}")

        # decrement integers in the cycle by 1 to account for 0-indexing
        cycles = [tuple(index - 1 for index in cycle) for cycle in cycles]

        # add generator
        generators.append(abstract.GroupMember(cycles))

    return generators


def gap_is_installed() -> bool:
    """Is GAP installed?"""
    result = subprocess.run(["script", "-c", "gap --version"], capture_output=True, text=True)
    lines = result.stdout.split("\n")
    return len(lines) == 2 and lines[1][:3] == "GAP"


def can_connect_to_groupnames() -> bool:
    """Can we connect to GroupNames.org?"""
    try:
        urllib.request.urlopen(GROUPNAMES_URL)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


class IndexedGroup(abstract.Group):
    """Groups indexed on GroupNames.org."""

    def __init__(self, order: int, index: int, with_GAP: bool = True) -> None:
        if gap_is_installed():
            generators = get_generators_with_GAP(order, index)
        elif can_connect_to_groupnames():
            generators = get_generators_from_groupnames(order, index)
        else:
            raise ValueError(
                "Cannot build GAP group\nGAP not installed and GroupNames.org is unreachable"
            )
        group = comb.PermutationGroup(*generators)
        super().__init__(group)
