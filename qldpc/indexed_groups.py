"""Module for loading indexed groups from GroupNames.org

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
import urllib.request

import sympy.combinatorics as comb

from qldpc import abstract


BASE_URL = "https://people.maths.bris.ac.uk/~matyd/GroupNames/"


def get_group_page(order: int, index: int):
    """Get the webpage for an indexed group on GroupNames.org."""

    # load index
    extra = "index500.html" if order > 60 else ""
    page = urllib.request.urlopen(BASE_URL + extra)
    page_text = page.read().decode("utf-8")

    # extract section with the specified group
    index_loc = page_text.find(f"<td>{order},{index}</td>")
    if index_loc == -1:
        raise ValueError(f"Group {order},{index} not found")
    end = index_loc + page_text[index_loc:].find("\n")
    start = index_loc - page_text[:index_loc][::-1].find("\n")
    section = page_text[start:end]

    # extract first link from this section
    pattern = r'href="([^"]*)"'
    match = re.search(pattern, section)
    return BASE_URL + match.group(1)


def get_group_generators(order: int, index: int):
    """Get a finite group by its index on GroupNames.org."""

    # load web page for the specified group
    url = get_group_page(order, index)
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


class IndexedGroup(abstract.Group):
    """Groups indexed on GroupNames.org."""

    def __init__(self, order: int, index: int) -> None:
        generators = get_group_generators(order, index)
        group = comb.PermutationGroup(*generators)
        super().__init__(group)
