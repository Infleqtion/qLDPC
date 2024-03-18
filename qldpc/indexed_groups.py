#!/usr/bin/env python3
import bs4
import io
import pandas
import urllib.request

import sympy.combinatorics as comb

from qldpc import abstract


BASE_URL = "https://people.maths.bris.ac.uk/~matyd/GroupNames/"


def get_group_page(order: int, index: int):
    """Get the webpage for an indexed group on GroupNames.org."""

    # load index
    extra = "index500.html" if order > 60 else ""
    page = urllib.request.urlopen(BASE_URL + extra)
    html = page.read().decode("utf-8")

    # extract section of groups with a given order
    loc = html.find(f"Groups of order {order}")
    end_str = "</table>"
    end = html[loc:].find(end_str)
    section = html[loc : loc + end + len(end_str)]

    # extract tables in this section
    tables = pandas.read_html(io.StringIO(section), extract_links="body")
    if not tables:
        raise ValueError(f"Groups of order {order} not found")

    # extract the row corresponding to the specified group
    table = tables[0]
    row = table.loc[table["ID"] == (f"{order},{index}", None)]

    if row.empty:
        raise ValueError(f"Group {order},{index} not found")

    # identify url for that group
    group_url = BASE_URL + row.iloc[0, 0][1]
    return group_url


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
    soup = bs4.BeautifulSoup(section, "html.parser")
    gen_strings = [gen for gen in soup.find("pre").get_text(separator="\n").splitlines() if gen]

    # build generators
    generators = []
    for gen_str in gen_strings:
        cycles_str = gen_str[1:-1].split(")(")
        cycles = [tuple(map(int, cycle.split())) for cycle in cycles_str]
        generators.append(abstract.GroupMember(cycles))

    return generators


class IndexedGroup(abstract.Group):
    """Groups indexed on GroupNames.org."""

    def __init__(self, order: int, index: int) -> None:
        group = comb.PermutationGroup(*get_group_generators(order, index))
        super().__init__(group)
