#!/usr/bin/env python3
import io
import pandas
import urllib.request


base_url = "https://people.maths.bris.ac.uk/~matyd/GroupNames/"


def get_group_page(order: int, index: int):
    """Get the webpage for an indexed group on GroupNames.org."""

    extra = "index500.html" if order > 60 else ""
    page = urllib.request.urlopen(base_url + extra)
    html = page.read().decode("utf-8")

    loc = html.find(f"Groups of order {order}")
    end_str = "</table>"
    end = html[loc:].find(end_str)

    table_html = html[loc : loc + end + len(end_str)]
    tables = pandas.read_html(io.StringIO(table_html), extract_links="body")

    if not tables:
        raise ValueError(f"Groups of order {order} not found")

    table = tables[0]
    row = table.loc[table["ID"] == (f"{order},{index}", None)]

    if row.empty:
        raise ValueError(f"Group {order},{index} not found")

    group_page = base_url + row.iloc[0, 0][1]
    return group_page


def get_group_generators(order: int, index: int):
    """Get a finite group by its index on GroupNames.org."""

    url = get_group_page(order, index)
    return url


print(get_group_generators(16, 3))
