"""Unit tests for indexed_groups.py

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

import unittest.mock
import urllib

import pytest

from qldpc import indexed_groups


def test_get_group_url() -> None:
    """Retrieval of url for group page on GroupNames.org."""

    order, index = 2, 1
    suffix = "1/C2.html"
    table = """<table class="gptable" columns="6" style='width: 70%;'>
<tr><th width="12%"></th><th width="60%"></th><th width="5%"><a href='T.html'>d</a></th><th width="5%"><a href='R.html'>&rho;</a></th><th width="12%">Label</th><th width="7%">ID</th></tr><tr><td id="c2"><a href="1/C2.html">C<sub>2</sub></a></td><td><a href="cyclic.html">Cyclic</a> group</td><td><a href="T15.html#c2">2</a></td><td><a href="R.html#dim1+">1+</a></td><td>C2</td><td>2,1</td></tr>
</table>"""  # noqa: E501 (ignore line-too-long)

    # cannot connect to general webpage
    with unittest.mock.patch(
        "urllib.request.urlopen", side_effect=urllib.error.URLError("message")
    ):
        assert indexed_groups.get_group_url(order, index) is None

    mock_page = unittest.mock.MagicMock()
    mock_page.read.return_value = table.encode("utf-8")
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        # cannot find group webpage
        with (
            pytest.raises(ValueError, match="not found"),
            unittest.mock.patch("re.search", return_value=None),
        ):
            indexed_groups.get_group_url(order, index)

        # retrieve correct url
        group_url = indexed_groups.get_group_url(order, index)
        assert group_url == indexed_groups.GROUPNAMES_URL + suffix
