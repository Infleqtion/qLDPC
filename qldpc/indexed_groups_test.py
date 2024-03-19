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

import subprocess
import unittest.mock
import urllib

import pytest

from qldpc import indexed_groups

ORDER, INDEX = 2, 1
GENERATORS = [[(0, 1)]]
GROUP_URL = indexed_groups.GROUPNAMES_URL + "1/C2.html"
MOCK_INDEX_HTML = """<table class="gptable" columns="6" style='width: 70%;'>
<tr><th width="12%"></th><th width="60%"></th><th width="5%"><a href='T.html'>d</a></th><th width="5%"><a href='R.html'>&rho;</a></th><th width="12%">Label</th><th width="7%">ID</th></tr><tr><td id="c2"><a href="1/C2.html">C<sub>2</sub></a></td><td><a href="cyclic.html">Cyclic</a> group</td><td><a href="T15.html#c2">2</a></td><td><a href="R.html#dim1+">1+</a></td><td>C2</td><td>2,1</td></tr>
</table>"""  # pylint: disable=line-too-long  # noqa: E501
MOCK_GROUP_HTML = """<b><a href='https://en.wikipedia.org/wiki/Group actions' title='See wikipedia' class='wiki'>Permutation representations of C<sub>2</sub></a></b><br><a id='shl1' class='shl' href="javascript:showhide('shs1','shl1','Regular action on 2 points');"><span class="nsgpn">&#x25ba;</span>Regular action on 2 points</a> - transitive group <a href="../T15.html#2t1">2T1</a><div id='shs1' class='shs'>Generators in S<sub>2</sub><br><pre class='pre' id='textgn1'>(1 2)</pre>&emsp;<button class='copytext' id='copygn1'>Copy</button><br>"""  # pylint: disable=line-too-long  # noqa: E501


def test_get_group_url() -> None:
    """Retrive url for group webpage on GroupNames.org."""

    # cannot connect to general webpage
    with unittest.mock.patch(
        "urllib.request.urlopen", side_effect=urllib.error.URLError("message")
    ):
        assert indexed_groups.get_group_url(ORDER, INDEX) is None

    mock_page = unittest.mock.MagicMock()
    mock_page.read.return_value = MOCK_INDEX_HTML.encode("utf-8")
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        # cannot find group webpage
        with (
            pytest.raises(ValueError, match="not found"),
            unittest.mock.patch("re.search", return_value=None),
        ):
            indexed_groups.get_group_url(ORDER, INDEX)

        # everything works as expected
        assert indexed_groups.get_group_url(ORDER, INDEX) == GROUP_URL


def test_get_generators_from_groupnames() -> None:
    """Retrive generators from group webpage on GroupNames.org."""

    # group url not found
    with unittest.mock.patch("qldpc.indexed_groups.get_group_url", return_value=None):
        assert indexed_groups.get_generators_from_groupnames(ORDER, INDEX) is None

    mock_page = unittest.mock.MagicMock()
    mock_page.read.return_value = MOCK_GROUP_HTML.encode("utf-8")
    with (
        unittest.mock.patch("qldpc.indexed_groups.get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        # cannot find generators
        with (
            pytest.raises(ValueError, match="not found"),
            unittest.mock.patch("re.search", return_value=None),
        ):
            indexed_groups.get_generators_from_groupnames(ORDER, INDEX)

        # everything works as expected
        assert indexed_groups.get_generators_from_groupnames(ORDER, INDEX) == GENERATORS


def test_get_generators_with_gap() -> None:
    """Retrive generators from GAP 4."""

    # GAP is not installed
    process = subprocess.CompletedProcess(args=[], returncode=0, stdout="")
    with unittest.mock.patch("subprocess.run", return_value=process):
        assert indexed_groups.get_generators_with_gap(ORDER, INDEX) is None

    # GAP is not installed
    process_1 = subprocess.CompletedProcess(args=[], returncode=0, stdout="\nGAP 4")
    process_2 = subprocess.CompletedProcess(args=[], returncode=0, stdout="\n")
    with (
        pytest.raises(ValueError, match="Cannot extract cycle"),
        unittest.mock.patch("subprocess.run", side_effect=[process_1, process_2]),
    ):
        indexed_groups.get_generators_with_gap(ORDER, INDEX)

    # everything works as expected
    process_2 = subprocess.CompletedProcess(args=[], returncode=0, stdout="(1, 2)\n")
    with unittest.mock.patch("subprocess.run", side_effect=[process_1, process_2]):
        assert indexed_groups.get_generators_with_gap(ORDER, INDEX) == GENERATORS


def test_get_generators() -> None:
    """Retrieve generators somehow."""

    # retrieve from cache
    mock_cache = {(ORDER, INDEX): GENERATORS}
    with unittest.mock.patch("diskcache.Cache", return_value=mock_cache):
        assert indexed_groups.get_generators(ORDER, INDEX) == GENERATORS

    # retrieve from GAP
    with (
        unittest.mock.patch("diskcache.Cache", return_value={}),
        unittest.mock.patch(
            "qldpc.indexed_groups.get_generators_with_gap", return_value=GENERATORS
        ),
    ):
        assert indexed_groups.get_generators(ORDER, INDEX) == GENERATORS

    # retrieve from GroupNames.org
    with (
        unittest.mock.patch("diskcache.Cache", return_value={}),
        unittest.mock.patch("qldpc.indexed_groups.get_generators_with_gap", return_value=None),
        unittest.mock.patch(
            "qldpc.indexed_groups.get_generators_from_groupnames", return_value=GENERATORS
        ),
    ):
        assert indexed_groups.get_generators(ORDER, INDEX) == GENERATORS

    # fail to retrieve from anywhere :(
    with (
        pytest.raises(ValueError, match="Cannot build GAP group"),
        unittest.mock.patch("diskcache.Cache", return_value={}),
        unittest.mock.patch("qldpc.indexed_groups.get_generators_with_gap", return_value=None),
        unittest.mock.patch(
            "qldpc.indexed_groups.get_generators_from_groupnames", return_value=None
        ),
    ):
        indexed_groups.get_generators(ORDER, INDEX)
