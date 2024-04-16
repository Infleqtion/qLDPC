"""Unit tests for groups.py

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

from qldpc import external

# strip cache wrappers
assert hasattr(external.groups.get_generators, "__wrapped__")
assert hasattr(external.groups.get_small_group_number, "__wrapped__")
external.groups.get_generators = external.groups.get_generators.__wrapped__
external.groups.get_small_group_number = external.groups.get_small_group_number.__wrapped__

# define global testing variables
ORDER, INDEX = 2, 1
GENERATORS = [[(0, 1)]]
GROUP = f"SmallGroup({ORDER},{INDEX})"
GROUP_URL = external.groups.GROUPNAMES_URL + "1/C2.html"
MOCK_INDEX_HTML = """<table class="gptable" columns="6" style='width: 70%;'>
<tr><th width="12%"></th><th width="60%"></th><th width="5%"><a href='T.html'>d</a></th><th width="5%"><a href='R.html'>&rho;</a></th><th width="12%">Label</th><th width="7%">ID</th></tr><tr><td id="c2"><a href="1/C2.html">C<sub>2</sub></a></td><td><a href="cyclic.html">Cyclic</a> group</td><td><a href="T15.html#c2">2</a></td><td><a href="R.html#dim1+">1+</a></td><td>C2</td><td>2,1</td></tr>
</table>"""  # pylint: disable=line-too-long  # noqa: E501
MOCK_GROUP_HTML = """<b><a href='https://en.wikipedia.org/wiki/Group actions' title='See wikipedia' class='wiki'>Permutation representations of C<sub>2</sub></a></b><br><a id='shl1' class='shl' href="javascript:showhide('shs1','shl1','Regular action on 2 points');"><span class="nsgpn">&#x25ba;</span>Regular action on 2 points</a> - transitive group <a href="../T15.html#2t1">2T1</a><div id='shs1' class='shs'>Generators in S<sub>2</sub><br><pre class='pre' id='textgn1'>(1 2)</pre>&emsp;<button class='copytext' id='copygn1'>Copy</button><br>"""  # pylint: disable=line-too-long  # noqa: E501


def get_mock_page(text: str) -> unittest.mock.MagicMock:
    """Fake webpage with the given text."""
    mock_page = unittest.mock.MagicMock()
    mock_page.read.return_value = text.encode("utf-8")
    return mock_page


def test_get_group_url() -> None:
    """Retrive url for group webpage on GroupNames.org."""

    # cannot connect to general webpage
    with unittest.mock.patch(
        "urllib.request.urlopen", side_effect=urllib.error.URLError("message")
    ):
        assert external.groups.get_group_url(ORDER, INDEX) is None

    # cannot find group in the index
    mock_page = get_mock_page(MOCK_INDEX_HTML.replace(f"{ORDER},{INDEX}", ""))
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="Group .* not found"),
    ):
        external.groups.get_group_url(ORDER, INDEX)

    # cannot find link to group webpage
    mock_page = get_mock_page(MOCK_INDEX_HTML.replace("href", ""))
    with (
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="Webpage .* not found"),
    ):
        external.groups.get_group_url(ORDER, INDEX)

    # everything works as expected
    mock_page = get_mock_page(MOCK_INDEX_HTML)
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        assert external.groups.get_group_url(ORDER, INDEX) == GROUP_URL


def test_get_generators_from_groupnames() -> None:
    """Retrive generators from group webpage on GroupNames.org."""

    # group not indexed
    assert external.groups.get_generators_from_groupnames("") is None

    # group url not found
    with unittest.mock.patch("qldpc.external.groups.get_group_url", return_value=None):
        assert external.groups.get_generators_from_groupnames(GROUP) is None

    # cannot find generators
    mock_page = get_mock_page(MOCK_GROUP_HTML.replace("pre", ""))
    with (
        unittest.mock.patch("qldpc.external.groups.get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
        pytest.raises(ValueError, match="Generators .* not found"),
    ):
        external.groups.get_generators_from_groupnames(GROUP)

    # everything works as expected
    mock_page = get_mock_page(MOCK_GROUP_HTML)
    with (
        unittest.mock.patch("qldpc.external.groups.get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        assert external.groups.get_generators_from_groupnames(GROUP) == GENERATORS


def get_mock_process(stdout: str) -> subprocess.CompletedProcess[str]:
    """Fake process with the given stdout."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def test_gap_is_installed() -> None:
    """Is GAP 4 installed?"""
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("")):
        assert not external.groups.gap_is_installed()
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("\nGAP 4")):
        assert external.groups.gap_is_installed()


def test_get_gap_result() -> None:
    """Run GAP commands and retrieve the GAP output."""
    output = "test"
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process(output)):
        assert external.groups.get_gap_result().stdout == output


def test_get_generators_with_gap() -> None:
    """Retrive generators from GAP 4."""

    # GAP is not installed
    with unittest.mock.patch("qldpc.external.groups.gap_is_installed", return_value=False):
        assert external.groups.get_generators_with_gap(GROUP) is None

    # cannot extract cycle from string
    mock_process = get_mock_process("\n(1, 2a)\n")
    with (
        unittest.mock.patch("qldpc.external.groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.groups.get_gap_result", return_value=mock_process),
        pytest.raises(ValueError, match="Cannot extract cycle"),
    ):
        assert external.groups.get_generators_with_gap(GROUP) is None

    # group not recognized by GAP
    mock_process = get_mock_process("")
    with (
        unittest.mock.patch("qldpc.external.groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.groups.get_gap_result", return_value=mock_process),
        pytest.raises(ValueError, match="not recognized by GAP"),
    ):
        assert external.groups.get_generators_with_gap(GROUP) is None

    # everything works as expected
    mock_process = get_mock_process("\n(1, 2)\n")
    with (
        unittest.mock.patch("qldpc.external.groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.groups.get_gap_result", return_value=mock_process),
    ):
        assert external.groups.get_generators_with_gap(GROUP) == GENERATORS


def test_get_generators() -> None:
    """Retrieve generators somehow."""

    # retrieve from GAP
    with (
        unittest.mock.patch(
            "qldpc.external.groups.get_generators_with_gap", return_value=GENERATORS
        ),
    ):
        assert external.groups.get_generators(GROUP) == GENERATORS

    # retrieve from GroupNames.org
    with (
        unittest.mock.patch("qldpc.external.groups.get_generators_with_gap", return_value=None),
        unittest.mock.patch(
            "qldpc.external.groups.get_generators_from_groupnames", return_value=GENERATORS
        ),
    ):
        assert external.groups.get_generators(GROUP) == GENERATORS

    # fail to retrieve from anywhere :(
    with (
        unittest.mock.patch("qldpc.external.groups.get_generators_with_gap", return_value=None),
        unittest.mock.patch(
            "qldpc.external.groups.get_generators_from_groupnames", return_value=None
        ),
    ):
        with pytest.raises(ValueError, match="Cannot build GAP group"):
            external.groups.get_generators(GROUP)
        with pytest.raises(ValueError, match="Cannot build GAP group"):
            external.groups.get_generators("CyclicGroup(2)")


def test_get_small_group_number() -> None:
    """Retrieve the number of groups of some order."""

    order, number = 16, 14
    text = rf"<td>{order},{number}</td>"

    # fail to determine group number
    with (
        unittest.mock.patch("qldpc.external.groups.maybe_get_webpage", return_value=None),
        unittest.mock.patch("qldpc.external.groups.gap_is_installed", return_value=False),
        pytest.raises(ValueError, match="Cannot determine"),
    ):
        external.groups.get_small_group_number(order)

    # retrieve from GAP
    mock_process = get_mock_process(str(number))
    with (
        unittest.mock.patch("qldpc.external.groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.groups.get_gap_result", return_value=mock_process),
    ):
        assert external.groups.get_small_group_number(order) == number

    # retrieve from GroupNames.org
    with (
        unittest.mock.patch("qldpc.external.groups.gap_is_installed", return_value=False),
        unittest.mock.patch("qldpc.external.groups.maybe_get_webpage", return_value=text),
    ):
        assert external.groups.get_small_group_number(order) == number


def test_get_small_group_structure() -> None:
    """Retrieve a description of the structure of a group."""
    order, index = 12, 3
    structure = "C3 : C4"

    # retrieve a structure from cache
    cache = {(order, index): structure}
    with unittest.mock.patch("qldpc.cache.get_disk_cache", return_value=cache):
        assert external.groups.get_small_group_structure(order, index) == structure

    # fail to retrieve structure from GAP
    process = get_mock_process("")
    with (
        unittest.mock.patch("qldpc.cache.get_disk_cache", return_value={}),
        unittest.mock.patch("qldpc.external.groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.groups.get_gap_result", return_value=process),
        pytest.raises(ValueError, match="Group not recognized"),
    ):
        external.groups.get_small_group_structure(order, index)

    # retrieve structure from GAP
    process = get_mock_process(structure)
    with (
        unittest.mock.patch("qldpc.cache.get_disk_cache", return_value={}),
        unittest.mock.patch("qldpc.external.groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.external.groups.get_gap_result", return_value=process),
    ):
        assert external.groups.get_small_group_structure(order, index) == structure

    # GAP is not installed
    with (
        unittest.mock.patch("qldpc.cache.get_disk_cache", return_value={}),
        unittest.mock.patch("qldpc.external.groups.gap_is_installed", return_value=False),
    ):
        structure = f"SmallGroup({order},{index})"
        assert external.groups.get_small_group_structure(order, index) == structure
