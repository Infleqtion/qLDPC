"""Unit tests for named_groups.py

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

from qldpc import named_groups

ORDER, INDEX = 2, 1
GENERATORS = [[(0, 1)]]
GROUP = f"SmallGroup({ORDER},{INDEX})"
GROUP_URL = named_groups.GROUPNAMES_URL + "1/C2.html"
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
        assert named_groups.get_group_url(ORDER, INDEX) is None

    # cannot find group in the index
    mock_page = get_mock_page(MOCK_INDEX_HTML.replace(f"{ORDER},{INDEX}", ""))
    with (
        pytest.raises(ValueError, match="Group .* not found"),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        named_groups.get_group_url(ORDER, INDEX)

    # cannot find link to group webpage
    mock_page = get_mock_page(MOCK_INDEX_HTML.replace("href", ""))
    with (
        pytest.raises(ValueError, match="Webpage .* not found"),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        named_groups.get_group_url(ORDER, INDEX)

    # everything works as expected
    mock_page = get_mock_page(MOCK_INDEX_HTML)
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        assert named_groups.get_group_url(ORDER, INDEX) == GROUP_URL


def test_get_generators_from_groupnames() -> None:
    """Retrive generators from group webpage on GroupNames.org."""

    # group not indexed
    assert named_groups.get_generators_from_groupnames("") is None

    # group url not found
    with unittest.mock.patch("qldpc.named_groups.get_group_url", return_value=None):
        assert named_groups.get_generators_from_groupnames(GROUP) is None

    # cannot find generators
    mock_page = get_mock_page(MOCK_GROUP_HTML.replace("pre", ""))
    with (
        pytest.raises(ValueError, match="Generators .* not found"),
        unittest.mock.patch("qldpc.named_groups.get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        named_groups.get_generators_from_groupnames(GROUP)

    # everything works as expected
    mock_page = get_mock_page(MOCK_GROUP_HTML)
    with (
        unittest.mock.patch("qldpc.named_groups.get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        assert named_groups.get_generators_from_groupnames(GROUP) == GENERATORS


def get_mock_process(stdout: str) -> subprocess.CompletedProcess[str]:
    """Fake process with the given stdout."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def test_gap_is_installed() -> None:
    """Is GAP 4 installed?"""
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("")):
        assert not named_groups.gap_is_installed()
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("\nGAP 4")):
        assert named_groups.gap_is_installed()


def test_get_gap_result() -> None:
    """Run GAP commands and retrieve the GAP output."""
    output = "test"
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process(output)):
        assert named_groups.get_gap_result().stdout == output


def test_get_generators_with_gap() -> None:
    """Retrive generators from GAP 4."""

    # GAP is not installed
    with unittest.mock.patch("qldpc.named_groups.gap_is_installed", return_value=False):
        assert named_groups.get_generators_with_gap(GROUP) is None

    # cannot extract cycle from string
    mock_process = get_mock_process("\n(1, 2a)\n")
    with (
        pytest.raises(ValueError, match="Cannot extract cycle"),
        unittest.mock.patch("qldpc.named_groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.named_groups.get_gap_result", return_value=mock_process),
    ):
        assert named_groups.get_generators_with_gap(GROUP) is None

    # group not recognized by GAP
    mock_process = get_mock_process("")
    with (
        pytest.raises(ValueError, match="not recognized by GAP"),
        unittest.mock.patch("qldpc.named_groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.named_groups.get_gap_result", return_value=mock_process),
    ):
        assert named_groups.get_generators_with_gap(GROUP) is None

    # everything works as expected
    mock_process = get_mock_process("\n(1, 2)\n")
    with (
        unittest.mock.patch("qldpc.named_groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.named_groups.get_gap_result", return_value=mock_process),
    ):
        assert named_groups.get_generators_with_gap(GROUP) == GENERATORS


def test_get_generators() -> None:
    """Retrieve generators somehow."""

    # use cache to save/retrieve results
    with unittest.mock.patch("diskcache.Cache", return_value={}):
        # compute and save result to cache
        with unittest.mock.patch(
            "qldpc.named_groups.get_generators_with_gap", return_value=GENERATORS
        ):
            assert named_groups.get_generators(GROUP) == GENERATORS

        # retrieve result from cache
        assert named_groups.get_generators(GROUP) == GENERATORS

    # strip cache wrapper
    if hasattr(named_groups.get_generators, "__wrapped__"):
        named_groups.get_generators = named_groups.get_generators.__wrapped__

    # retrieve from GAP
    with (
        unittest.mock.patch("qldpc.named_groups.get_generators_with_gap", return_value=GENERATORS),
    ):
        assert named_groups.get_generators(GROUP) == GENERATORS

    # retrieve from GroupNames.org
    with (
        unittest.mock.patch("qldpc.named_groups.get_generators_with_gap", return_value=None),
        unittest.mock.patch(
            "qldpc.named_groups.get_generators_from_groupnames", return_value=GENERATORS
        ),
    ):
        assert named_groups.get_generators(GROUP) == GENERATORS

    # fail to retrieve from anywhere :(
    with (
        unittest.mock.patch("qldpc.named_groups.get_generators_with_gap", return_value=None),
        unittest.mock.patch("qldpc.named_groups.get_generators_from_groupnames", return_value=None),
    ):
        with pytest.raises(ValueError, match="Cannot build GAP group"):
            named_groups.get_generators(GROUP)
        with pytest.raises(ValueError, match="Cannot build GAP group"):
            named_groups.get_generators("CyclicGroup(2)")


def test_get_small_group_number() -> None:
    """Retrieve the number of groups of some order."""
    # strip cache wrapper
    if hasattr(named_groups.get_small_group_number, "__wrapped__"):
        named_groups.get_small_group_number = named_groups.get_small_group_number.__wrapped__

    order, number = 16, 14
    text = rf"<td>{order},{number}</td>"

    # fail to determine group number
    with (
        pytest.raises(ValueError, match="Cannot determine"),
        unittest.mock.patch("qldpc.named_groups.maybe_get_webpage", return_value=None),
        unittest.mock.patch("qldpc.named_groups.gap_is_installed", return_value=False),
    ):
        named_groups.get_small_group_number(order)

    # retrieve from GAP
    mock_process = get_mock_process(str(number))
    with (
        unittest.mock.patch("qldpc.named_groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.named_groups.get_gap_result", return_value=mock_process),
    ):
        assert named_groups.get_small_group_number(order) == number

    # retrieve from GroupNames.org
    with (
        unittest.mock.patch("qldpc.named_groups.gap_is_installed", return_value=False),
        unittest.mock.patch("qldpc.named_groups.maybe_get_webpage", return_value=text),
    ):
        assert named_groups.get_small_group_number(order) == number
