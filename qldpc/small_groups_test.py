"""Unit tests for small_groups.py

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

from qldpc import small_groups

ORDER, INDEX = 2, 1
GENERATORS = [[(0, 1)]]
GROUP_URL = small_groups.GROUPNAMES_URL + "1/C2.html"
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
        assert small_groups.get_group_url(ORDER, INDEX) is None

    # cannot find group in the index
    mock_page = get_mock_page(MOCK_INDEX_HTML.replace(f"{ORDER},{INDEX}", ""))
    with (
        pytest.raises(ValueError, match="Group .* not found"),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        small_groups.get_group_url(ORDER, INDEX)

    # cannot find link to group webpage
    mock_page = get_mock_page(MOCK_INDEX_HTML.replace("href", ""))
    with (
        pytest.raises(ValueError, match="Webpage .* not found"),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        small_groups.get_group_url(ORDER, INDEX)

    # everything works as expected
    mock_page = get_mock_page(MOCK_INDEX_HTML)
    with unittest.mock.patch("urllib.request.urlopen", return_value=mock_page):
        assert small_groups.get_group_url(ORDER, INDEX) == GROUP_URL


def test_get_generators_from_groupnames() -> None:
    """Retrive generators from group webpage on GroupNames.org."""

    # group url not found
    with unittest.mock.patch("qldpc.small_groups.get_group_url", return_value=None):
        assert small_groups.get_generators_from_groupnames(ORDER, INDEX) is None

    # cannot find generators
    mock_page = get_mock_page(MOCK_GROUP_HTML.replace("pre", ""))
    with (
        pytest.raises(ValueError, match="Generators .* not found"),
        unittest.mock.patch("qldpc.small_groups.get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        small_groups.get_generators_from_groupnames(ORDER, INDEX)

    # everything works as expected
    mock_page = get_mock_page(MOCK_GROUP_HTML)
    with (
        unittest.mock.patch("qldpc.small_groups.get_group_url", return_value=GROUP_URL),
        unittest.mock.patch("urllib.request.urlopen", return_value=mock_page),
    ):
        assert small_groups.get_generators_from_groupnames(ORDER, INDEX) == GENERATORS


def get_mock_process(stdout: str) -> subprocess.CompletedProcess[str]:
    """Fake process with the given stdout."""
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout)


def test_gap_is_installed() -> None:
    """Is GAP 4 installed?"""
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("")):
        assert not small_groups.gap_is_installed()
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process("\nGAP 4")):
        assert small_groups.gap_is_installed()


def test_get_gap_result() -> None:
    """Run GAP commands and retrieve the GAP output."""
    output = "test"
    with unittest.mock.patch("subprocess.run", return_value=get_mock_process(output)):
        assert small_groups.get_gap_result([]).stdout == output


def test_get_generators_with_gap() -> None:
    """Retrive generators from GAP 4."""

    # GAP is not installed
    with unittest.mock.patch("qldpc.small_groups.gap_is_installed", return_value=False):
        assert small_groups.get_generators_with_gap(ORDER, INDEX) is None

    # cannot extract cycle from string
    mock_process = get_mock_process("\n(1, 2a)\n")
    with (
        pytest.raises(ValueError, match="Cannot extract cycle"),
        unittest.mock.patch("qldpc.small_groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.small_groups.get_gap_result", return_value=mock_process),
    ):
        assert small_groups.get_generators_with_gap(ORDER, INDEX) is None

    # group not recognized by GAP
    mock_process = get_mock_process("")
    with (
        pytest.raises(ValueError, match="not recognized by GAP"),
        unittest.mock.patch("qldpc.small_groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.small_groups.get_gap_result", return_value=mock_process),
    ):
        assert small_groups.get_generators_with_gap(ORDER, INDEX) is None

    # everything works as expected
    mock_process = get_mock_process("\n(1, 2)\n")
    with (
        unittest.mock.patch("qldpc.small_groups.gap_is_installed", return_value=True),
        unittest.mock.patch("qldpc.small_groups.get_gap_result", return_value=mock_process),
    ):
        assert small_groups.get_generators_with_gap(ORDER, INDEX) == GENERATORS


def test_get_generators() -> None:
    """Retrieve generators somehow."""

    # use cache to save/retrieve results
    with unittest.mock.patch("diskcache.Cache", return_value={}):
        # compute and save result to cache
        with unittest.mock.patch(
            "qldpc.small_groups.get_generators_with_gap", return_value=GENERATORS
        ):
            assert small_groups.get_generators(ORDER, INDEX) == GENERATORS

        # retrieve result from cache
        assert small_groups.get_generators(ORDER, INDEX) == GENERATORS

    # strip cache wrapper
    if hasattr(small_groups.get_generators, "__wrapped__"):
        small_groups.get_generators = small_groups.get_generators.__wrapped__

    # retrieve from GAP
    with (
        unittest.mock.patch("qldpc.small_groups.get_generators_with_gap", return_value=GENERATORS),
    ):
        assert small_groups.get_generators(ORDER, INDEX) == GENERATORS

    # retrieve from GroupNames.org
    with (
        unittest.mock.patch("qldpc.small_groups.get_generators_with_gap", return_value=None),
        unittest.mock.patch(
            "qldpc.small_groups.get_generators_from_groupnames", return_value=GENERATORS
        ),
    ):
        assert small_groups.get_generators(ORDER, INDEX) == GENERATORS

    # fail to retrieve from anywhere :(
    with (
        pytest.raises(ValueError, match="Cannot build GAP group"),
        unittest.mock.patch("qldpc.small_groups.get_generators_with_gap", return_value=None),
        unittest.mock.patch("qldpc.small_groups.get_generators_from_groupnames", return_value=None),
    ):
        small_groups.get_generators(ORDER, INDEX)
