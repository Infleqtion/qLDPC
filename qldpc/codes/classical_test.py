"""Unit tests for classical.py

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

import networkx as nx
import pytest

from qldpc import codes


@pytest.mark.parametrize("field", [2, 3])
def test_basic(field: int) -> None:
    """Repetition, ring, and Hamming codes."""
    num_bits = 4
    for code in [codes.RepetitionCode(num_bits, field), codes.RingCode(num_bits, field)]:
        assert code.num_bits == num_bits
        assert code.dimension == 1

    # that rank of repetition and Hamming codes is independent of the field
    assert codes.RepetitionCode(3, 2).rank == codes.RepetitionCode(3, 3).rank
    assert codes.HammingCode(3, 2).rank == codes.HammingCode(3, 3).rank


def test_special_codes() -> None:
    """Reed-Solomon, BCH, and Reed-Muller codes."""
    assert codes.ReedSolomonCode(3, 2).dimension == 2

    bits, dimension = 7, 4
    assert codes.BCHCode(bits, dimension).dimension == dimension
    with pytest.raises(ValueError, match=r"2\^m - 1 bits"):
        codes.BCHCode(bits - 1, dimension)

    order, size = 1, 3
    code = codes.ReedMullerCode(order, size)
    assert code.dimension == codes.ClassicalCode(code.matrix).dimension
    assert ~code == codes.ReedMullerCode(size - order - 1, size)

    with pytest.raises(ValueError, match="0 <= r <= m"):
        codes.ReedMullerCode(-1, 0)


def test_tanner_code() -> None:
    """Classical Tanner codes on random regular graphs."""
    subcode = codes.ClassicalCode.random(5, 3)
    subgraph = nx.random_regular_graph(subcode.num_bits, subcode.num_bits * 2 + 2)

    tag = "sort_label"
    for node_a, node_b in subgraph.edges:
        subgraph[node_a][node_b]["sort"] = {node_a: tag, node_b: tag}

    code = codes.TannerCode(subgraph, subcode)
    assert code.num_bits == subgraph.number_of_edges()
    assert code.num_checks == subgraph.number_of_nodes() * code.subcode.num_checks
    assert all(code.subgraph.get_edge_data(*edge)["sort"] == tag for edge in code.subgraph.edges)
