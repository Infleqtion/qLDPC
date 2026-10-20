# SPDX-License-Identifier: Apache-2.0

"""Tests for the information-set families and certificates shared by BZ distance and counting."""

from __future__ import annotations

import ast
import itertools
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest

import qldpc

from . import _binary as binary
from . import _information_sets
from . import brouwer_zimmermann as bz
from . import logical_counts as lc

STAGES = ("_get_disjoint_family", "_prune_family", "_get_overlapping_family")


def test_binary_row_reduction_and_information_sets() -> None:
    """Binary elimination transforms quotient labels and retains the residual rank."""
    matrix = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    labels = np.eye(2, dtype=np.uint8)
    reduced, reduced_labels, pivots = binary._row_reduce_binary(matrix, labels)
    np.testing.assert_array_equal(reduced, [[1, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(reduced_labels, [[1, 1], [1, 0]])
    np.testing.assert_array_equal(pivots, [0, 1])

    basis = np.hstack(
        [
            np.eye(3, dtype=np.uint8),
            np.eye(3, dtype=np.uint8),
            np.array([[1, 0], [0, 1], [0, 0]], dtype=np.uint8),
        ]
    )
    information_sets = _information_sets._get_information_set_generators(
        basis, np.eye(3, dtype=np.uint8)
    )
    assert [rank for _, _, rank, _ in information_sets] == [3, 3, 2]
    assert not _information_sets._get_information_set_generators(
        np.zeros((1, 3), dtype=np.uint8), None
    )

    basis = np.array(
        [
            [1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    information_sets = _information_sets._get_information_set_generators(basis, None)
    assert [rank for _, _, rank, _ in information_sets] == [3, 3]

    # Only one column carries the second row, so no column order packs a second full set.
    basis = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
    information_sets = _information_sets._get_information_set_generators(basis, None)
    assert [rank for _, _, rank, _ in information_sets] == [2, 1]


def test_certificate_lower_bound_and_work() -> None:
    """Certificates combine rank deficiency, coverage, and proven divisors, as the guide derives."""
    certificate = _information_sets._Certificate(10, (10, 6, 2))
    # A rank-r_i block contributes max(0, t + 1 - (r - r_i)) after layer t.
    assert [certificate.lower_bound(t) for t in (0, 1, 5, 8)] == [1, 2, 8, 15]

    # Three full QC64 bases with coverage two certify ceil(3 (t + 1) / 2), rounded to even weights.
    qc64 = _information_sets._Certificate(41, (41,) * 3, weight_divisor=2, max_coverage=2)
    assert [qc64.lower_bound(t) for t in (2, 3, 4)] == [6, 6, 8]
    assert [_information_sets._Certificate(41, (41,) * 3, max_coverage=2).lower_bound(4)] == [8]
    assert qc64.work(8) == 3 * sum(math.comb(41, weight) for weight in range(1, 5))

    # An unreachable target counts every layer of every set.
    assert _information_sets._Certificate(3, (3, 3)).work(100) == 2 * 7
    assert _information_sets._Certificate(5, (5,), weight_divisor=2).lower_bound(2) == 4


def test_family_records_actual_coverage() -> None:
    """Families certify their own ranks and never underestimate pivot coverage."""
    basis = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    first = _information_sets._InformationSet(basis, None, 2, np.array([0, 1]))
    generators, labels, pivots = binary._row_reduce_binary(basis, columns=np.array([0, 2]))
    second = _information_sets._InformationSet(generators, labels, 2, pivots)
    family = _information_sets._Family.from_information_sets(
        [first, second], dimension=2, weight_divisor=1
    )
    assert family.certificate == _information_sets._Certificate(2, (2, 2), 1, 2)
    empty = _information_sets._Family.from_information_sets([], dimension=2, weight_divisor=1)
    assert empty.certificate == _information_sets._Certificate(2, ())

    with pytest.raises(AssertionError):
        _information_sets._Family((first,), _information_sets._Certificate(2, (1,)))
    with pytest.raises(AssertionError):
        _information_sets._Family((first, second), _information_sets._Certificate(2, (2, 2)))


def test_weight_divisor_inference() -> None:
    """Even ambient parity combines with a caller's divisor, and divides every spanned weight."""
    even = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.uint8)
    odd = np.array([[1, 0, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    assert _information_sets._infer_weight_divisor(even) == 2
    assert _information_sets._infer_weight_divisor(even, 2) == 2
    assert _information_sets._infer_weight_divisor(even, 3) == 6
    assert _information_sets._infer_weight_divisor(odd) == 1
    assert _information_sets._infer_weight_divisor(odd, 2) == 2
    for bits in itertools.product((0, 1), repeat=len(even)):
        assert np.count_nonzero(np.asarray(bits) @ even % 2) % 2 == 0
    assert _information_sets._get_disjoint_family(even, None).certificate.weight_divisor == 2
    assert (
        _information_sets._get_disjoint_family(
            odd, None, weight_divisor=2
        ).certificate.weight_divisor
        == 2
    )


@pytest.mark.parametrize("ranks, upper_bound", [([10, 6, 2], 6), ([10, 6, 2], 12), ([6, 2], 6)])
def test_prune_family(ranks: list[int], upper_bound: int) -> None:
    """Pruning partial search bases does not increase estimated enumeration work."""
    generators = np.empty((10, 0), dtype=np.uint8)
    pivots = np.empty(0, dtype=int)
    information_sets = tuple(
        _information_sets._InformationSet(generators, None, rank, pivots) for rank in ranks
    )
    family = _information_sets._Family(
        information_sets, _information_sets._Certificate(10, tuple(ranks))
    )
    pruned = _information_sets._prune_family(family, upper_bound=upper_bound)
    assert pruned.information_sets
    assert pruned.certificate.ranks == tuple(item.rank for item in pruned.information_sets)
    assert pruned.certificate.work(upper_bound) <= family.certificate.work(upper_bound)
    if 10 not in ranks:
        assert pruned is family


def test_overlapping_family_certificates() -> None:
    """Check pivot coverage and label transformations against exhaustive codeword weights."""

    # Label each row separately so its row operations can be checked.
    basis = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis = binary._get_independent_rows(basis)
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = _information_sets._prune_family(
        _information_sets._get_disjoint_family(basis, labels), upper_bound=3
    )
    assert _information_sets._get_overlapping_family(basis, labels, disjoint, upper_bound=3) is None
    family = _information_sets._get_overlapping_family(
        basis, labels, disjoint, upper_bound=3, minimum_work=0
    )
    assert family is not None
    coverage = family.certificate.max_coverage
    loads = np.bincount(
        np.concatenate([item.pivots for item in family.information_sets]),
        minlength=basis.shape[1],
    )
    assert coverage >= loads.max() > 1
    for item in family.information_sets:
        assert item.labels is not None
        np.testing.assert_array_equal(item.generators, item.labels @ basis % 2)
        np.testing.assert_array_equal(item.generators[:, item.pivots], np.eye(len(basis)))

    # Check the Hamming-weight bound for every word still unsearched in all bases.
    for bits in itertools.product((0, 1), repeat=len(basis)):
        word = np.asarray(bits) @ basis % 2
        for completed in (1, 2):
            if all(
                np.count_nonzero(word[item.pivots]) > completed for item in family.information_sets
            ):
                assert np.count_nonzero(word) >= family.certificate.lower_bound(completed)


def test_overlapping_family_guards() -> None:
    """Reject unhelpful coverage and terminate repeated attempts at the same pivot set."""

    # Nearly complete disjoint coverage leaves no cheaper family within the size budget.
    basis = np.hstack([np.eye(25, dtype=np.uint8)] * 4)
    shortened = basis[:, :-1]
    disjoint = _information_sets._get_disjoint_family(shortened, None)
    assert (
        _information_sets._get_overlapping_family(shortened, None, disjoint, upper_bound=26) is None
    )
    # The three disjoint full sets already reach a smaller family-size limit.
    assert (
        _information_sets._get_overlapping_family(
            shortened, None, disjoint, upper_bound=26, max_sets=3, minimum_work=0
        )
        is None
    )

    # Repeatedly rediscovering the same pivots must terminate without selecting a family.
    basis = basis[:, :40]
    first = _information_sets._get_disjoint_family(basis, None).information_sets[0]
    family = _information_sets._Family.from_information_sets(
        [first], dimension=25, weight_divisor=1
    )
    with mock.patch.object(
        _information_sets,
        "_row_reduce_binary",
        return_value=(first.generators, first.labels, first.pivots),
    ):
        assert _information_sets._get_overlapping_family(basis, None, family, upper_bound=8) is None


@pytest.mark.usefixtures("force_information_sets")
def test_distance_and_counting_share_family_stages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both consumers search the families built by the shared stages, at their own targets.

    Distance aims each stage at its witness weight ``U`` and stops at ``L >= U``.  Counting aims
    each stage at ``w + 1`` and stops only at ``L > w``, so it continues past ``L == w``.
    """
    rng = np.random.default_rng(40)
    checks_x = rng.integers(0, 2, (22, 40), dtype=np.uint8)
    kernel = binary._get_kernel(checks_x)
    checks_z = rng.integers(0, 2, (18, len(kernel))) @ kernel % 2
    code = qldpc.codes.CSSCode(checks_x.astype(int), checks_z.astype(int))
    expected = lc.CSSLogicalCounter(code).count_logicals(method="brute_force")

    events: list[tuple[Any, ...]] = []
    consumer = ""

    def record_stage(name: str) -> Callable[..., Any]:
        original = getattr(_information_sets, name)

        def stage(*args: Any, **kwargs: Any) -> Any:
            if name == "_get_overlapping_family":
                # Lower only the work threshold, so that both consumers search overlapping families.
                kwargs["minimum_work"] = 0
            result = original(*args, **kwargs)
            events.append((consumer, name, kwargs.get("upper_bound"), result))
            return result

        return stage

    def record_consumer(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def run(*args: Any, **kwargs: Any) -> Any:
            nonlocal consumer
            consumer = name
            try:
                return original(*args, **kwargs)
            finally:
                consumer = ""

        return run

    original_prepare = bz._prepare_brouwer_zimmermann_search
    original_family = lc._get_counting_information_sets
    original_bound = _information_sets._Certificate.lower_bound

    def prepare(*args: Any, **kwargs: Any) -> Any:
        result = original_prepare(*args, **kwargs)
        events.append((consumer, "search", result))
        return result

    def counting_family(*args: Any, **kwargs: Any) -> Any:
        sets, certificate = original_family(*args, **kwargs)
        events.append((consumer, "certificate", certificate))
        return sets, certificate

    def lower_bound(certificate: _information_sets._Certificate, completed_weight: int) -> int:
        value = original_bound(certificate, completed_weight)
        events.append((consumer, "bound", certificate, completed_weight, value))
        return value

    for name in STAGES:
        monkeypatch.setattr(_information_sets, name, record_stage(name))
    monkeypatch.setattr(_information_sets._Certificate, "lower_bound", lower_bound)
    monkeypatch.setattr(bz, "_prepare_brouwer_zimmermann_search", prepare)
    monkeypatch.setattr(lc, "_get_counting_information_sets", counting_family)
    monkeypatch.setattr(
        lc,
        "_get_distance_brouwer_zimmermann_many",
        record_consumer("distance", lc._get_distance_brouwer_zimmermann_many),
    )
    monkeypatch.setattr(lc, "_count_words", record_consumer("count", lc._count_words))

    # One public call finds the distance of both sectors, then counts both sectors at that weight.
    counts = lc.CSSLogicalCounter(code).count_logicals()
    assert counts == expected
    distance = counts.weight
    assert distance is not None

    def get_searched_families(name: str) -> list[_information_sets._Family]:
        """The last family that the stages produced for each sector of one consumer."""
        searched: list[_information_sets._Family] = []
        for event in events:
            if event[0] == name and event[1] == "_get_disjoint_family":
                searched.append(event[3])
            elif event[0] == name and event[1] in STAGES and event[3] is not None:
                searched[-1] = event[3]
        return searched

    # Distance searches the shared certificates at its witness weight, and stops at equality.
    distance_families = get_searched_families("distance")
    searches = [event[2] for event in events if event[1] == "search"]
    assert len(searches) == len(distance_families) == 2
    assert all(
        search.certificate is family.certificate
        for search, family in zip(searches, distance_families, strict=True)
    )
    assert any(family.certificate.max_coverage > 1 for family in distance_families)
    assert {event[2] for event in events if event[0] == "distance" and event[1] in STAGES[1:]} == {
        distance
    }
    assert any(search.lower_bound == distance for search in searches)
    assert all(search.finished for search in searches)

    # Counting searches the shared certificates aimed at w + 1, and stops only once L > w.
    counting_families = get_searched_families("count")
    certificates = [event[2] for event in events if event[1] == "certificate"]
    assert [family.certificate for family in counting_families] == certificates
    assert all(
        certificate is family.certificate
        for certificate, family in zip(certificates, counting_families, strict=True)
    )
    assert any(family.certificate.max_coverage > 1 for family in counting_families)
    assert {event[2] for event in events if event[0] == "count" and event[1] in STAGES[1:]} == {
        distance + 1
    }
    for certificate in certificates:
        start = next(
            index
            for index, event in enumerate(events)
            if event[1] == "certificate" and event[2] is certificate
        )
        stops = [
            event[4]
            for event in events[start:]
            if event[0] == "count" and event[1] == "bound" and event[2] is certificate
        ]
        assert stops[-1] > distance >= max(stops[:-1])
        if certificate.max_coverage > 1:
            assert distance in stops[:-1]


def test_consumers_use_the_neutral_family_layer() -> None:
    """Consumers reach shared stages through the module and do not redefine family policy."""
    shared = {
        name
        for name, value in vars(_information_sets).items()
        if getattr(value, "__module__", None) == _information_sets.__name__
    }
    assert set(STAGES) | {"_prepare_family", "_Certificate", "_Family"} <= shared
    for module in (bz, lc):
        tree = ast.parse(Path(module.__file__ or "").read_text())
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "_information_sets"
            for alias in node.names
        }
        assert not defined & shared
        # Names imported from the module would escape replacements of the module's bindings.
        assert not imported
        assert vars(module)["_information_sets"] is _information_sets
