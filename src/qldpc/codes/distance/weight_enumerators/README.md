# Navigating `weight_enumerators`

This directory counts the minimum-weight logical operators of a binary CSS stabilizer code, exactly.
It is a guide for contributors and agents who need to find their way around the code, change it safely, or decide whether to trust a result.
The exact signatures and per-function details are in the docstrings, which this guide points to rather than repeats.

## 1. What it computes

For a binary CSS stabilizer code, the public entry point [`CSSLogicalWeightCounter`](counter.py) returns two kinds of exact results:

- **Weight spectra**, from `get_spectrum`: the number of X-only and Z-only nontrivial logical words of each weight.
- **Minimum multiplicities**, from `get_minimum`: the distance, the number of X-only and Z-only logical words of that weight, and (by default) the number of physical Pauli operators of that weight, including mixed X/Y/Z and Y-only ones.

```python
from qldpc import codes

code = codes.SteaneCode()
counter = codes.CSSLogicalWeightCounter(code.matrix_x, code.matrix_z)
counter.get_minimum()  # CSSMinimumLogicalWeights(distance=3, count_x=7, count_z=7, count_pauli=21)
counter.conjugated_cnot(0, 1).get_minimum()  # a new counter for the CNOT-conjugated code
```

Three conventions are easy to get wrong:

- **Physical representatives.**
  Counts are of physical operators ignoring phase, not of logical equivalence classes modulo stabilizers.
- **Global minimum.**
  `count_x` and `count_z` count words at the *global* minimum weight, not at each sector's own minimum, so one of them can be zero.
- **Dimension zero.**
  A code with no logical qubits has distance `None` and zero counts.

The scope is deliberately narrow: binary CSS stabilizer codes only.
The constructor rejects non-commuting checks, so subsystem codes and nonbinary codes are not supported.
Counting is exponential in the worst case, and the counters raise an error rather than return an approximate or partial count.

## 2. The files at a glance

| File | Responsibility | Entry points | Tests |
| --- | --- | --- | --- |
| [counter.py](counter.py) | The public API: result dataclasses, `CSSLogicalWeightCounter`, method dispatch, cached bases, Clifford conjugation | `get_minimum`, `get_spectrum`, `conjugated`, `get_minimum_logical_weights_css` | [counter_test.py](counter_test.py) |
| [dual.py](dual.py) | The `"dual"` method: stabilizer weight histograms and the exact MacWilliams transform | `get_stabilizer_histograms`, `get_macwilliams_spectrum`, `get_macwilliams_minimum` | [dual_test.py](dual_test.py) |
| [supports.py](supports.py) | Supports of minimum-weight words by Brouwer-Zimmermann (BZ) or exhaustive search, the packed-row layout, the storage budget, and the lift to full Pauli counts | `get_bz_supports`, `get_exhaustive_supports`, `get_exhaustive_minimum`, `count_minimum_paulis`, `StorageBudget` | [supports_test.py](supports_test.py) |
| [collision.py](collision.py) | The opt-in `"collision"` method: physical half-shell joins that find supports without a precomputed distance | `recover_supports` | [collision_test.py](collision_test.py) |
| [conftest.py](conftest.py) | Test helpers shared by the four test files | `_native`, `_get_sector_bases` | |

Each implementation module has a matching test file.
The package also uses two modules one directory up:

- [../brouwer_zimmermann.py](../brouwer_zimmermann.py) provides the information-set family selection and the lower certificate that `supports.py` reuses.
  Its algorithm is explained in [../brouwer_zimmermann.md](../brouwer_zimmermann.md).
- [../_binary.py](../_binary.py) provides binary row reduction, nested bases, and the native `popcount_word` and `trailing_zeros` primitives.

## 3. How a call flows

`get_minimum` works in two phases.
Phase 1 finds the distance and the X-only and Z-only counts.
Phase 2 turns those into a full Pauli count, and is skipped for `full_pauli=False` and for self-dual codes.

```text
get_minimum(method, full_pauli)
 |
 |-- Phase 1: _find_minimum(method)            "auto" first picks "dual" or "brouwer_zimmermann"
 |     |-- "dual"                 dual.py: histograms -> MacWilliams.   Counts only, no supports.
 |     |-- "brouwer_zimmermann"   bz distance search, then supports.get_bz_supports per sector.
 |     |-- "brute_force"          supports.get_exhaustive_minimum per sector.
 |     `-- "collision"            collision.recover_supports at increasing weights.
 |
 `-- Phase 2: only if full_pauli
       |-- self-dual code (X and Z row spaces equal): count_pauli = 3 * count_x
       `-- otherwise: supports.count_minimum_paulis(supports_x, supports_z, ...)
             (the "dual" method has no supports yet, so _get_supports_at recovers them first)
```

A few facts that help when reading `counter.py`:

- **Sectors are nested bases.**
  For the X sector, `_get_sector_bases` builds a basis of the X normalizer `ker(H_Z)` in which the first rows span the X stabilizers, together with quotient labels.
  A combination of basis rows is a nontrivial logical exactly when its label is nonzero.
  This is the same quotient-label idea as in [Section 3.4 of the BZ guide](../brouwer_zimmermann.md#34-quotient-labels-for-the-stabilizer-test).
- **Bases are cached, scores are not.**
  `conjugated` returns a new counter and transports the cached bases through the local Clifford.
  Minimum scores and histograms are recomputed on every call.
- **`auto` never picks `collision`.**
  It chooses between `dual` and `brouwer_zimmermann` using `_choose_method`, which estimates work from the stabilizer ranks.
- **Rank limits.**
  The `dual` method enumerates `2**rank` stabilizer words, and the exhaustive routes enumerate `2**size` combinations of a normalizer basis.
  Both raise `OverflowError` above 62, because a native histogram or loop counter could not represent more.

## 4. Concepts and conventions

- **Support.**
  A binary word is determined by its support, so the library stores and compares supports as packed rows.
  Counting distinct minimum words is counting distinct supports.
- **Packed layout.**
  Physical coordinate `j` is bit `j % 64` of word `j // 64`, so coordinate zero is the low bit.
  This is deliberately different from `_rows_to_ints` in `_binary.py`.
  Do not pass one layout to code that expects the other.
- **Strict certificate.**
  BZ certifies that every word it has not yet examined has weight at least `L(t)`.
  Computing a distance may stop at `L(t) >= U`, but counting must also have examined every word of weight exactly `d`, so `get_bz_supports` stops only at `L(t) > d`.
  The `supports.py` module docstring and the correctness requirements in the BZ guide state this contract.
- **Lifting to Pauli operators.**
  Every minimum Pauli operator has a minimum pure support in at least one sector.
  A support shared by both sectors gives exactly three operators (X, Z and Y).
  A support found in one sector is dressed by the opposite-type stabilizers contained in it, which costs one rank query.
  The derivation is in the docstring of `count_minimum_paulis`.
- **Exact arithmetic.**
  Final counts are Python integers.
  Native code uses int64 only where the partial sums are proved to fit (see `_get_safe_coefficients` in `dual.py`), and the lifting exponentiates dressing counts in Python.
- **Resumable native work.**
  Native scans and collision queries return to Python after a bounded amount of work, which lets the interpreter be interrupted.
  Any helper that you extract from them must preserve this.

## 5. Resource limits

`memory_limit_bytes` and `batch_size` are constructor arguments of the counter, and every backend receives them.
Every large allocation is checked against the budget first, and a failure raises `MemoryError` or `OverflowError`, never a partial count.

The budget uses named, conservative multiples defined next to `StorageBudget` in `supports.py`, such as the number of simultaneous copies of the support array.
It governs the logical bases, support enumeration, Pauli lifting and the collision tables.
It does **not** govern the BZ distance search that runs before support enumeration in the `brouwer_zimmermann` method, or the small tables of the `dual` method.
The counter's docstring says the same.

## 6. Where to look

| If you want to... | Start at |
| --- | --- |
| Change which method `auto` picks | `_choose_method` in `counter.py` |
| Add a counting backend | A new `_find_minimum_*` method in `counter.py`, plus an entry in `WeightEnumeratorMethod` and `get_minimum`'s validation |
| Change the BZ stopping rule or family selection | `get_bz_supports` and `_choose_information_sets` in `supports.py`, and read the BZ guide's correctness requirements first |
| Understand a wrong full-Pauli count | `count_minimum_paulis` and `_shortened_histogram` in `supports.py` |
| Understand a wrong spectrum | `get_macwilliams_spectrum` and `_iter_krawtchouk` in `dual.py` |
| Debug a collision miscount | `_build_chunk` and `_query_chunk` in `collision.py`, then the module docstring's ownership argument |
| Change memory accounting | The constants beside `StorageBudget` in `supports.py`, and `_plan_storage` in `collision.py` |
| Change how local Cliffords act | `conjugated` and `transform_rows` in `counter.py` |

## 7. Source conventions

Sections are introduced by a line of 100 `#` characters and a one-line title.
Public entry points come first, helper methods next, and native kernels at the bottom.
Comments before a block of code state the invariant or purpose of that phase.

## 8. Building confidence in a change

The tests are built around independent oracles, not around the code's own outputs.

- **Explicit enumeration.**
  `_oracle` in `counter_test.py` enumerates the whole normalizer in plain Python integers.
  The tests compare all five methods against it on random codes, sparse codes, and codes with redundant or zero rows.
- **Cross-method agreement.**
  `dual`, `brouwer_zimmermann`, `brute_force` and `collision` are four independent algorithms, so agreement between them is strong evidence.
- **Native parity.**
  Many tests take an `interpreted` parameter and run the Python implementation of a kernel (through `_native` in `conftest.py`), and the single-word fast paths are compared with the multiword paths on identical data.
- **Resource checks.**
  `test_storage_budget_bounds_measured_allocations` measures real allocations with `tracemalloc`, and the collision tests check each preflight before it allocates.

When you change an algorithm, check that the tests would notice a mistake, rather than only that they pass.
Temporarily break the invariant (for example, change `lower > weight` to `lower >= weight` in `get_bz_supports`), confirm that a test fails, and revert.
The invariants most worth breaking this way are:

| Invariant | Where | A test that catches a violation |
| --- | --- | --- |
| Strict `L(t) > d` certificate | `get_bz_supports` | `test_known_codes_and_strict_bz_boundary` |
| Selector target `d + 1` | `_choose_information_sets` | `test_overlapping_supports_keep_strict_coverage_and_physical_ownership` |
| Actual maximum coverage in the bound | `get_bz_supports` | `test_connected_multiword_surface` |
| Physical deduplication across bases | `_unique_supports` | `test_known_codes_and_strict_bz_boundary` |
| Factor of three for shared supports | `get_minimum`, `count_minimum_paulis` | `test_random_css_oracles` |
| Opposite-type dressing | `count_minimum_paulis` | `test_shared_and_dressed_supports` |
| Counts refer to the global minimum | `_find_minimum_exhaustive` | `test_all_methods_match_explicit_enumeration_on_random_codes` |
| Unique ownership of each collision support | `_query_chunk` | `test_all_weights_and_degenerate_chains` |
| Basis transport under conjugation | `conjugated` | `test_multiword_vectors_and_labels` |
| Storage multipliers are not too small | `supports.py` constants | `test_storage_budget_bounds_measured_allocations` |

Useful commands, from the repository root:

```bash
python -m pytest src/qldpc/codes/distance/weight_enumerators -p no:randomly
python checks/coverage_.py src/qldpc/codes/distance/weight_enumerators/supports.py --jobs 1
python checks/coverage_.py src/qldpc/codes/distance/weight_enumerators/supports.py --jobs 1 --branch
```

Run the coverage check on one module at a time, using that module's own test file.
An aggregate run can hide cases that moved between files, and the `--branch` form also finds untaken arcs that statement coverage cannot.
Add a new implementation module together with a matching test file.

## 9. Known limits

- Binary CSS stabilizer codes only.
  Subsystem codes, general (non-CSS) stabilizer codes and nonbinary codes are not supported.
- Time and memory are exponential in the worst case.
  On codes where `collision` is a poor fit, it needs more memory than a modest budget allows and raises an error.
- No claim is made about performance on any particular processor architecture.
