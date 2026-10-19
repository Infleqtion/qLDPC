# Brouwer-Zimmermann distance searches in qLDPC

The classical disjoint/rank-deficient certificates follow [Algorithm 994](https://arxiv.org/abs/1603.06757).
The symplectic-to-Hamming reduction follows its [quantum-code adaptation](https://arxiv.org/abs/2408.10743).
The coverage-divided overlapping-set certificate is derived here for qLDPC, rather than reproduced from either paper.

[core.py](core.py) provides the public binary-distance interfaces and dispatches to [brouwer_zimmermann.py](brouwer_zimmermann.py) or [brute_force.py](brute_force.py).
This guide derives the stopping certificates and explains how the implementation uses them, including quotient labels, overlapping information sets, and packed enumeration.

## 1. Find a witness, then rule out lighter words

A **witness** is an eligible word whose weight has been computed.
Eligibility means nonzero for a classical code, or logically nontrivial for a quantum code.
A witness of weight `U` establishes that distance is at most `U`; it does not establish that no lighter eligible word exists.
Brouwer-Zimmermann (BZ) searches combinations of generator rows and derives a lower bound on eligible words that remain unsearched.
That **lower certificate** lets the search stop once no unsearched word can improve `U`.
An optional `cutoff` permits an earlier return when a witness reaches a requested weight threshold, without necessarily proving the exact distance.

**Coefficient weight** counts selected generator rows; **Hamming weight** counts nonzero coordinates in their sum.
These weights need not agree.
A **coefficient layer** enumerates all row selections with a given coefficient weight.
For classical codes and X-only or Z-only operators in Calderbank-Shor-Steane (CSS) codes, Hamming weight equals physical support weight.
For general Pauli operators, qLDPC uses a doubled Hamming weight internally (section 4).
Systematic bases relate the two weights and can allow BZ to stop before every codeword has been examined.

A BZ call proceeds as follows:

1. `get_distance_quantum` can choose direct exhaustive enumeration of small independent inputs before constructing an adapted basis.
2. Otherwise, construct the independent search basis, attach any quotient labels, and transform general Pauli inputs to Hamming coordinates.
3. Eligible basis rows initialize the witness bound `U`; preparation can choose an early exhaustive route before building information sets.
4. Construct systematic information-set bases and check their eligible rows, completing coefficient layer one; a selected overlapping replacement also has its rows checked.
5. Preparation may close the certificate immediately or choose a later exhaustive route using the information-set work estimate.
6. Otherwise, pack the selected bases and scan coefficient layers starting at two, updating the certificate only after each complete layer.

These helpers implement the search:

| Responsibility | Implementation |
| --- | --- |
| Binary classical, quantum, and joint CSS interfaces | [core.py](core.py): `get_distance_classical`, `get_distance_quantum`, `get_distance_css_brouwer_zimmermann` |
| Independent bases and logical labels | [_binary.py](_binary.py): `_row_reduce_binary`, `_get_nested_code_basis` |
| Disjoint and overlapping systematic bases | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_get_information_set_generators`, `_get_overlapping_information_sets` |
| Search preparation and bounds | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_prepare_brouwer_zimmermann_search`, `_get_brouwer_zimmermann_lower_bound` |
| One search's state and progression | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_BrouwerZimmermannSearch` |
| Packed candidate enumeration | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_scan_brouwer_zimmermann` |
| Shared upper bounds across searches | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_get_distance_brouwer_zimmermann_many` |
| Exhaustive alternatives and their cost estimates | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_exhaustive_is_cheaper`, `_brute_force_is_cheaper`; [brute_force.py](brute_force.py): `_get_distance_quantum_brute_force` |

## 2. Binary distance problems

BZ uses linear algebra over the binary field.
Addition is bitwise XOR.
Write `wt(v)` for Hamming weight; section 4 explains how general Pauli weight reduces to this case.

### Classical codes

For a classical binary code with generator matrix `G`, find

$$
d = \min_{v \in V,\;v \ne 0} \operatorname{wt}(v),
\qquad V = \operatorname{rowspan}(G).
$$

`get_distance_classical` removes dependent generator rows before selecting either method.
The resulting basis has `r = dim(V)` rows and `N` columns.
Thus `r` is the dimension being searched, not the number of parity checks, and `N` is the Hamming length.
Enumerating only nonempty selections of independent rows excludes the zero word.

### Quantum codes and nested spaces

For `get_distance_quantum`, let `S` be the span of the supplied `stabilizers` and let

$$
V = \operatorname{span}(S,\text{logical operators}).
$$

Find

$$
d = \min_{v \in V \setminus S} \operatorname{wt}(v).
$$

A nonzero vector in `S` is still logically trivial, so minimizing over all nonzero vectors in `V` would include invalid witnesses.
The search therefore needs a membership test for `S`.

For a binary CSS stabilizer code, the X-sector problem has `V = ker(H_z)` and `S = rowspan(H_x)`, with X and Z interchanged for the other sector.
Code methods in [common.py](../common.py) supply logical and stabilizer generators rather than asking this module to construct those kernels.
For subsystem codes, those callers also include gauge operators in the excluded span.
The distance helper treats its inputs as nested binary spaces; it does not assume that every supplied excluded generator is a commuting stabilizer.

## 3. Logical labels exclude the trivial subspace

`_get_nested_code_basis` builds an adapted independent basis:

```text
G = [ independent excluded-subspace rows ]
    [ independent logical quotient rows ]
```

If `s = dim(S)` and `q = dim(V/S)`, then `r = s + q`.
Associate a binary label matrix `M` with these rows:

```text
M = [ zero labels                    ]  s rows
    [ identity matrix of dimension q ]  q rows
```

For any coefficient row `a`,

$$
v = aG \in S \quad\Longleftrightarrow\quad aM = 0.
$$

Thus a candidate is eligible precisely when the XOR of its selected row labels is nonzero.
For classical searches, `labels=None` means every nonempty selection is eligible.

Every subsequent row operation is applied to both `G` and `M`.
If an invertible row transformation produces `G' = EG`, it must also produce `M' = EM`.
`aM'` then remains the quotient label of `aG'`.
Failing to transform labels can either accept stabilizers as witnesses or reject genuine logical operators.
`_row_reduce_binary` performs these coupled transformations.

`_assert_binary` rejects nonbinary values and nonbinary field types before conversion to binary storage arrays.
Code methods in [common.py](../common.py) use exhaustive field-aware enumeration for nonbinary inputs instead.

## 4. Converting general Pauli weight into Hamming weight

Homogeneous X-only or Z-only operators already use binary support vectors of length `n`, with Hamming weight equal to physical Pauli weight.
General operators use symplectic rows `[x | z]` of length `2n`.
Their physical weight counts positions where either `x` or `z` is nonzero.

`_symplectic_to_hamming` uses the `Saved_isometry` construction described in [section 2.4 of the quantum-code adaptation](https://arxiv.org/abs/2408.10743v2):

$$
T(x,z) = (x \mid z \mid x+z).
$$

At each qubit, `(0,0)` contributes weight zero and each of `(1,0)`, `(0,1)`, and `(1,1)` contributes Hamming weight two.
Consequently,

$$
\operatorname{wt}_{H}(T(v)) = 2\operatorname{wt}_{P}(v).
$$

BZ therefore searches vectors of Hamming length `3n`, scales the cutoff by two, and divides the final result by two.
Labels continue to describe the same excluded subspace because the map is linear and injective.
The resulting even Hamming weights do **not** imply even physical Pauli weights.

Below, a **coordinate** means a search-space Hamming coordinate after this transformation, if one was needed.
For general Pauli inputs there are `3n` such coordinates, not `n` physical qubits.
An **original coordinate** means one in this search space before per-basis pivot deletion and packing, not a coordinate before the symplectic transformation.

Direct exhaustive quantum enumeration can instead use symplectic weight without constructing this transformed space.

## 5. Systematic bases and information sets

A full information set `I` contains `r` search-space Hamming coordinates on which the basis has rank `r`.
Row reduction on those coordinates produces a basis `G_I` whose pivot columns form an identity matrix:

$$
G_I[:,I] = I_r.
$$

For a coefficient row `a`, the restriction of `aG_I` to these coordinates is exactly `a`.
Selecting `w` rows therefore puts exactly `w` nonzero entries in the information set, although the complete word can have more support elsewhere.

> Searching every coefficient support of size at most `t` in this basis covers every eligible word occupying at most `t` coordinates in its information set.

Different information sets generally give different bases for the same space.
The same coefficient indices in two bases need not describe the same Hamming word.
Duplicate represented words across bases are allowed; counting all candidate/set evaluations is not the same as counting distinct codewords.

`_InformationSet` stores the transformed generators, transformed labels, pivot rank, and original pivot-coordinate identities.
Column orders guide pivot selection but do not redefine column identities in the search space.
Coverage must be counted in those original coordinates, before each basis deletes its own pivots for packing.

### Partial systematic blocks

After extracting full disjoint information sets, a residual coordinate block may have rank `r_i < r`.
Its first `r_i` systematic rows contribute distinct pivot bits, while the remaining `r-r_i` rows have zero support on those pivots.
The complete generator matrix still spans the original `r`-dimensional space.
It is the selected coordinate block, not the ambient code, that has deficient rank.

If an unsearched word needs at least `t+1` basis rows, at most `r-r_i` of those rows can avoid its pivot block.
That block therefore contains at least

$$
b_i(t) = \max(0,\;t+1-(r-r_i))
$$

occupied coordinates.
For a full information set, this reduces to `b_i(t)=t+1`.

## 6. Stopping certificates

Suppose every basis has been searched through coefficient weight `t`, and let `U` be the best known eligible witness weight.
Any eligible word of weight below `U` that could still improve the answer must have escaped the completed enumeration in every basis.
Its pivot occupancy in block `i` must therefore be at least `b_i(t)`.

### Disjoint blocks

For disjoint pivot blocks, their occupancies add without double counting:

$$
\operatorname{wt}(v) \ge \sum_i b_i(t).
$$

`_get_information_set_generators` initially constructs this kind of family.
It uses a bounded deterministic packing heuristic, including fixed-seed alternative column orders when the initial packing does not reach the full-set count allowed by length and dimension.
`_select_brouwer_zimmermann_information_sets` removes partial blocks that cannot help before the full blocks would already certify `U`.

### Overlapping blocks

Let `C` be the maximum number of pivot blocks containing any one search-space Hamming coordinate.
Summing pivot occupancies can count a Hamming one at most `C` times, so

$$
\sum_i b_i(t)
\le C\operatorname{wt}(v).
$$

Define a lower certificate by

$$
L(t) =
\left\lceil \frac{\sum_i b_i(t)}{C} \right\rceil.
$$

For `m` full information sets, this becomes

$$
L(t) = \left\lceil \frac{m(t+1)}{C} \right\rceil.
$$

Disjoint blocks are the special case `C=1`.
The bound helper supports the rank-deficiency terms above, but the current overlapping-family selector deliberately proposes only full information sets.

[Algorithm 1 of the Algorithm 994 paper](https://arxiv.org/abs/1603.06757v2) gives the disjoint/rank-deficient bound.
The coverage-divided certificate above is derived here for qLDPC; it is not a formula reproduced from either cited paper.
The Algorithm 994 paper refers to Grassl for overlapping sets, but that formulation is not reproduced here.

The implementation evaluates these formulas with exact integer arithmetic.

### Common layers versus per-basis progress

Different bases may have different completed weights `t_i`, giving

$$
L = \left\lceil \frac{\sum_i b_i(t_i)}{C} \right\rceil.
$$

The Algorithm 994 paper notes in its prose that a bound can increase after each basis, although Algorithm 1 updates `L` after all bases complete a layer, as qLDPC does.
Its enumeration processes each basis's combinations in turn; qLDPC evaluates each support in all retained bases before advancing to the next support.
qLDPC therefore tracks a common completed weight `t`, not individual `t_i` values.
Tracking separate `t_i` values would permit other schedules justified by the same inequality.

### Rounding to a proved weight lattice

If every possible candidate weight is a multiple of `D`, round `L` up to `D*ceil(L/D)`, the smallest multiple of `D` that is at least `L`.
The symplectic transformation from section 4 supplies `D=2`.
Preparation also checks whether every ambient basis row has even Hamming weight.
Binary parity is linear, so an all-even basis proves that the entire ambient space is even; the implementation combines that fact with an existing divisor using the least common multiple.

Rows individually divisible in weight by four do not establish that every XOR has weight divisible by four.
For example, `11110` and `01111` each have weight four, but their XOR `10001` has weight two.
Only proved divisibility may strengthen the certificate.
The automatic even-row check and the family-selection policy below are qLDPC implementation choices, not extra steps claimed from the cited algorithms.

### What the certificate does, and does not, claim

`L(t)` bounds potential improvements still unsearched.
It is **not** necessarily a lower bound on the weight of witnesses already found.
Once `L(t) >= U`, there is no unsearched eligible word lighter than `U`, which is enough to stop.

For exact minimum multiplicity, [weight_enumerators.py](weight_enumerators.py) must also exhaust words whose weight equals the minimum `d`.
Its support collector stops only when `L(t) > d` and passes `upper_bound=d+1` to the shared family selector.
It transforms logical labels with every basis, uses the actual maximum pivot coverage in the bound, and removes duplicate physical supports across bases.
The selected family and its construction workspace must fit the counter's working-memory budget.

For example, the code generated by `101` and `011` has distance two.
All three pairs of its three coordinates are full information sets, with maximum coverage two.
After searching single rows in all three bases, the formula gives `ceil(3*2/2)=3`.
That does not assert distance three: all weight-two words have already been examined, and no unsearched word can improve the known weight-two witness.

## 7. Choosing overlapping information sets

`_get_overlapping_information_sets` starts from the retained full disjoint blocks.
It can return a replacement family and its actual maximum coverage, or `None` to keep the existing route.
Family selection is heuristic; its certificate uses verified ranks and coordinate counts.

Here `n_active` is the number of columns that are not identically zero in the entire ambient search basis.
It includes coordinates already used by existing pivot blocks, not just the residual uncovered columns.
For general Pauli inputs, these are columns of the transformed Hamming basis.

Family construction has four stages:

1. **Avoid unnecessary setup.**
   Skip already cheap work, complete disjoint packings, and families already at the size limit.
2. **Rule out impossible savings.**
   With `m` sets of size `r` over `n_active` active coordinates, some coordinate must occur at least `ceil(m*r/n_active)` times.
   Use that optimistic multiplicity to reject searches where even an ideal family cannot reduce estimated enumeration.
3. **Build coverage-balanced bases.**
   Prioritize least-used coordinates, break ties with a fixed seed, row-reduce the candidates, and transform labels in parallel.
   Do not append or score previously seen pivot sets.
   Detecting a duplicate still requires row reduction and consumes one attempt in the fixed budget.
4. **Keep the lowest-work family.**
   After each distinct set is appended, estimate work for the family built so far and retain the family with the lowest estimate.
   Stop if that estimate reaches the optimistic work bound or the family reaches the size limit.

By default, a replacement family contains at most eight full sets.
If construction starts with `m_0` full sets, it allows at most `2*(8-m_0)` attempts.
The setup screen skips existing estimates below 100,000 candidate/set evaluations.
These are performance-policy choices, not assumptions in the mathematical proof.
Selection uses the basis, labels, current upper bound, and fixed policy constants, together with the disjoint family and weight divisor derived during preparation.
That upper bound can come from row witnesses, another sector, or a cached distance in an omitted sector.
No code-family name or stored construction-specific pivot list enters it.
Changing a valid upper bound can change the selected family, but not the soundness of its certificate.

### The work estimate and exhaustive alternatives

`_get_brouwer_zimmermann_work` pessimistically assumes that no lighter witness will be found before the current upper bound is certified.
For `m` retained bases and a last required coefficient layer `t_*`, its estimate is

$$
W = m\sum_{w=1}^{t_*} \binom{r}{w}.
$$

This includes the systematic rows at coefficient weight one.
It is a work estimate, not a runtime guarantee: basis construction, word widths, and early witness improvements also affect actual performance.

Exhaustive enumeration can be selected at three points:

- A small independent quantum input can be enumerated directly before building an adapted basis.
- Preparation can choose exhaustive enumeration when information-set setup would dominate.
- After constructing information sets, `_brute_force_is_cheaper` can compare estimated BZ work with the eligible exhaustive count `(2**q - 1) * 2**(r-q)`.

Both methods accept redundant generators.
Classical inputs are reduced to an independent basis; explicit quantum brute force uses the adapted basis from section 3, separating the excluded span from independent quotient representatives.
The direct quantum shortcut verifies independence before bypassing that preparation.
The private packed enumerator therefore receives independent bases prepared or verified within [core.py](core.py) and [brouwer_zimmermann.py](brouwer_zimmermann.py); independence is not a caller-side precondition.
The brute-force implementation uses packed, vectorized blocks and Gray-code updates rather than the BZ coefficient-layer certificate.
Selecting the public BZ method does not promise that every input will actually run a BZ scan.

## 8. Packed enumeration and state progression

### Avoid reconstructing known pivot weights

Preparation deletes each basis's systematic pivot columns before packing the remaining bits into `uint64` words.
For a full set, a support of size `w` contributes exactly `w` deleted pivot bits.
For a partial block, its known contribution is the number of selected row indices below that block's rank.
The scan adds this known contribution to the population count of the remaining packed XOR, giving a Hamming weight in the search space.

Generator arrays and logical-label arrays are padded to common word widths across sets.
Each set retains its actual generator-word and label-word widths, so padding does not change a weight or an eligibility decision.
Weights are accumulated with sufficiently wide integer types rather than narrow per-word population-count storage.

### Enumerate increasing supports and reuse XOR prefixes

`_scan_brouwer_zimmermann` represents a coefficient support by its selected row indices in increasing order.
For coefficient weight `w`, it visits the same supports as `itertools.combinations(range(r), w)`.
It maintains the invariant

```text
prefixes[j, set] = XOR of the first j selected rows in that set's basis.
```

Extending a support therefore needs one additional row XOR per set and packed word, rather than rebuilding the complete sum.
The scan stores the current upper bound `U` in `best`.
For each completed support, it:

1. Restores the known pivot contribution and counts the remaining Hamming bits.
2. Skips candidates whose weight cannot improve `best`.
3. For lighter candidates, XORs the logical labels and accepts only nontrivial ones.
4. Updates the witness bound and advances to the next support.

Candidates with Hamming weight at least `best` can be discarded regardless of their logical label.

### Chunks, layers, and interruption

The Numba kernel is compiled lazily, cached, and bounds checked.
It returns after a bounded number of complete coefficient supports, at a reached cutoff, or when the layer finishes.
The BZ batch limit is `2**min(block_size, 15)` supports per native call.
A support is evaluated in every retained basis, so that limit is not a limit on candidate/set evaluations.

Within one `advance()` call, support indices, tree depth, and XOR prefixes persist across chunks.
Only when the **entire layer across every retained basis** is complete does `_BrouwerZimmermannSearch.advance` update the lower certificate and increment `next_weight`.
A cutoff, exception, or interruption does not certify an unfinished layer.
If more scanning is needed after interruption, a new `advance()` call starts that layer from the beginning; support indices and XOR prefixes are local to the previous call.

`_BrouwerZimmermannSearch` keeps the following state:

| State | Meaning |
| --- | --- |
| `generators`, `labels` | Packed systematic bases and their quotient labels |
| `ranks`, `widths`, `label_widths` | Each set's pivot rank and actual packed widths |
| `best` | Best currently usable upper bound |
| `lower_bound` | Certificate from completed coefficient layers |
| `next_weight` | Next coefficient weight to search |
| `weight_divisor` | Proved divisor of candidate Hamming weights |
| `max_coverage` | Maximum search-space Hamming-coordinate multiplicity across pivot sets |

A search is finished when its cutoff is reached, its certificate reaches `best`, or all coefficient weights have been enumerated.

## 9. CSS sectors share witnesses, not proofs

For overall CSS distance, the goal is `min(d_x, d_z)`.
`_get_distance_brouwer_zimmermann_many` prepares the sectors and advances them round-robin, one coefficient layer at a time.
An improved witness in either sector lowers the common upper bound.

Each sector retains its own information sets, coverage, parity divisor, and completed-layer certificate.
A weight-seven witness from one sector can cap an even-weight sector at seven, but it does not make seven an admissible weight in that even sector.
That sector only needs to establish that it cannot beat the shared witness.
It need not establish its own exact distance.

Consequently, the joint result proves the minimum, not necessarily both individual sector distances.
The code-level cache in [common.py](../common.py) records that distinction.
A known distance in an omitted sector can also supply an upper bound to the remaining searches.
In the ordinary binary BZ overall-distance route, an equal-X/Z-distance flag makes the code-level method search just one sector.
Identical X and Z check matrices establish this equality automatically.
A promise supplied by a code constructor, such as `QCCode`, or by a user can also set the flag.
The distance methods trust such promises rather than verifying them.
Explicit brute-force and specialized distance paths can use different routing, even when that flag is set.

## 10. Worked example: certify eight with fewer row combinations

Consider the QC64 input used by `test_brouwer_zimmermann_overlap_qc` in [brouwer_zimmermann_test.py](brouwer_zimmermann_test.py).
One CSS sector has 64 coordinates and ambient dimension 41, already supplies a weight-eight witness, and spans only even-weight words.
Its useful disjoint search has one full information set.

With one set, coefficient layer six gives the raw bound seven, which the even-weight proof rounds to eight.
The search therefore enumerates layers two through six after checking the systematic rows.

A coverage-balanced family can instead use three full sets whose maximum coordinate multiplicity is two.
Completing layer four gives

$$
L(4)=\left\lceil\frac{3(4+1)}{2}\right\rceil=8.
$$

The certificate reaches eight directly, even before parity rounding.

| Strategy | Last coefficient layer | Candidate/set evaluations after systematic-row setup |
| --- | ---: | ---: |
| One disjoint set, with even-weight rounding | 6 | `sum(comb(41,w) for w in range(2,7)) = 5,358,536` |
| Three full sets, maximum coverage two | 4 | `3 * sum(comb(41,w) for w in range(2,5)) = 338,250` |

Three bases require fewer candidate/set evaluations here because they avoid layers five and six.
The table excludes basis construction and packing.
The X and Z families must be constructed and rank-checked independently; equal sector distances do not establish that the same pivot columns work in both sectors.
The QC64 construction carries an equal-sector promise, so its ordinary no-sector distance call can search only X rather than run a joint X/Z calculation.

Independent per-basis scanning could finish layer four in one basis while the other two remain complete through layer three.
The resulting `ceil((5+4+4)/2)=7` certificate rounds to eight by even parity, requiring 135,710 candidate/set evaluations after row setup for this family.
The current shared-support scan does not implement that schedule; the table describes its actual work, not the minimum enumeration mathematically necessary for these sets.

## 11. Public behavior and practical limits

### Exactness and cutoff

Without an early cutoff return, the search either closes its certificate or exhausts the eligible space.
Both establish the exact minimum.
With a positive cutoff, it may instead stop at an observed upper bound at or below that cutoff.
For a valid nontrivial binary problem, the default cutoff of one is still exact because a positive distance cannot be less than one.
Use `cutoff=0` when explicitly requiring a fully certified result rather than an early threshold result.
Larger cutoffs must not be reported as proving an exact distance merely because the method is called an exact-distance method.

The layer-loop argument assumes a nonempty independent ambient basis and an eligible nontrivial quotient.
The module-level interfaces handle empty spaces before enumeration.
A zero-dimensional classical space returns its block length.
For quantum inputs, no supplied logical rows returns the physical qubit count, while a nonempty logical family contained in the excluded span returns zero.
Both methods use these conventions regardless of redundant excluded rows.
These are explicit return conventions for problems with no eligible words, not witness-derived distance certificates.

### Caches and special constructions

The code-level methods in [common.py](../common.py) can return cached parameters or construction-specific formulas without invoking this module.
Use `forget_distance()` before timing an ordinary public distance call that is meant to recompute the answer.
It does not remove specialized formulas or disable equal-sector promises.
Similarly, `method="brute_force"` is not a command to ignore a code object's known-distance cache.

These small examples may use exhaustive enumeration even with the default BZ method:

```python
import numpy as np
from qldpc import codes

generators = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)
assert codes.get_distance_classical(generators, cutoff=0) == 2

logicals = np.array([[0, 1, 1]], dtype=int)
stabilizers = np.array([[1, 1, 0]], dtype=int)
assert codes.get_distance_quantum(logicals, stabilizers, homogeneous=True, cutoff=0) == 2
```

### Complexity and unsupported assumptions

BZ remains a combinatorial search; a weak certificate can require exponentially many row combinations.
Additional bases cost elimination time, packed storage, and prefix-XOR scratch.
Neither the selector's candidate estimate nor a lower candidate count guarantees faster execution on every input.

`_scan_brouwer_zimmermann` runs serially.
Fixed-seed tie-breaking affects which valid bases are tried, not whether a certificate is sound.
`NUMBA_DISABLE_JIT=1` set before import runs the decorated numerical code without JIT compilation for debugging; it is not a separate optimized backend.

## 12. Invariants to preserve when modifying the algorithm

- **Preserve the searched quotient.**
  Row operations must transform logical labels with the generators.
- **Keep search-space coordinates identifiable.**
  Compute pivot coverage before per-basis pivot deletion and bit packing.
- **Use the right rank term.**
  Partial blocks require their deficiency correction; the full-set formula cannot simply be applied to them.
- **Count overlap exactly.**
  Coverage is an actual coordinate multiplicity, not the number of statistically independent bases.
- **Advance certificates only after complete work.**
  A cutoff, exception, or interrupted chunk is not a completed coefficient layer.
- **Keep bounds in consistent units.**
  General symplectic searches use doubled Hamming weights internally.
- **Prove divisibility rather than infer it from measurements.**
  Even ambient parity is linear; higher divisibility needs additional proof.
- **Share upper bounds without sharing local certificates.**
  Sector-specific parity, coverage, and completed work must remain local.
- **Make regression tests non-vacuous.**
  Verify that overlap was actually selected and that an independent oracle checks a case where enumeration improves the starting witness.

The [BZ test module](brouwer_zimmermann_test.py) contains exhaustive small-word certificate checks, logical-label checks, a default-policy QC work-reduction check, and a shortened-BCH quotient whose setup witnesses are heavier than its exhaustive minimum.
The last case checks actual witness discovery, not just certification of an already optimal row.
The repository also requires statement coverage, typing, formatting, lint, and strict documentation validation.
From the repository root, run:

```bash
python -m pytest src/qldpc/codes/distance/ -q
python checks/coverage_.py src/qldpc/codes/distance/
python checks/all_.py
```

## References and further reading

- [core.py](core.py): public signatures and dispatch, re-exported by [the package facade](__init__.py).
- [brouwer_zimmermann.py](brouwer_zimmermann.py): BZ certificates, information sets and numerical kernel.
- [brute_force.py](brute_force.py): exact exhaustive enumeration.
- [_binary.py](_binary.py): shared binary algebra, packing and weight functions.
- [brouwer_zimmermann_test.py](brouwer_zimmermann_test.py) and [core_test.py](core_test.py): executable invariants and independent small-instance oracles.
- [common.py](../common.py): code-level caches, CSS sector routing, and subsystem gauge exclusion.
- [Library map](../../../../docs/source/library_map.rst): package-level introduction to exact distances and bounds.
- [Algorithm 994 paper, v2](https://arxiv.org/abs/1603.06757v2): section 2 and Algorithm 1 give the classical disjoint/rank-deficient certificate and discuss per-basis progress.
- [Quantum-code adaptation, v2](https://arxiv.org/abs/2408.10743v2): section 2.4 describes the `Saved_isometry` symplectic-to-Hamming construction used here.
