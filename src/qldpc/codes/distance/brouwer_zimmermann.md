# The Brouwer-Zimmermann distance algorithm in qLDPC

This guide explains how qLDPC computes exact distances of binary codes using the Brouwer-Zimmermann (BZ) algorithm.
It covers the mathematics behind BZ's stopping criterion, the extensions that qLDPC adds, and how the code in this directory implements them.

qLDPC's BZ search combines three sources:

- The stopping criterion for disjoint and rank-deficient information sets follows [Algorithm 994](https://arxiv.org/abs/1603.06757), the paper that describes a BZ implementation for classical codes.
- The reduction from symplectic (Pauli) weight to Hamming weight follows a [paper that adapts BZ to quantum codes](https://arxiv.org/abs/2408.10743).
- The stopping criterion for overlapping information sets (section 6) is derived in this guide.
  It is not taken from either paper.

The public interfaces for binary distances live in [core.py](core.py).
They dispatch either to [brouwer_zimmermann.py](brouwer_zimmermann.py), which implements BZ, or to [brute_force.py](brute_force.py), which enumerates words exhaustively.

## 1. Overview

### The idea

Every codeword is a sum of rows of a generator matrix.
BZ examines such sums in order of how many rows they combine, and tracks two quantities:

- An **eligible word** is a word that is allowed to determine the distance.
  For a classical code, every nonzero codeword is eligible.
  For a quantum code, a word is eligible if it is a logical operator, meaning a nonzero element of the code's logical space that is not a stabilizer (or gauge) element.
- A **witness** is an eligible word whose weight has been computed.
  The lightest known witness has weight `U`, which is an **upper bound** on the distance: the distance is at most `U`, but a lighter eligible word might still exist.
- A **lower certificate** `L` is a proven statement that no eligible word that has not yet been examined has weight below `L`.
  Once `L >= U`, no unexamined word can beat the best witness, so `U` is the distance and the search stops.

The certificate is strong because BZ uses several generator matrices in *systematic form*.
In such a matrix, a small number of rows can only produce words with large Hamming weight outside a chosen set of coordinates, so words that have not been examined are forced to be heavy.
Sections 5 and 6 make this precise.

### Terminology

Throughout this guide, a **basis** is a matrix whose rows are linearly independent and span the space being searched.
Let `r` be the number of rows (the dimension of that space), and let `N` be the number of columns.

- A **coefficient vector** `a` is a binary vector of length `r` that selects rows of a basis `G`.
  The corresponding word is `aG`, the XOR (binary sum) of the selected rows.
- The **coefficient weight** of `aG` is the number of rows selected, which is the number of nonzero entries of `a`.
- The **Hamming weight** of a word is its number of nonzero coordinates.
  The coefficient weight and the Hamming weight of the same word are different quantities.
- A **coefficient layer** (or simply a layer) is the set of all words with a given coefficient weight `w`, taken over every row selection of size `w`.

Hamming weight equals physical weight for classical codes and for operators that contain only X or only Z Pauli operators.
General Pauli operators use a transformation, described in section 4, that makes their Hamming weight twice their physical weight.

### How one BZ call proceeds

1. `get_distance_quantum` may first choose exhaustive enumeration directly, when the input is small and independent.
2. Otherwise, the search builds an independent basis for the space to be searched, attaches quotient labels to its rows (section 3), and, for general Pauli operators, transforms them to Hamming coordinates (section 4).
3. The lightest eligible row of that basis gives the initial upper bound `U`.
   Preparation may stop here and choose exhaustive enumeration, if the cost estimates favor it.
4. The search builds one or more systematic bases (section 5) and checks their eligible rows.
   These rows are all the coefficient-weight-one words, so layer one is complete.
   If it selected an overlapping family of bases (section 7), it also checks the rows of that family.
5. Preparation may stop immediately, either because the lower certificate already reaches `U`, or because an exhaustive enumeration is estimated to be cheaper than the remaining BZ work.
6. Otherwise, the search packs the bases into `uint64` words and scans layers starting from layer two.
   It updates the lower certificate only after a whole layer is complete in every basis.

An optional `cutoff` lets the search return as soon as `U` is at most the cutoff.
Such a result is only an upper bound, not a proof of the exact distance (section 11).

### Where each step is implemented

| Responsibility | Implementation |
| --- | --- |
| Binary classical, quantum, and joint CSS interfaces | [core.py](core.py): `get_distance_classical`, `get_distance_quantum`, `get_distance_css_brouwer_zimmermann` |
| Independent bases and quotient labels | [_binary.py](_binary.py): `_row_reduce_binary`, `_get_nested_code_basis` |
| Disjoint and overlapping systematic bases | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_get_information_set_generators`, `_get_overlapping_information_sets` |
| Search preparation and the lower certificate | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_prepare_brouwer_zimmermann_search`, `_get_brouwer_zimmermann_lower_bound` |
| State and progress of one search | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_BrouwerZimmermannSearch` |
| Packed enumeration of candidate words | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_scan_brouwer_zimmermann` |
| Sharing upper bounds between searches | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_get_distance_brouwer_zimmermann_many` |
| Exhaustive alternatives and their cost estimates | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_exhaustive_is_cheaper`, `_brute_force_is_cheaper`; [brute_force.py](brute_force.py): `_get_distance_quantum_brute_force` |

## 2. Binary distance problems

BZ works over the binary field.
Addition is bitwise XOR.
`wt(v)` denotes the Hamming weight of `v`.
Section 4 explains how the weight of a general Pauli operator reduces to this case.

### Classical codes

Let `G` be a generator matrix of a classical binary code.
Its distance is

$$
d = \min_{v \in V,\;v \ne 0} \operatorname{wt}(v),
\qquad V = \operatorname{rowspan}(G).
$$

`get_distance_classical` first removes dependent rows of `G`, leaving a basis with `r = dim(V)` rows and `N` columns.
Here `r` is the dimension of the code, not the number of parity checks, and `N` is the code's block length.
Enumerating only nonempty selections of independent rows excludes the zero word.

### Quantum codes and nested spaces

Let `S` be the span of the `stabilizers` passed to `get_distance_quantum`, and let `V` be the span of `S` together with the supplied logical operators:

$$
V = \operatorname{span}(S,\text{logical operators}).
$$

The distance is

$$
d = \min_{v \in V \setminus S} \operatorname{wt}(v).
$$

Words in `S` are logically trivial, so they must not be accepted as witnesses.
The search therefore needs a way to test whether a word lies in `S`, which is the purpose of the quotient labels in section 3.

For a binary CSS code, the X-sector problem has `V = ker(H_z)` and `S = rowspan(H_x)`.
The Z-sector problem exchanges X and Z.
The code methods in [common.py](../common.py) supply the logical and stabilizer generators, so this module never has to compute those kernels.
For subsystem codes, the same callers also include the gauge operators in `S`.
This module only sees a pair of nested binary spaces `S ⊆ V`; it does not assume that the generators of `S` are commuting stabilizers.

## 3. Quotient labels exclude the trivial subspace

`_get_nested_code_basis` builds a basis of `V` that is adapted to `S`.
Let `s = dim(S)` and `q = dim(V/S)`, so that `r = s + q`.
The basis has the form

```text
G = [ s independent rows spanning S             ]
    [ q independent rows, one per logical class ]
```

Each row of `G` is given a binary **label**.
The labels form a matrix `M` with `r` rows and `q` columns:

```text
M = [ zero matrix with s rows     ]
    [ identity matrix, q x q      ]
```

For any coefficient vector `a`,

$$
aG \in S \quad\Longleftrightarrow\quad aM = 0.
$$

A candidate word `aG` is therefore eligible exactly when the XOR of its selected row labels, `aM`, is nonzero.
For a classical search there is no excluded subspace, so `labels=None` and every nonempty selection is eligible.

Every row operation on `G` must be applied identically to `M`.
If an invertible row transformation replaces `G` by `G' = EG`, the labels must become `M' = EM`.
Then `aM'` is still the label of the word `aG'`.
Omitting this step would either accept stabilizers as witnesses or reject genuine logical operators.
`_row_reduce_binary` performs these paired transformations.

`_assert_binary` rejects nonbinary values and field types before they are converted to binary storage arrays.
For nonbinary inputs, the code methods in [common.py](../common.py) use a field-aware exhaustive enumeration instead.

## 4. Converting Pauli weight to Hamming weight

An operator that contains only X (or only Z) Pauli operators on `n` qubits is a binary support vector of length `n`.
Its Hamming weight is its physical weight.

A general Pauli operator is a symplectic row `[x | z]` of length `2n`.
Its physical weight is the number of qubits on which `x` or `z` is nonzero, which is not a Hamming weight of `[x | z]`.

`_symplectic_to_hamming` fixes this using the `Saved_isometry` construction from [section 2.4 of the quantum-code adaptation](https://arxiv.org/abs/2408.10743v2):

$$
T(x,z) = (x \mid z \mid x+z).
$$

On each qubit, the pair `(x_i, z_i)` is `(0,0)` for the identity and `(1,0)`, `(0,1)`, or `(1,1)` for a nontrivial Pauli operator.
The identity contributes Hamming weight zero to `T`, and each nontrivial Pauli operator contributes weight exactly two.
Hence

$$
\operatorname{wt}_{H}(T(v)) = 2\operatorname{wt}_{P}(v),
$$

where `wt_P` is the physical Pauli weight.

BZ therefore runs on words of Hamming length `3n`.
It doubles any `cutoff` before comparing, and halves the final result.
The labels from section 3 remain valid because `T` is linear and injective.
Note that `T` makes every Hamming weight even; this does **not** mean that physical Pauli weights are even.

For the rest of this guide:

- The **search space** is the space of binary words on which BZ actually runs.
  It is the original binary space for classical and X-only or Z-only problems, and the image of `T` for general Pauli problems.
- A **search coordinate** is a coordinate of the search space.
  For general Pauli problems there are `3n` search coordinates, not `n` qubits.
- Search coordinates are numbered before each basis removes its own pivot columns for packing (section 8).
  Pivot coverage in sections 5 through 7 is counted using these numbers.

The direct exhaustive route for quantum problems may use symplectic weight directly, without constructing the search space.

## 5. Systematic bases and information sets

An **information set** is a set `I` of `r` search coordinates on which the rows of the basis restrict to a rank-`r` matrix.
Row reduction on those coordinates produces a **systematic basis** `G_I`, which restricts to an identity matrix on `I`:

$$
G_I[:,I] = I_r.
$$

The coordinates in `I` are the **pivot columns** of `G_I`.

For a coefficient vector `a`, the restriction of `aG_I` to the coordinates in `I` is exactly `a`.
Selecting `w` rows therefore gives a word with exactly `w` nonzero entries on `I`.
The word may have additional support outside `I`.
This gives the fact that BZ relies on:

> If every selection of at most `t` rows of `G_I` has been examined, then every eligible word with at most `t` nonzero coordinates in `I` has been examined.

Different information sets generally give different bases of the same space.
The same coefficient vector in two such bases usually describes two different words.
Some words may be examined in more than one basis, so the number of candidate/set evaluations (candidates counted once per basis) is not the number of distinct codewords examined.

`_InformationSet` stores a basis's transformed generators and labels, its rank, and the search-coordinate numbers of its pivots.
The column order used while choosing pivots affects only which pivots are found, not the numbering of search coordinates.

### Partial systematic blocks

After extracting as many disjoint full information sets as possible, the remaining coordinates may form a block on which the basis has rank `r_i < r`.
Row reduction on this block gives a basis of the whole space in which:

- the first `r_i` rows have distinct pivot columns in the block, and
- the remaining `r - r_i` rows are zero on the block.

The block is rank deficient, but the basis as a whole still spans the full `r`-dimensional space.

Suppose a word needs at least `t+1` basis rows.
At most `r - r_i` of those rows can be zero on the block, so at least `t + 1 - (r - r_i)` of them are pivot rows.
Therefore the word has at least

$$
b_i(t) = \max(0,\;t+1-(r-r_i))
$$

nonzero coordinates in the block.
For a full information set (`r_i = r`), this reduces to `b_i(t) = t + 1`.

## 6. Stopping certificates

Suppose every basis has been fully searched through coefficient weight `t`, so every selection of at most `t` rows has been examined in every basis.
Let `U` be the weight of the lightest known witness.

Consider an eligible word `v` that has not been examined, in any basis.
In each basis `i`, the word `v` must need at least `t+1` rows, so it has at least `b_i(t)` nonzero coordinates in block `i` (section 5).
The following bounds on `wt(v)` come from combining these per-block counts.

### Disjoint blocks

If the blocks share no coordinates, their nonzero coordinates are all different, so the counts add:

$$
\operatorname{wt}(v) \ge \sum_i b_i(t).
$$

`_get_information_set_generators` builds a family of disjoint blocks with a bounded, deterministic packing heuristic.
If the first column order does not yield as many full information sets as the length and dimension allow, it retries with up to eight fixed-seed random column orders.
`_select_brouwer_zimmermann_information_sets` then discards any partial block that cannot help before the full blocks would already certify `U`.

### Overlapping blocks

Let `C` be the **maximum coverage**: the largest number of blocks that contain any single search coordinate.
When summing `b_i(t)` over blocks, each nonzero coordinate of `v` is counted at most `C` times, so

$$
\sum_i b_i(t) \le C\operatorname{wt}(v).
$$

This gives the **lower certificate**

$$
L(t) =
\left\lceil \frac{\sum_i b_i(t)}{C} \right\rceil.
$$

With `m` full information sets, it becomes

$$
L(t) = \left\lceil \frac{m(t+1)}{C} \right\rceil.
$$

Disjoint blocks are the special case `C = 1`.
`_get_brouwer_zimmermann_lower_bound` supports the rank-deficient terms `b_i(t)` for partial blocks.
The selector that builds overlapping families, however, proposes only full information sets.

Algorithm 1 of the [Algorithm 994 paper](https://arxiv.org/abs/1603.06757v2) gives the disjoint and rank-deficient bounds.
That paper mentions overlapping sets in connection with Grassl's work, but does not give a formulation of the overlapping bound.
The coverage-divided bound above is derived for qLDPC.

All of these formulas are evaluated with exact integer arithmetic.

### Common layers versus per-basis progress

The bases could in principle be searched to different depths `t_i`.
The same argument then gives

$$
L = \left\lceil \frac{\sum_i b_i(t_i)}{C} \right\rceil.
$$

Algorithm 994 enumerates the combinations of each basis in turn, and its prose notes that the bound can increase after every basis.
Its Algorithm 1 nevertheless updates the bound only after all bases complete a layer.
qLDPC does the same, with one difference in scheduling: it examines each support (a selection of rows) in every basis before moving to the next support.
It therefore tracks one completed coefficient weight `t` shared by all bases, rather than separate values `t_i`.
Tracking `t_i` separately would allow other schedules, justified by the same inequality.

### Rounding to a proven weight multiple

Suppose every candidate has a weight that is a multiple of some integer `D`.
Then `L` can be rounded up to `D * ceil(L/D)`, the smallest multiple of `D` that is at least `L`.
The value of `D` can come from two sources:

- The symplectic transformation of section 4 gives `D = 2`.
- Preparation also checks whether every row of the search basis has even Hamming weight.
  The sum of two even-weight words has even weight, so if all basis rows are even, every word in the space is even.
  When this holds, the implementation combines `D = 2` with any existing divisor by taking the least common multiple.

Divisibility must be proven for every word, not only for the basis rows, so rows that individually have weight divisible by four are not enough.
For example, `11110` and `01111` each have weight four, but their XOR `10001` has weight two.
Only the even-weight check above is used.
This check and the heuristic for selecting overlapping families (section 7) are implementation choices of qLDPC, not steps from the cited algorithms.

### What the certificate does and does not claim

`L(t)` bounds the weight of words that have *not yet been examined*.
It is **not** necessarily a lower bound on the weight of witnesses that were already found.
When `L(t) >= U`, no unexamined eligible word is lighter than `U`, which is enough to stop.

For example, take the code generated by `101` and `011`, whose distance is two.
Each of the three pairs of coordinates is a full information set, so there are `m = 3` full sets, and each coordinate lies in two of them, so `C = 2`.
After examining the single rows in all three bases (`t = 1`), the formula gives `ceil(3*2/2) = 3`.
This does not claim that the distance is three.
It says that every word of weight below three has already been examined, and the known witness of weight two is the lightest one found.

## 7. Choosing overlapping information sets

`_get_overlapping_information_sets` starts from the full disjoint blocks that were retained.
It returns either a replacement family of full information sets together with the family's actual maximum coverage, or `None` to keep the disjoint family.
Which family to choose is decided heuristically.
The certificate for the chosen family uses only verified ranks and actual coordinate counts, so it is sound regardless of the heuristic.

Let `n_active` be the number of columns of the search basis that are not identically zero.
This count includes columns that existing pivot blocks already use, not only the columns that remain uncovered.
For general Pauli problems, these are columns of the transformed basis from section 4.

Family construction has four stages.

1. **Skip unnecessary work.**
   Return `None` if the disjoint family is already cheap to search, already covers every active column, or is already at the size limit.
2. **Rule out impossible savings.**
   `m` full sets contain `m*r` pivot coordinates in total, spread over `n_active` columns, so some column is covered at least `ceil(m*r/n_active)` times.
   Using that value as `C` gives an optimistic estimate of the work for each `m`.
   If even the best optimistic estimate does not improve on the current family's estimated work, return `None`.
3. **Build coverage-balanced bases.**
   Order columns so that the least-used ones come first, breaking ties with a fixed random seed, then row reduce on that order and transform the labels in parallel.
   Skip any information set that was already seen.
   Detecting a duplicate still requires the row reduction, so it consumes one attempt from a fixed budget.
4. **Keep the cheapest family.**
   After each new information set is added, estimate the work for the family built so far, and remember the best family.
   Stop when the estimate reaches the optimistic bound from stage 2, or the family reaches the size limit.

The policy constants are:

- A family has at most eight full sets.
- If construction starts with `m_0` full sets, it makes at most `2*(8 - m_0)` attempts.
- Stage 1 skips families whose estimated work is below 100,000 candidate/set evaluations.

These are performance choices, not assumptions of the mathematical argument.

The selection depends only on:

- the basis and its labels,
- the current upper bound `U`,
- the fixed constants above, and
- the disjoint family and weight divisor from preparation.

The upper bound can come from a row witness, from the other sector of a CSS code, or from a cached distance of an omitted sector.
The selection never uses the name of a code family or a stored, construction-specific list of pivots.
A different valid upper bound can change which family is selected, but it cannot make the resulting certificate unsound.

### The work estimate and exhaustive alternatives

`_get_brouwer_zimmermann_work` estimates the number of candidate/set evaluations needed to certify `U`.
It pessimistically assumes that no lighter witness appears before the certificate reaches `U`.
Let `m` be the number of bases and let `t_*` be the smallest layer after which `L` reaches `U`.
The estimate is

$$
W = m\sum_{w=1}^{t_*} \binom{r}{w}.
$$

The sum starts at `w = 1`, so it includes the systematic rows.
This is an estimate of work, not a guarantee on runtime.
Basis construction, the width of packed words, and early improvements of the witness also affect the actual runtime.

Exhaustive enumeration can be chosen at three points:

- Before building an adapted basis, `get_distance_quantum` can enumerate a small independent quantum input directly.
- During preparation, `_exhaustive_is_cheaper` can choose exhaustive enumeration when the setup cost of information sets would dominate.
- After the information sets are built, `_brute_force_is_cheaper` compares the estimated BZ work with the number of eligible words, `(2**q - 1) * 2**(r-q)`, where `q` is the number of logical rows.

Neither method requires its caller to supply independent generators.
Classical inputs are first reduced to an independent basis.
Quantum brute force uses the adapted basis from section 3.
The direct quantum shortcut checks independence before skipping that preparation.
The packed enumerators therefore always receive independent bases, prepared or verified in [core.py](core.py) or [brouwer_zimmermann.py](brouwer_zimmermann.py).

The exhaustive implementation enumerates words in vectorized, packed blocks and updates them using Gray-code steps.
It does not use the BZ certificate.
Consequently, calling the public BZ method does not guarantee that a BZ scan actually runs.

## 8. Packed enumeration and state progression

### Reconstructing weights without storing pivots

Preparation removes the pivot columns of each basis before packing the remaining columns into `uint64` words.
The known contribution of the removed columns to a word's weight is easy to restore:

- For a full information set, a support of `w` rows contributes exactly `w` removed pivot bits.
- For a partial block of rank `r_i`, the contribution is the number of selected row indices below `r_i`.

The scan adds this contribution to the population count of the packed XOR of the remaining columns.
The sum is the Hamming weight in the search space.

The arrays of generators and labels are zero-padded to a common number of words across all bases.
Each basis keeps its own true generator width and label width, so padding changes neither weights nor eligibility decisions.
Weights are accumulated in sufficiently wide integer types, not in narrow per-word population counts.

### Enumerating supports and reusing XOR prefixes

`_scan_brouwer_zimmermann` describes a candidate by its **support**: the selected row indices in increasing order.
For coefficient weight `w`, it visits the same supports as `itertools.combinations(range(r), w)`.
It maintains the invariant

```text
prefixes[j, set] = XOR of the first j selected rows in that set's basis.
```

Extending a support by one row then costs one row XOR per basis and packed word, instead of recomputing the whole sum.

The scan stores the current upper bound `U` in the variable `best`.
For each completed support, and in each basis, it does the following:

1. Add the known pivot contribution to the population count of the packed XOR.
2. If the resulting weight is at least `best`, discard the candidate, whatever its label is.
3. Otherwise, XOR the selected row labels and accept the candidate only if the result is nonzero.
4. If the candidate is accepted, lower `best` to its weight, and continue with the next support.

### Chunks, layers, and interruption

The Numba kernel is compiled lazily, cached, and bounds checked.
Each call to the kernel returns after one of three events:

- a bounded number of complete supports has been visited,
- the cutoff has been reached, or
- the layer is finished.

The bound is `2**min(block_size, 15)` supports per kernel call.
Every support is evaluated in every retained basis, so this limit is not a limit on candidate/set evaluations.

Within one `advance()` call, the support indices, tree depth, and XOR prefixes persist across kernel calls.
`_BrouwerZimmermannSearch.advance` updates the lower certificate and increments `next_weight` only after the **whole layer is complete in every retained basis**.
A cutoff, an exception, or an interruption does not certify an unfinished layer.
After an interruption, a new `advance()` call restarts the unfinished layer from its first support, because the support indices and prefixes were local to the earlier call.

`_BrouwerZimmermannSearch` stores the following state:

| State | Meaning |
| --- | --- |
| `generators`, `labels` | Packed systematic bases and their quotient labels |
| `ranks`, `widths`, `label_widths` | Pivot rank and true packed widths of each basis |
| `best` | Lightest witness known, which is the current upper bound |
| `lower_bound` | Lower certificate from the layers completed so far |
| `next_weight` | Next coefficient weight to search |
| `weight_divisor` | Proven divisor of every candidate Hamming weight |
| `max_coverage` | Maximum number of pivot blocks containing any search coordinate |

A search is finished when its cutoff is reached, its lower certificate reaches `best`, or every coefficient weight has been enumerated.

## 9. CSS sectors share witnesses, not certificates

The distance of a CSS code is `min(d_x, d_z)`, where `d_x` and `d_z` are the distances of the X and Z sectors.
`_get_distance_brouwer_zimmermann_many` prepares one search per sector and advances them in round-robin order, one coefficient layer at a time.
A witness found in either sector lowers the upper bound used by both.

Each sector keeps its own information sets, coverage, weight divisor, and lower certificate.
This matters when the sectors differ.
For example, a witness of weight seven from one sector caps the other sector's upper bound at seven, even if the other sector contains only even-weight words and so can never have a witness of weight seven.
That sector only has to prove that it cannot beat the shared witness.
It does not need to determine its own exact distance.

Consequently, a joint search proves the minimum `min(d_x, d_z)`, but not necessarily each of `d_x` and `d_z`.
The code-level cache in [common.py](../common.py) records this distinction.
A known distance for a sector that is omitted from the search can also provide an upper bound for the remaining search.

If a code's X and Z distances are known to be equal, the code-level method for the overall distance, using the ordinary binary BZ route, searches only one sector.
The code records this knowledge in an equal-X/Z-distance flag.
The flag is set automatically when the X and Z check matrices are identical.
A code constructor, such as `QCCode`, or a user can also set it as a promise.
The distance methods trust such a promise and do not verify it.
Explicit brute-force searches and specialized distance routes can ignore the flag.

## 10. Worked example: certifying distance eight with fewer candidates

This example uses the QC64 input from `test_brouwer_zimmermann_overlap_qc` in [brouwer_zimmermann_test.py](brouwer_zimmermann_test.py).
One CSS sector of that code has these properties:

- It has 64 coordinates and ambient dimension `r = 41`.
- Its setup rows already provide a witness of weight eight (`U = 8`).
- Every word in it has even weight.

**One disjoint information set.**
The search is certified after layer six: with `m = 1` and `C = 1`, `L(6) = 6 + 1 = 7`, which the even-weight divisor rounds up to eight.
After checking the systematic rows, the search therefore enumerates layers two through six.

**Three overlapping information sets.**
A coverage-balanced family can have `m = 3` full sets with maximum coverage `C = 2`.
Completing layer four gives

$$
L(4)=\left\lceil\frac{3(4+1)}{2}\right\rceil=8.
$$

This reaches eight directly, even before rounding to an even weight.

| Strategy | Last layer | Candidate/set evaluations after the systematic rows |
| --- | ---: | ---: |
| One disjoint set, with even-weight rounding | 6 | `sum(comb(41,w) for w in range(2,7)) = 5,358,536` |
| Three full sets, maximum coverage two | 4 | `3 * sum(comb(41,w) for w in range(2,5)) = 338,250` |

The three-set family needs fewer evaluations because it avoids layers five and six.
The table does not include the cost of constructing and packing the bases.
The X and Z sectors need separate families, each checked for rank independently, because equal distances do not imply that the same pivot columns work in both sectors.
QC64 carries an equal-sector promise (section 9), so the ordinary overall distance call searches only the X sector instead of running a joint X/Z search.

A different schedule could search the bases to different depths: for instance, finish layer four in one basis while the other two stop after layer three.
Then `ceil((5+4+4)/2) = 7`, which rounds up to eight by even parity, and this requires 135,710 candidate/set evaluations after the systematic rows.
qLDPC does not implement this schedule, because it examines each support in all bases.
The table therefore shows the actual work of qLDPC's scan, not the minimum amount of enumeration that these three sets would mathematically require.

## 11. Public behavior and practical limits

### Exactness and cutoff

Without an early return caused by a cutoff, a BZ search ends in one of two ways: its certificate reaches `U`, or it exhausts the eligible space.
Either way, the result is the exact distance.

With a positive `cutoff`, the search may instead stop as soon as `U` is at most the cutoff.
The result is then an upper bound that is at or below the cutoff, and it is not necessarily the exact distance.
The default `cutoff` of one is still exact for a valid nontrivial binary problem, because a positive distance cannot be less than one.
Use `cutoff=0` to require a fully certified result.
A larger cutoff must not be described as proving an exact distance.

The layer-by-layer argument assumes a nonempty independent basis and at least one eligible word.
The module-level interfaces handle empty cases before any enumeration, with the following conventions:

- A classical code of dimension zero returns its block length.
- A quantum problem with no supplied logical rows returns the number of physical qubits.
- A quantum problem with a nonempty set of logical rows that all lie in the excluded span returns zero.

These conventions hold regardless of redundant excluded rows.
They are return conventions for problems with no eligible words, not certificates derived from a witness.

### Caches and special constructions

The code-level methods in [common.py](../common.py) can return cached parameters or use construction-specific formulas without calling this module.
To time an ordinary public distance call that actually recomputes the distance, call `forget_distance()` first.
That call does not remove specialized formulas or equal-sector promises.
Similarly, `method="brute_force"` does not ignore a code's cache of known distances.

Small problems can use exhaustive enumeration even with the default BZ method.
For example, both of the following calls do:

```python
import numpy as np
from qldpc import codes

generators = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)
assert codes.get_distance_classical(generators, cutoff=0) == 2

logicals = np.array([[0, 1, 1]], dtype=int)
stabilizers = np.array([[1, 1, 0]], dtype=int)
assert codes.get_distance_quantum(logicals, stabilizers, homogeneous=True, cutoff=0) == 2
```

### Complexity and limitations

BZ is still a combinatorial search.
A weak certificate can require exponentially many row combinations.
Additional bases cost elimination time, memory for packed generators, and scratch space for prefix XORs.
Neither the work estimate nor a lower candidate count guarantees that a search runs faster on every input.

`_scan_brouwer_zimmermann` runs serially.
Fixed-seed tie-breaking affects which valid bases are tried, not whether a certificate is sound.
Setting `NUMBA_DISABLE_JIT=1` before importing qLDPC runs the Numba-decorated code as ordinary Python, which is useful for debugging; it is not a separate optimized backend.

## 12. Invariants to preserve when modifying the algorithm

- **Preserve the searched quotient.**
  Every row operation on the generators must also be applied to the labels.
- **Keep search coordinates identifiable.**
  Compute pivot coverage before each basis deletes its pivot columns and packs its bits.
- **Use the correct rank term.**
  Partial blocks need the rank-deficiency correction `b_i(t)`; the full-set formula `t+1` does not apply to them.
- **Count overlap exactly.**
  Coverage is the actual maximum multiplicity of a search coordinate across the pivot blocks, not a count of "independent" bases.
- **Advance certificates only after complete work.**
  A cutoff, exception, or interrupted chunk does not complete a layer.
- **Keep bounds in consistent units.**
  General symplectic searches use doubled Hamming weights internally.
- **Prove divisibility; do not infer it from examples.**
  An even-weight basis proves that the whole space is even.
  Higher divisibility requires a separate proof.
- **Share upper bounds without sharing certificates.**
  The parity, coverage, and completed layers of each sector must remain local to that sector.
- **Make regression tests non-vacuous.**
  Verify that an overlapping family was actually selected, and use an independent oracle on a case where enumeration improves the starting witness.

The [BZ test module](brouwer_zimmermann_test.py) contains these checks:

- exhaustive small-word checks of the certificates,
- checks of the logical labels,
- a check that the default policy reduces work on a quasi-cyclic (QC) code, and
- a shortened-BCH quotient whose witnesses from setup are heavier than its true minimum.
  This last case verifies that the search discovers a lighter witness, not only that it certifies an already optimal one.

The repository also requires statement coverage, typing, formatting, lint, and a strict documentation build.
From the repository root, run:

```bash
python -m pytest src/qldpc/codes/distance/ -q
python checks/coverage_.py src/qldpc/codes/distance/
python checks/all_.py
```

## References and further reading

- [core.py](core.py): public signatures and dispatch, re-exported by [the package facade](__init__.py).
- [brouwer_zimmermann.py](brouwer_zimmermann.py): BZ certificates, information sets, and the numerical kernel.
- [brute_force.py](brute_force.py): exact exhaustive enumeration.
- [_binary.py](_binary.py): shared binary linear algebra, packing, and weight functions.
- [brouwer_zimmermann_test.py](brouwer_zimmermann_test.py) and [core_test.py](core_test.py): executable invariants and independent small-instance oracles.
- [common.py](../common.py): code-level caches, CSS sector routing, and subsystem gauge exclusion.
- [Library map](../../../../docs/source/library_map.rst): package-level introduction to exact distances and bounds.
- [Algorithm 994 paper, v2](https://arxiv.org/abs/1603.06757v2): section 2 and Algorithm 1 give the classical disjoint and rank-deficient bound, and discuss per-basis progress.
- [Quantum-code adaptation, v2](https://arxiv.org/abs/2408.10743v2): section 2.4 describes the `Saved_isometry` symplectic-to-Hamming construction used here.
