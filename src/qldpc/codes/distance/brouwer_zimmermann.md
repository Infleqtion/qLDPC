# Brouwer-Zimmermann distance search in qLDPC

## 1. Purpose and scope

This guide explains how qLDPC computes the exact distance of a binary code using the Brouwer-Zimmermann (BZ) algorithm.
It is written for contributors who need to understand, debug, or modify the BZ code in this directory.
It has three goals:

- Explain the BZ algorithm itself (section 2).
- Explain how qLDPC implements it (section 3).
- Explain the improvements that qLDPC adds to the algorithm as published (section 4).

Section 5 lists the invariants that a change to the implementation must preserve.

The implementation draws on two papers and a derivation in this guide:

- [Algorithm 994](https://arxiv.org/abs/1603.06757) describes a BZ implementation for classical codes.
  qLDPC follows its stopping criterion for disjoint and rank-deficient information sets.
- A [paper that adapts BZ to quantum codes](https://arxiv.org/abs/2408.10743) reduces symplectic (Pauli) weight to Hamming weight.
  qLDPC follows this reduction.
- The stopping criterion for overlapping information sets (section 4.1) is derived in this guide.

The public interfaces for binary distances live in [core.py](core.py).
They dispatch either to [brouwer_zimmermann.py](brouwer_zimmermann.py), which implements BZ, or to [brute_force.py](brute_force.py), which enumerates words exhaustively.

## 2. The Brouwer-Zimmermann algorithm

### 2.1 The problem

BZ works over the binary field, where addition is bitwise XOR.
`wt(v)` denotes the Hamming weight of a binary word `v`, meaning its number of nonzero coordinates.

For a classical code with generator matrix `G`, the distance is the smallest weight of a nonzero codeword:

$$
d = \min_{v \in V,\;v \ne 0} \operatorname{wt}(v),
\qquad V = \operatorname{rowspan}(G).
$$

For a quantum code, the distance is the smallest weight of a nontrivial logical operator.
Let `S` be the span of the stabilizers (or, for a subsystem code, the stabilizers and gauge operators), and let `V` be the span of `S` together with the logical operators.
Words in `S` act trivially on the encoded information, so the nontrivial logical operators are the words in `V` that are outside `S`.
If the code encodes at least one logical qubit (`V ≠ S`), its distance is

$$
d = \min_{v \in V \setminus S} \operatorname{wt}(v).
$$

The classical and quantum definitions are instances of one problem: given binary spaces `S ⊆ V` with `V ≠ S`, find the lightest word in `V \ S`.
For a classical code, `V` is the code and `S = {0}`.

A word in `V \ S` is called **eligible**, because it can determine the distance.
For a classical code, the eligible words are the nonzero codewords.
For a quantum code, they are the nontrivial logical operators.

### 2.2 The idea: upper bounds and lower certificates

Every codeword is a sum of rows of a basis of `V`.
Throughout this guide, a **basis** is a matrix whose rows are linearly independent and span `V`.
Let `r = dim(V)` be its number of rows, and let `N` be its number of columns, which is the number of **coordinates** of a word.

For a basis `G` of `V`, a **coefficient vector** `a` is a binary vector of length `r` that selects rows, and `aG` is the XOR of the selected rows.
The **coefficient weight** of `aG` is `wt(a)`, the number of rows selected.
Since the rows of `G` are independent, each word of `V` has exactly one coefficient vector `a` in a given basis.
A **layer** is the set of all words with a given coefficient weight `w`, taken over all choices of `w` rows.

BZ keeps track of two quantities:

- A **witness** is an eligible word whose weight is known.
  The lightest known witness has weight `U`, which is an **upper bound** on the distance.
  The distance is at most `U`, but a lighter eligible word might exist.
- A **lower certificate** `L` is a proven statement that every eligible word that has not yet been examined has weight at least `L`.

Once `L ≥ U`, no unexamined word can be lighter than the best witness.
Then `U` is the distance, and the search stops.

The search examines words layer by layer, starting with coefficient weight one.
Examining a layer can only lower `U`.
Completing layers also strengthens `L`, because the words that remain unexamined all need more rows.
The remaining question is why unexamined words must be heavy, which is the role of information sets.

### 2.3 Systematic bases and information sets

An **information set** is a set `I` of `r` coordinates on which the rows of the basis restrict to a rank-`r` matrix.
Row reduction on `I` produces a **systematic basis** `G_I`, whose restriction to `I` is the identity matrix:

$$
G_I[:,I] = I_r.
$$

The coordinates in `I` are the **pivot columns** of `G_I`.

For a coefficient vector `a`, the restriction of `aG_I` to `I` is exactly `a`.
Selecting `w` rows therefore gives a word with exactly `w` nonzero entries on `I`.
The word may also have nonzero entries outside `I`.
This gives the fact on which BZ's stopping certificate rests:

> If every selection of at most `t` rows of `G_I` has been examined, then every eligible word with at most `t` nonzero coordinates in `I` has been examined.

Different information sets give different bases of the same space.
The same coefficient vector in two such bases usually describes two different words.
Some words appear in more than one basis.
The work of a search is therefore measured in **candidate/set evaluations**, which count a word once for each basis in which it is examined.

### 2.4 Rank-deficient blocks

Suppose several bases are built one after another, each from a block of coordinates disjoint from the earlier blocks.
Once the available coordinates run low, the final block may have rank `r_i < r`, so it is not an information set.
Row reduction on such a block gives a basis of the whole space in which:

- the first `r_i` rows have distinct pivot columns in the block, and
- the remaining `r - r_i` rows are zero on the block.

Suppose a word needs at least `t+1` basis rows.
At most `r - r_i` of those rows can be zero on the block, so at least `t + 1 - (r - r_i)` of them are pivot rows.
The word therefore has at least

$$
b_i(t) = \max(0,\;t+1-(r-r_i))
$$

nonzero coordinates in the block.
For a full information set (`r_i = r`), this is `b_i(t) = t + 1`.

### 2.5 The stopping certificate

Suppose that every basis has been fully searched through coefficient weight `t`, meaning that every selection of at most `t` rows has been examined in every basis.
Consider an eligible word `v` that has not been examined.
In every basis `i`, the word `v` needs at least `t+1` rows, so it has at least `b_i(t)` nonzero coordinates in block `i`.

If the blocks are disjoint, they share no coordinates, so these counts add up:

$$
\operatorname{wt}(v) \ge \sum_i b_i(t) =: L(t).
$$

For `m` full information sets, `L(t) = m(t+1)`.

`L(t)` is the lower certificate.
It bounds the weight of the words that have not been examined.
Witnesses that were already found can be lighter than `L(t)`.
When `L(t) ≥ U`, no unexamined eligible word is lighter than `U`, which is enough to stop.

BZ is therefore the following loop:

1. Build several disjoint blocks, and a systematic basis for each.
2. Find the lightest eligible row among all the bases, which gives an initial `U`.
3. For `t = 1, 2, ...`, examine every selection of `t` rows in every basis, lowering `U` whenever a lighter eligible word is found.
   Then compute `L(t)`, and stop if `L(t) ≥ U`.

The search always terminates, because after `t = r` every nonzero word has been examined.
Algorithm 1 of the [Algorithm 994 paper](https://arxiv.org/abs/1603.06757v2) gives this bound for disjoint and rank-deficient blocks.

## 3. Implementation in qLDPC

### 3.1 Overview of a search

A call to qLDPC's BZ distance method may return through exhaustive enumeration before any BZ scan runs.
When a scan does run, the call proceeds as follows:

1. For a quantum problem, `get_distance_quantum` can first choose exhaustive enumeration directly, if the input is small and independent.
2. Otherwise, the search builds an independent basis of `V`, attaches quotient labels to its rows (section 3.2), and, for general Pauli operators, transforms the rows to Hamming coordinates (section 3.3).
3. The lightest eligible row of that basis gives the initial upper bound `U`.
   Preparation may stop here, either because `U` is at most the `cutoff` (section 3.6) or because exhaustive enumeration is estimated to be cheaper (section 4.4).
4. The search builds disjoint systematic bases (section 3.4) and checks their eligible rows.
   These rows are all the coefficient-weight-one words, so layer one is complete.
5. Preparation may stop again if the lower certificate already reaches `U`, or if the work estimate favors exhaustive enumeration.
   It may also replace the disjoint family by an overlapping family (section 4.1).
6. Otherwise, the search packs the bases into `uint64` words and scans the layers starting from layer two (section 3.5).
   It updates the lower certificate only after a whole layer is complete in every basis.

The implementation is organized as follows:

| Responsibility | Implementation |
| --- | --- |
| Binary classical, quantum, and joint CSS interfaces | [core.py](core.py): `get_distance_classical`, `get_distance_quantum`, `get_distance_css_brouwer_zimmermann` |
| Independent bases and quotient labels | [_binary.py](_binary.py): `_row_reduce_binary`, `_get_nested_code_basis` |
| Disjoint and overlapping systematic bases | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_get_information_set_generators`, `_get_overlapping_information_sets` |
| Search preparation and the lower certificate | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_prepare_brouwer_zimmermann_search`, `_get_brouwer_zimmermann_lower_bound` |
| State and progress of one search | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_BrouwerZimmermannSearch` |
| Packed enumeration of candidate words | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_scan_brouwer_zimmermann` |
| Sharing upper bounds between CSS sectors | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_get_distance_brouwer_zimmermann_many` |
| Exhaustive alternatives and their cost estimates | [brouwer_zimmermann.py](brouwer_zimmermann.py): `_exhaustive_is_cheaper`, `_brute_force_is_cheaper`; [brute_force.py](brute_force.py): `_get_distance_quantum_brute_force` |

### 3.2 Quantum problems: quotient labels

For the problem of section 2.1 with nested spaces `S ⊆ V`, `_get_nested_code_basis` builds a basis of `V` that is adapted to `S`.
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
Omitting this step would either accept stabilizers as witnesses or reject nontrivial logical operators.
`_row_reduce_binary` performs these paired transformations.

The code methods in [common.py](../common.py) supply generators for `S` and for the logical operators, so this module only receives the nested spaces `S ⊆ V`.

`_assert_binary` rejects nonbinary values and field types before they are converted to binary storage arrays, so the search only ever sees binary data.

### 3.3 Quantum problems: Pauli weight as Hamming weight

An operator that contains only X (or only Z) Pauli operators on `n` qubits is a binary vector of length `n`, and its Hamming weight is its physical weight.

A general Pauli operator is a symplectic row `[x | z]` of length `2n`.
Its physical weight is the number of qubits on which `x` or `z` is nonzero, which differs from the Hamming weight of `[x | z]`.

`_symplectic_to_hamming` fixes this using the `Saved_isometry` construction from [section 2.4 of the quantum-code adaptation](https://arxiv.org/abs/2408.10743v2):

$$
T(x,z) = (x \mid z \mid x+z).
$$

On each qubit, the pair `(x_i, z_i)` is `(0,0)` for the identity, and `(1,0)`, `(0,1)`, or `(1,1)` for a nontrivial Pauli operator.
The identity contributes Hamming weight zero to `T`, and each nontrivial Pauli operator contributes weight exactly two.
Hence

$$
\operatorname{wt}_{H}(T(v)) = 2\operatorname{wt}_{P}(v),
$$

where `wt_P` is the physical Pauli weight.

BZ therefore runs on words of Hamming length `3n`.
It doubles any `cutoff` before comparing, and halves the final result.
The labels from section 3.2 remain valid because `T` is linear and injective.
Every Hamming weight is even, but this does **not** mean that physical Pauli weights are even.

From this point on, a **coordinate** means a coordinate of the binary words that BZ searches.
For general Pauli problems, there are `3n` such coordinates after the transformation.

The direct exhaustive route for quantum problems can use symplectic weight directly, without this transformation.

### 3.4 Building disjoint information sets

`_get_information_set_generators` builds a family of disjoint blocks, using a bounded, deterministic heuristic:

1. Row reduce on the nonzero columns, in their natural order.
   Each pivot set becomes one block.
   Remove its pivots from the remaining columns, and repeat.
2. Stop at the first block of rank `r_i < r` (the rank-deficient block of section 2.4), or when no columns remain.
3. If this does not yield `floor(n_active / r)` full information sets, where `n_active` is the number of nonzero columns of the basis, retry with up to eight fixed-seed random column orders.
   Keep the family with the lexicographically largest tuple of ranks.

`_select_brouwer_zimmermann_information_sets` then discards any partial block that cannot help when the full blocks alone would already raise the lower certificate to `U`.
(A block with rank `r_i` has `b_i(t) = 0` until `t + 1 > r - r_i`.)

### 3.5 Packed enumeration

#### Reconstructing weights without storing pivots

Preparation removes the pivot columns of each basis before packing the remaining columns into `uint64` words.
The known contribution of the removed columns to a word's weight is easy to restore:

- For a full information set, a support of `w` rows contributes exactly `w` removed pivot bits.
- For a partial block of rank `r_i`, the contribution is the number of selected row indices below `r_i`.

The scan adds this contribution to the population count of the packed XOR of the remaining columns.
The sum is the Hamming weight of the word.

The arrays of generators and labels are zero-padded to a common number of words across all bases.
Each basis keeps its true generator width and label width, so padding changes neither weights nor eligibility decisions.
Weights are accumulated in sufficiently wide integer types.

#### Enumerating supports and reusing XOR prefixes

`_scan_brouwer_zimmermann` describes a candidate by its **support**: the selected row indices in increasing order.
For coefficient weight `w`, it visits the same supports as `itertools.combinations(range(r), w)`.
It maintains the invariant

```text
prefixes[j, set] = XOR of the first j selected rows in that set's packed (pivot-deleted) basis.
```

Extending a support by one row then costs one row XOR per basis and packed word, instead of recomputing the whole sum.

The scan stores the current upper bound `U` in the variable `best`.
For each completed support, and in each basis, it does the following:

1. Add the known pivot contribution to the population count of the packed XOR.
2. If the resulting weight is at least `best`, discard the candidate, whatever its label is.
3. Otherwise, XOR the selected row labels and accept the candidate only if the result is nonzero.
4. If the candidate is accepted, lower `best` to its weight.
   Then continue with the next support.

#### Chunks, layers, and interruption

The Numba kernel is compiled lazily, cached, and bounds checked.
Each call to the kernel returns after one of three events:

- a bounded number of complete supports has been visited,
- the cutoff has been reached, or
- the layer is finished.

The bound is `2**min(block_size, 15)` supports per kernel call.
Every support is evaluated in every retained basis, so one kernel call performs up to that many supports times the number of bases candidate/set evaluations.

Within one `advance()` call, the support indices, tree depth, and XOR prefixes persist across kernel calls.
`_BrouwerZimmermannSearch.advance` updates the lower certificate and increments `next_weight` only after the **whole layer is complete in every retained basis**.
A cutoff, an exception, or an interruption leaves the certificate unchanged.
After an interruption, a new `advance()` call restarts the unfinished layer from its first support, because the support indices and prefixes were local to the earlier call.

`_BrouwerZimmermannSearch` stores the following state:

| State | Meaning |
| --- | --- |
| `generators`, `labels` | Packed bases, with pivot columns deleted, and their quotient labels |
| `ranks`, `widths`, `label_widths` | Pivot rank and true packed widths of each basis |
| `best` | Lightest witness known, which is the current upper bound |
| `lower_bound` | Lower certificate from the layers completed so far |
| `next_weight` | Next coefficient weight to search |
| `cutoff` | Cutoff that ends the search early (section 3.6) |
| `batch_size` | Maximum number of supports per kernel call |
| `weight_divisor` | Proven divisor of every candidate Hamming weight (section 4.2) |
| `max_coverage` | Maximum number of blocks containing any coordinate (section 4.1) |

A search is finished when its cutoff is reached, its lower certificate reaches `best`, or every coefficient weight has been enumerated.

### 3.6 Exactness, cutoff, and empty problems

Without an early return caused by a cutoff, a BZ search ends in one of two ways: its certificate reaches `U`, or it exhausts the eligible space.
Either way, the result is the exact distance.

With a positive `cutoff`, the search may instead stop as soon as `U` is at most the cutoff.
The result is then an upper bound that is at or below the cutoff, and it can exceed the exact distance.
The default `cutoff` of one is still exact for a valid nontrivial binary problem, because a positive distance cannot be less than one.
Use `cutoff=0` to require a fully certified result.

The layer-by-layer argument assumes a nonempty basis and at least one eligible word.
`get_distance_classical` and `get_distance_quantum` in [core.py](core.py) handle empty cases before any enumeration, with the following conventions:

- A classical code of dimension zero returns its block length.
- A quantum problem with no supplied logical rows returns the number of physical qubits.
- A quantum problem with a nonempty set of logical rows that all lie in the excluded span returns zero.

These three return values are fixed conventions for problems that have no eligible words.

## 4. Improvements in qLDPC

### 4.1 Overlapping information sets

Disjoint blocks are limited by the number of coordinates: at most `floor(n_active / r)` full information sets fit without sharing a coordinate.
Allowing blocks to overlap permits more bases, at the price of a weaker certificate.

#### The certificate

Let `C` be the **maximum coverage**: the largest number of blocks that contain any single coordinate.
When summing `b_i(t)` over blocks, each nonzero coordinate of an unexamined word `v` is counted at most `C` times, so

$$
\sum_i b_i(t) \le C\operatorname{wt}(v).
$$

This gives the lower certificate

$$
L(t) =
\left\lceil \frac{\sum_i b_i(t)}{C} \right\rceil.
$$

With `m` full information sets, it becomes

$$
L(t) = \left\lceil \frac{m(t+1)}{C} \right\rceil.
$$

Disjoint blocks are the special case `C = 1`, which recovers section 2.5.
`_get_brouwer_zimmermann_lower_bound` evaluates this formula with exact integer arithmetic, including the rank-deficient terms `b_i(t)`.
The selector that builds overlapping families, however, proposes only full information sets.

The [Algorithm 994 paper](https://arxiv.org/abs/1603.06757v2) refers to Grassl's work for overlapping sets, without giving a bound.
The coverage-divided bound above is derived for qLDPC.

As a small example, take the code generated by `101` and `011`, whose distance is two.
Each of the three pairs of coordinates is a full information set, so `m = 3`, and each coordinate lies in two of them, so `C = 2`.
After examining the single rows in all three bases (`t = 1`), the formula gives `ceil(3*2/2) = 3`.
This means that every word of weight below three has already been examined, so no unexamined word can improve on the weight-two witness.

#### Choosing the family

`_get_overlapping_information_sets` starts from the full disjoint blocks that were retained.
It returns either a replacement family of full information sets together with the family's actual maximum coverage, or `None` to keep the disjoint family.
Which family to choose is decided heuristically.
The certificate for the chosen family uses only verified ranks and actual coordinate counts, so it is sound regardless of the heuristic.

Here `n_active` is the number of nonzero columns of the basis, as in section 3.4.
This count includes the columns that existing blocks already use.
Coverage is counted using the column numbers before each basis removes its pivots for packing.

Family construction has four stages.

1. **Skip unnecessary work.**
   Return `None` if the disjoint family is already cheap to search, already uses every active column, or is already at the size limit.
2. **Rule out impossible savings.**
   A family of `m` full sets has `m*r` pivot coordinates in total.
   Spread over `n_active` columns, they force some column to be covered at least `ceil(m*r/n_active)` times.
   Using that value as `C` gives the most optimistic work estimate for each `m`.
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

The choice of these constants affects only performance; the certificate holds for any values.
The selection depends only on the basis and its labels, the current upper bound `U`, the fixed constants above, and the disjoint family and weight divisor from preparation.
The upper bound can come from a row witness, from the other sector of a CSS code (section 4.5), or from a distance that is already known.
A different valid upper bound can change which family is selected, and the certificate is sound for whichever family results.

#### The work estimate

`_get_brouwer_zimmermann_work` estimates the number of candidate/set evaluations needed to certify `U`.
It pessimistically assumes that no lighter witness appears before the certificate reaches `U`.
Let `m` be the number of bases, and let `t_*` be the smallest layer after which `L` reaches `U`.
The estimate is

$$
W = m\sum_{w=1}^{t_*} \binom{r}{w}.
$$

The sum starts at `w = 1`, so it includes the systematic rows.
Actual runtime also depends on basis construction, the width of packed words, and early improvements of the witness.

#### Worked example

This example uses the QC64 input from `test_brouwer_zimmermann_overlap_qc` in [brouwer_zimmermann_test.py](brouwer_zimmermann_test.py).
One CSS sector of that code has these properties:

- It has 64 coordinates and dimension `r = 41`.
- Its setup rows already provide a witness of weight eight (`U = 8`).
- Every word in it has even weight, which section 4.2 uses.

With **one disjoint information set**, `L(t) = t + 1`.
The certificate `L(6) = 7` is rounded up to eight by the even-weight divisor, so the search enumerates layers two through six after checking the systematic rows.

With **three overlapping information sets**, a coverage-balanced family can have `m = 3` full sets with maximum coverage `C = 2`.
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
The counts cover only the enumeration; constructing and packing the bases costs extra.

### 4.2 Rounding the certificate to a proven weight multiple

Suppose every candidate has a weight that is a multiple of some integer `D`.
Then `L` can be rounded up to `D * ceil(L/D)`, the smallest multiple of `D` that is at least `L`.
In qLDPC, `D` comes from two sources:

- The symplectic transformation of section 3.3 gives `D = 2`.
- Preparation also checks whether every row of the search basis has even Hamming weight.
  The sum of two even-weight words has even weight, so if all basis rows are even, every word in the space is even.
  When this holds, the implementation combines `D = 2` with any existing divisor by taking the least common multiple.

Divisibility must hold for every word in the space, which is stronger than holding for each basis row.
For example, `11110` and `01111` each have weight four, but their XOR `10001` has weight two.
The even-weight check above is the only divisibility check that qLDPC performs on the basis, and it is a qLDPC addition.

### 4.3 One completed layer for all bases

The bases could in principle be searched to different depths `t_i`.
The same argument as in section 4.1 then gives

$$
L = \left\lceil \frac{\sum_i b_i(t_i)}{C} \right\rceil.
$$

Algorithm 994 enumerates the combinations of each basis in turn, and the paper's discussion notes that the bound can increase after every basis.
Its Algorithm 1 nevertheless updates the bound only after all bases complete a layer.
qLDPC does the same, with one difference in scheduling: it examines each support in every basis before moving to the next support.
This lets one set of prefix XORs be reused across all bases (section 3.5), and it means that qLDPC tracks a single completed coefficient weight `t` shared by all bases.

Separate depths `t_i` would allow a lower certificate to be reached sooner.
In the worked example of section 4.1, finishing layer four in one basis while the other two stop after layer three gives `ceil((5+4+4)/2) = 7`.
Rounding up to an even weight gives eight, and this requires only 135,710 candidate/set evaluations after the systematic rows, compared with 338,250 in the table.
qLDPC's shared schedule does not exploit this, so the table counts the work of qLDPC's actual scan.

### 4.4 Exhaustive enumeration as an alternative

For small or low-dimensional problems, the setup cost of BZ exceeds the cost of enumerating every word.
Exhaustive enumeration can therefore be chosen at three points:

- Before building an adapted basis, `get_distance_quantum` can enumerate a small independent quantum input directly.
- During preparation, `_exhaustive_is_cheaper` can choose exhaustive enumeration when the setup cost of information sets would dominate.
- After the information sets are built, `_brute_force_is_cheaper` compares the estimated BZ work from section 4.1 with the number of eligible words, `(2**q - 1) * 2**(r-q)`, where `q` is the number of logical rows.

The exhaustive implementation in [brute_force.py](brute_force.py) enumerates words in vectorized, packed blocks and updates them using Gray-code steps.
This enumeration does not use the BZ certificate.

Neither method requires its caller to supply independent generators.
Classical inputs are first reduced to an independent basis, quantum brute force uses the adapted basis from section 3.2, and the direct quantum shortcut checks independence before skipping that preparation.
The packed enumerators therefore always receive independent bases, prepared or verified in [core.py](core.py) or [brouwer_zimmermann.py](brouwer_zimmermann.py).

### 4.5 Sharing witnesses between CSS sectors

The distance of a CSS code is `min(d_x, d_z)`, where `d_x` and `d_z` are the distances of the X and Z sectors.
`_get_distance_brouwer_zimmermann_many` prepares one search per sector and advances them in round-robin order, one layer at a time.
A witness found in either sector lowers the upper bound `U` used by both.

Each sector keeps its own information sets, coverage, weight divisor, and lower certificate.
The sectors can differ in these properties.
For example, a witness of weight seven from one sector caps the other sector's upper bound at seven, even if the other sector contains only even-weight words and so can never have a witness of weight seven.
That sector only has to show that it cannot beat the shared witness, so its own exact distance may remain undetermined.
A joint search therefore determines `min(d_x, d_z)`, while the individual sector distances may remain unknown.

## 5. Invariants to preserve when modifying the implementation

- **Preserve the searched quotient.**
  Every row operation on the generators must also be applied to the labels.
- **Keep coordinates identifiable.**
  Compute pivot coverage before each basis deletes its pivot columns and packs its bits.
- **Use the correct rank term.**
  Partial blocks contribute `b_i(t)`, which accounts for their rank deficiency, rather than `t+1`.
- **Count overlap exactly.**
  Coverage is the actual maximum multiplicity of a coordinate across the blocks.
- **Advance certificates only after complete work.**
  A cutoff, exception, or interrupted chunk leaves its layer incomplete.
- **Keep bounds in consistent units.**
  General symplectic searches use doubled Hamming weights internally.
- **Prove divisibility for the whole space.**
  An even-weight basis proves that the whole space is even.
  Higher divisibility requires a separate proof.
- **Share upper bounds without sharing certificates.**
  The parity, coverage, and completed layers of each sector must remain local to that sector.
- **Make regression tests non-vacuous.**
  Verify that an overlapping family was actually selected, and use an independent oracle on a case where enumeration improves the starting witness.

## References

- [Algorithm 994 paper, v2](https://arxiv.org/abs/1603.06757v2): section 2 and Algorithm 1 give the classical disjoint and rank-deficient bound.
- [Quantum-code adaptation, v2](https://arxiv.org/abs/2408.10743v2): section 2.4 describes the `Saved_isometry` symplectic-to-Hamming construction.
- [core.py](core.py), [brouwer_zimmermann.py](brouwer_zimmermann.py), [brute_force.py](brute_force.py), and [_binary.py](_binary.py): the implementation.
- [brouwer_zimmermann_test.py](brouwer_zimmermann_test.py) and [core_test.py](core_test.py): executable invariants and independent small-instance oracles.
