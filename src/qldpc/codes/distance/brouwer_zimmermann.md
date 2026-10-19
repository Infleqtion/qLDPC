# Brouwer-Zimmermann distance search in qLDPC

## 1. Purpose and scope

The Brouwer-Zimmermann (BZ) algorithm computes the distance of a classical code ([arXiv:1603.06757](https://arxiv.org/abs/1603.06757)), and has been adapted to quantum codes ([arXiv:2408.10743](https://arxiv.org/abs/2408.10743)).
This guide explains how qLDPC computes the exact distance of a binary code using BZ.
It is written for contributors who need to understand, debug, or modify the BZ code in this directory.
It is organized as follows:

- [Section 2](#2-the-brouwer-zimmermann-algorithm) explains the BZ algorithm as published, including its adaptation to quantum codes.
- [Section 3](#3-extensions-to-bz-in-qldpc) explains the extensions that qLDPC adds to the published algorithm.
- [Section 4](#4-implementation-in-qldpc) explains how the code in this directory implements the algorithm and the extensions.
- [Section 5](#5-invariants-to-preserve-when-modifying-the-implementation) lists the invariants that a change to the implementation must preserve.

## 2. The Brouwer-Zimmermann algorithm

This section describes BZ as presented in [arXiv:1603.06757](https://arxiv.org/abs/1603.06757) and [arXiv:2408.10743](https://arxiv.org/abs/2408.10743).

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
Throughout this guide, a **basis** of `V` is a matrix whose rows are linearly independent and span `V`.
Let `r = dim(V)` be the number of in a basis of `V`, and let `N` be its number of columns, which is the number of **coordinates** of a word.
A coordinate at which every word of `V` is zero carries no information, so we assume that there are none.

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

An **information set** is a set `I` of `r` coordinate indices (column indices) such that the columns of the basis indexed by `I` form a rank-`r` matrix.
Row reduction on `I` produces a **systematic basis** `G_I`, whose restriction to `I` is the identity matrix:

$$
G_I[:,I] = I_r.
$$

The columns indexed by `I` are the **pivot columns** of `G_I`.

For a word `v`, write `v|_I` for the restriction of `v` to the coordinates indexed by `I`.
Let `v` be any word in `V`, and let `a` be its coefficient vector in `G_I`, so that `v = aG_I`.
Then `v|_I = a`, because the columns of `G_I` indexed by `I` form the identity matrix:

$$
v|_I = a\,G_I[:,I] = a\,I_r = a.
$$

Hence `wt(v|_I) = wt(a)`:

> For every word `v` in `V`, the Hamming weight `wt(v|_I)` equals the coefficient weight of `v` in `G_I`, which is a lower bound for the Hamming weight `wt(v)`.

**Searching a basis through weight `t`** means examining every word with coefficient weight at most `t` with respect to `G_I`; that is, every XOR of at most `t` rows of `G_I`.
Each word in `V` has exactly one coefficient vector in `G_I`, so a word that is not examined by this search has coefficient weight at least `t+1`.
Since `wt(v|_I)` equals the coefficient weight of `v`, such a word `v` satisfies `wt(v) ≥ wt(v|_I) ≥ t+1`.
[Section 2.5](#25-the-stopping-certificate) uses this consequence to lower-bound the weight of every word that the search has not examined.

### 2.4 Search bases and rank-deficient blocks

BZ searches `m` bases `G_1, ..., G_m` of `V`, which we call the **search bases**.
Each `G_i` is row reduced on a **block** `B_i`, a set of column indices, and the blocks `B_1, ..., B_m` are pairwise disjoint.
We call such a collection of search bases a **disjoint family of search bases**.
These blocks are chosen greedily: `B_1` is an information set, `B_2` is an information set among the columns not in `B_1`, and so on, until the remaining columns no longer contain an information set.
If any columns remain at that point, they form one last block, which has rank less than `r`.

Each block `B_i` that is an information set gives a systematic basis, as in [Section 2.3](#23-systematic-bases-and-information-sets).
Only the last block, `B_m`, can have rank `r_m < r`, and then it is not an information set.
In general, if `B_i` has rank `r_i`, row reducing on `B_i` gives a basis `G_i` of the whole space `V` in which:

- the first `r_i` rows have distinct pivot columns in `B_i`, and
- the remaining `r - r_i` rows are zero on the columns indexed by `B_i`.

Suppose a word `v` has coefficient weight at least `t+1` in `G_i`.
At most `r - r_i` of its selected rows can be zero on `B_i`, so at least `t + 1 - (r - r_i)` of them are pivot rows.
Each pivot row contributes a distinct nonzero coordinate to `v|_{B_i}`, the restriction of `v` to `B_i`, so

$$
\operatorname{wt}(v|_{B_i}) \ge b_i(t) := \max(0,\;t+1-(r-r_i)).
$$

For a full information set (`r_i = r`, `B_i = I`), this is `b_i(t) = t + 1`, matching [Section 2.3](#23-systematic-bases-and-information-sets).

### 2.5 The stopping certificate

Suppose that each of the search bases `G_1, ..., G_m` of [Section 2.4](#24-search-bases-and-rank-deficient-blocks) has been searched through coefficient weight `t`, in the sense of [Section 2.3](#23-systematic-bases-and-information-sets).
Consider an eligible word `v` that was not examined in any of them.
For each `i`, the coefficient weight of `v` in `G_i` is at least `t+1`.
By Sections [2.3](#23-systematic-bases-and-information-sets) and [2.4](#24-search-bases-and-rank-deficient-blocks), `wt(v|_{B_i}) ≥ b_i(t)` for every `i`.

Because the blocks are pairwise disjoint, they share no column indices, so these weights add up:

$$
\operatorname{wt}(v) \ge \sum_i b_i(t) =: L(t).
$$

If all `m` blocks are information sets, `L(t) = m(t+1)`.

`L(t)` is the lower certificate: every eligible word that has not been examined has weight at least `L(t)`.
It makes no claim about the words that have been examined, so a witness that was already found can be lighter than `L(t)`, and `L(t)` can exceed the distance.
When `L(t) ≥ U`, no unexamined eligible word is lighter than `U`, which is enough to stop.

BZ is therefore the following loop:

1. Build the search bases `G_1, ..., G_m` from disjoint blocks, as in [Section 2.4](#24-search-bases-and-rank-deficient-blocks).
2. Find the lightest eligible row among the rows of the search bases, which gives an initial `U`.
3. For `t = 1, 2, ...`, examine every selection of `t` rows in each search basis, lowering `U` whenever a lighter eligible word is found.
   Then compute `L(t)`, and stop if `L(t) ≥ U`.

The search always terminates, because after `t = r` every nonzero word has been examined.
The result is then the exact distance.
Algorithm 1 of [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) gives this loop and this certificate for disjoint and rank-deficient blocks.

The cost of this loop is measured in **candidate/set evaluations**.
One candidate/set evaluation is the examination of one selection of rows in one basis.
Searching the `m` search bases through coefficient weight `t` therefore costs `m` times the number of selections of at most `t` rows, which is `m * sum_{w=1}^{t} binom(r, w)`.

### 2.6 Excluding stabilizers

For a quantum code, the lightest word in `V` may lie in `S`.
Such a word is not eligible, so it must not lower `U`.
Section 2.1 of [arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2) therefore checks whether a candidate word of weight below `U` lies in `S` before updating `U`.
The lower certificate of [Section 2.5](#25-the-stopping-certificate) is unchanged, because it holds for all unexamined words, eligible or not.

### 2.7 Pauli weight as Hamming weight

An operator that contains only X (or only Z) Pauli operators on `n` qubits is a binary vector of length `n`, and its Hamming weight is its physical weight.

A general Pauli operator is a symplectic row `[x | z]` of length `2n`.
Its physical weight is the number of qubits on which `x` or `z` is nonzero, which differs from the Hamming weight of `[x | z]`.

To search for the minimum physical weight with a Hamming-weight algorithm, Section 2.4 of [arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2) uses the `Saved_isometry` construction, which follows earlier work of White and Grassl.
It maps each symplectic row to a binary word of length `3n`:

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

BZ therefore searches words with `3n` coordinates, and the quantum distance is half of the minimum Hamming weight that BZ finds.
Because `T` is linear and injective, it maps the spaces `S ⊆ V` to spaces `T(S) ⊆ T(V)`, and a word of `T(V)` is eligible exactly when its preimage lies outside `S`.

## 3. Extensions to BZ in qLDPC

qLDPC extends the published BZ algorithm in three ways, none of which affects the exactness of the result:

- [Section 3.1](#31-overlapping-search-bases) strengthens the lower certificate by allowing for search bases with overlapping blocks.
- [Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple) strengthens the lower certificate by rounding it up when the weight of every eligible word is a multiple of some integer.
- [Section 3.3](#33-sharing-witnesses-between-css-sectors) tightens the upper bound by sharing witnesses between the two sectors of a CSS code.

[Section 3.4](#34-worked-example) illustrates the first two extensions with an example.

### 3.1 Overlapping search bases

#### Motivation

For a disjoint family of search bases, [Section 2.5](#25-the-stopping-certificate) gives the lower certificate as a sum over the search bases:

$$
L(t) = \sum_i b_i(t).
$$

The term `b_i(t)` for the search basis `G_i` is at most `t+1`, and it equals `t+1` exactly when the block `B_i` is an information set ([Section 2.4](#24-search-bases-and-rank-deficient-blocks)).
So `L(t)` grows with the number of search bases whose blocks are information sets.

An information set has `r` column indices, and the blocks of a disjoint family share no column indices.
A disjoint family therefore has at most `floor(N / r)` blocks that are information sets.
When `N / r` is not an integer, the remaining coordinates cannot form another information set, so they form at most one rank-deficient block, whose term `b_i(t)` is smaller.

For example, take `N = 64` and `r = 41`.
A disjoint family has one block that is an information set, with term `b_1(t) = t+1`.
The other 23 coordinates form a rank-deficient block whose term `b_2(t)` is zero for `t ≤ 17`.
Hence `L(t) = t+1` for `t ≤ 17`, and the 23 other coordinates do not help the certificate.

An **overlapping family of search bases** is a collection of search bases whose blocks may share column indices.
Because its blocks need not be disjoint, such a family is not limited to `floor(N / r)` blocks that are information sets.
However, a coordinate that lies in several blocks is counted in several of the terms `b_i(t)`.
The certificate derived below accounts for this duplication by dividing the sum of the terms by the maximum coverage `C`, defined below.
For `m` blocks that are information sets, this yields a certificate `L(t)` of about `(m/C)(t+1)`, which exceeds the `k(t+1)` of a disjoint family with `k` information-set blocks whenever `m/C > k`.

#### The certificate

Let `C` be the **maximum coverage** of an overlapping family, which is the largest number of blocks `B_i` that contain any single column index.
Suppose that each search basis of the family has been searched through coefficient weight `t`, and let `v` be an eligible word that was not examined in any of them.
As in [Section 2.5](#25-the-stopping-certificate), `wt(v|_{B_i}) ≥ b_i(t)` for every `i`.
Each nonzero coordinate `j` of `v` is counted in `wt(v|_{B_i})` for every block `B_i` that contains `j`, and there are at most `C` such blocks.
Summing over blocks therefore gives

$$
\sum_i b_i(t) \le \sum_i \operatorname{wt}(v|_{B_i}) \le C\operatorname{wt}(v).
$$

Because `wt(v)` is an integer, this gives the lower certificate

$$
L(t) =
\left\lceil \frac{\sum_i b_i(t)}{C} \right\rceil.
$$

If all `m` blocks are information sets, it becomes

$$
L(t) = \left\lceil \frac{m(t+1)}{C} \right\rceil.
$$

A disjoint family is the special case `C = 1`, which recovers [Section 2.5](#25-the-stopping-certificate).
The formula holds whether or not each block is an information set, through the terms `b_i(t)`.

[arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) mentions in its Section 2 that overlapping information sets require a different formula for the certificate, citing Grassl, but does not give one.
The coverage-divided certificate above is derived for qLDPC.

#### A small example

Take the code generated by the rows `101` and `011`, whose distance is two.
It has `N = 3` coordinates and `r = 2`, so a disjoint family contains at most `floor(3/2) = 1` search basis whose block is an information set.
An overlapping family can use all three pairs of column indices, `{0,1}`, `{0,2}`, and `{1,2}`, each of which is an information set.
Then `m = 3`, and each column index lies in two of the blocks, so `C = 2`.
After all three search bases have been searched through coefficient weight `t = 1`, the formula gives `L(1) = ceil(3*2/2) = 3`.
This exceeds the distance, which is allowed because `L(t)` is a statement only about words that have not been examined.
Every row has weight two, so `U = 2 ≤ L(1)`, and the search stops with distance two.

### 3.2 Rounding the certificate to a proven weight multiple

Suppose that every eligible word has a weight that is a multiple of an integer `D`.
An unexamined eligible word then has a weight that is at least `L(t)` and a multiple of `D`, so it has a weight of at least `D * ceil(L(t)/D)`, the smallest multiple of `D` that is at least `L(t)`.
The certificate can therefore be rounded up to that value.

In qLDPC, `D` comes from two sources:

- For general Pauli problems, the map of [Section 2.7](#27-pauli-weight-as-hamming-weight) doubles every weight, which gives `D = 2`.
- For any problem, qLDPC checks whether every row of the search basis has even Hamming weight.
  The XOR of two even-weight words has even weight, so if all rows are even, every word in the space is even.
  When this holds, `D = 2` is combined with any existing divisor by taking the least common multiple.

Parity is the only property checked in this way, because evenness of the rows carries over to all of their sums.
A larger divisor does not: the rows `11110` and `01111` each have weight four, but their XOR `10001` has weight two.

### 3.3 Sharing witnesses between CSS sectors

The distance of a CSS code is `min(d_x, d_z)`, where `d_x` and `d_z` are the distances of its X and Z sectors.
qLDPC runs one BZ search for each sector, and advances the searches in round-robin order, one layer at a time.
A witness found in either sector is a nontrivial logical operator of the code, so it lowers the upper bound `U` used by both searches.

Each search keeps its own lower certificate, because a certificate bounds only the unexamined words of its own sector.
The sectors can also differ in their search bases, coverage, and weight divisor.
For example, suppose that the X sector finds a witness of weight seven, so that `U = 7` for both sectors.
If the Z sector contains only even-weight words, it can have no witness of weight seven.
Its search still stops as soon as its own certificate reaches `U = 7`.
At that point it has shown only that `d_z ≥ 7`, and hence `d_z ≥ 8`, without determining the exact value of `d_z`.
A joint search therefore determines `min(d_x, d_z)`, while the individual sector distances may remain unknown.

### 3.4 Worked example

This example uses one CSS sector of the QC64 code from the test suite ([brouwer_zimmermann_test.py](brouwer_zimmermann_test.py)).
The sector has these properties:

- It has 64 coordinates and dimension `r = 41`.
- The lightest eligible row of its basis has weight eight, so initially `U = 8`.
- Every word in it has even weight, so `D = 2` ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)).

A disjoint family can contain only `floor(64/41) = 1` search basis whose block is an information set.
The leftover columns hold a rank-deficient block, which contributes nothing to `L(t)` for `t ≤ 17`.
Hence `L(t) = t + 1` for `t ≤ 17`, and `L(6) = 7` rounds up to eight.
The search stops after layer six, so it enumerates layers two through six (layer one consists of the rows already checked).

An overlapping family of `m = 3` search bases, each with an information set as its block, can be chosen with maximum coverage `C = 2`.
Completing layer four gives

$$
L(4)=\left\lceil\frac{3(4+1)}{2}\right\rceil=8,
$$

which reaches eight even before rounding to an even weight.
The search stops after layer four.

| Family | Last layer | Candidate/set evaluations in layers two and above |
| --- | ---: | ---: |
| Disjoint, one search basis | 6 | $\sum_{w=2}^{6}\binom{41}{w} = 5{,}358{,}536$ |
| Overlapping, three search bases, `C = 2` | 4 | $3\sum_{w=2}^{4}\binom{41}{w} = 338{,}250$ |

The overlapping family needs fewer evaluations because it avoids layers five and six.
The counts cover only enumeration; building and packing the search bases costs extra.

## 4. Implementation in qLDPC

### 4.1 Overview of a search

A BZ distance computation has two phases.
The first phase is **preparation**, which `_prepare_brouwer_zimmermann_search` performs for each search.
Preparation either returns a final distance without any enumeration of layers, or returns a search that is ready to scan layers.
The second phase is the **scan**, which enumerates layers `t = 2, 3, ...` of a prepared search ([Section 4.4](#44-packed-enumeration)).

Preparation proceeds as follows, and it returns at the first step whose stopping condition holds:

1. Build an independent basis of `V`, attach quotient labels to its rows ([Section 4.2](#42-testing-membership-in-s-with-quotient-labels)), and, for general Pauli operators, transform the rows to Hamming coordinates ([Section 2.7](#27-pauli-weight-as-hamming-weight)).
   The public distance functions in `core.py` do this before they call preparation.
2. Take the lightest eligible row of the basis as the initial upper bound `U`.
   - If `U` is at most the user-provided `cutoff` ([Section 4.6](#46-early-exit-with-a-cutoff)), return `U`.
   - If `_exhaustive_is_cheaper` approves the dimension and length of the basis ([Section 4.5](#45-exhaustive-enumeration-as-an-alternative)), return the result of exhaustive enumeration.
3. Build a disjoint family of search bases ([Section 4.3](#43-building-search-bases)), and check their eligible rows, lowering `U` if one of them is lighter.
   These rows are all the words of coefficient weight one, so layer one is now complete.
   If `U` is at most the `cutoff`, return `U`.
4. Compute the lower certificate `L(1)` of the disjoint family, which is valid because only layer one has been searched so far.
   If `L(1)` is at least `U`, return `U`.
5. Try to replace the disjoint family by an overlapping family of search bases ([Section 3.1](#31-overlapping-search-bases)).
   If a replacement is found, check the eligible rows of every search basis of the replacement family, which completes layer one for the new family, and lower `U` if one of them is lighter.
   If `U` is at most the `cutoff`, return `U`.
   Then compute `L(1)` for the replacement family, and if it is at least `U`, return `U`.
6. Compare the estimated BZ work with the cost of exhaustive enumeration ([Section 4.5](#45-exhaustive-enumeration-as-an-alternative)).
   If exhaustive enumeration is estimated to be cheaper, return its result.
7. Pack the search bases into `uint64` words ([Section 4.4](#44-packed-enumeration)) and return a search that is ready to scan layers, starting from layer two.

Preparation evaluates the lower certificate only at `t = 1`.
During the scan, it is evaluated again at `t = 2, 3, ...`, each time only after layer `t` is complete in every search basis.
The distance computation can also return before any preparation if `get_distance_quantum` enumerates the supplied rows exhaustively ([Section 4.5](#45-exhaustive-enumeration-as-an-alternative)).

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

### 4.2 Testing membership in `S` with quotient labels

The membership test of [Section 2.6](#26-excluding-stabilizers) is performed with quotient labels, which are a qLDPC implementation choice.
For the problem of [Section 2.1](#21-the-problem) with nested spaces `S ⊆ V`, `_get_nested_code_basis` builds a basis of `V` that is adapted to `S`.
Let `s = dim(S)` and `q = dim(V/S)`, so that `r = s + q`.
The basis has the form

```text
G = [ s independent rows spanning S                       ]
    [ q independent rows, whose cosets form a basis of V/S ]
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
For a classical search there is no excluded subspace, so every nonempty selection is eligible.

Every row operation on `G` must be applied identically to `M`.
If an invertible row transformation replaces `G` by `G' = EG`, the labels must become `M' = EM`.
Then `aM'` is still the label of the word `aG'`.
Omitting this step would either accept stabilizers as witnesses or reject nontrivial logical operators.
`_row_reduce_binary` performs these paired transformations.

### 4.3 Building search bases

#### Disjoint family of search bases

`_get_information_set_generators` builds a disjoint family of search bases, using a bounded, deterministic heuristic:

1. Row reduce on the nonzero columns, in their natural order.
   The set of pivot columns found becomes one block.
   Remove those columns from the remaining columns, and repeat.
2. Stop at the first block of rank `r_i < r` (the rank-deficient block of [Section 2.4](#24-search-bases-and-rank-deficient-blocks)), or when no columns remain.
   A block consists only of the pivot columns of its row reduction, so the non-pivot columns that remain after a rank-deficient reduction belong to no block.

3. If the above steps do not yield `floor(n_active / r)` full information sets, where `n_active` is the number of nonzero columns of the basis, retry with up to eight fixed-seed random column orders.
   Keep the disjoint family of search bases with the lexicographically largest tuple of ranks.

A search basis whose block is rank deficient (a **partial** search basis) costs as much to enumerate as a full one, but its term `b_i(t)` is zero for small `t`.
`_select_brouwer_zimmermann_information_sets` therefore removes some partial search bases from the family, so that they are not enumerated or packed.
It works as follows:

1. Using only the search bases whose blocks are information sets, find the smallest `t_f` for which their certificate `L(t_f)`, after rounding, is at least `U`.
   Without any partial search basis, the search would stop after layer `t_f`.
2. Keep a partial search basis only if its term can be nonzero for some `t < t_f`.
   Its rank deficiency is `d = r - r_i`, and `b_i(t) = max(0, t+1-d)` is zero for `t < d`.
   So the search basis is kept if `d < t_f`, and removed if `d ≥ t_f`.

A removed search basis cannot raise the certificate before the information-set search bases alone already reach `U`, so removing it does not change the result, and it saves enumeration work.
If the family has no search basis whose block is an information set, nothing is removed.

#### Choosing an overlapping family of search bases

`_get_overlapping_information_sets` starts from the search bases of the disjoint family whose blocks are information sets.
It returns either a replacement overlapping family of search bases, all of whose blocks are information sets, together with the actual maximum coverage of that family, or `None` to keep the disjoint family of search bases.
Which overlapping family to choose is decided heuristically.
The certificate for the chosen overlapping family uses only verified ranks and the actual maximum coverage, so it is sound regardless of the heuristic.

Here `n_active` is the number of nonzero columns of the basis.
The implementation counts only nonzero columns, because an all-zero column can never be a pivot, so `n_active` plays the role of `N` in [Section 3.1](#31-overlapping-search-bases).
This count includes the columns that existing blocks already use.
Coverage is counted using the original column indices, before each basis removes its pivot columns for packing.

Construction of the overlapping family has four stages.

1. **Skip unnecessary work.**
   Return `None` if the disjoint family of search bases is already cheap to search, already uses every nonzero column, or is already at the size limit.
2. **Rule out impossible savings.**
   A family of `m` search bases whose blocks are all information sets has `m*r` pivot columns in total, counted with multiplicity.
   Spread over `n_active` columns, they force some column to be covered at least `ceil(m*r/n_active)` times.
   Using that value as `C` gives the most optimistic work estimate for each `m`.
   If even the best optimistic estimate does not improve on the estimated work of the current disjoint family of search bases, return `None`.
3. **Build coverage-balanced search bases.**
   Repeat the following attempt until the budget below is used up.
   Order the nonzero columns so that the least-used columns come first, where the use of a column is the number of search bases already in the family whose block contains it, and break ties with a fixed random seed.
   Row reduce the basis on that column order, applying the same row operations to the labels.
   The pivot columns found form a new block, which is an information set.
   If the same set of pivot columns is already the block of a search basis in the family, discard the new search basis, because a repeated block adds work and raises the coverage without adding new information.
   Otherwise, add it to the family.
   A discarded attempt still costs a row reduction, so it consumes one attempt from the budget.
4. **Keep the cheapest overlapping family.**
   After each search basis is added, estimate the work for the family built so far, and remember the family with the lowest estimate.
   Stop when the estimate reaches the optimistic estimate of stage 2, or the family reaches the size limit.

The policy constants are:

- An overlapping family has at most eight search bases.
- If construction starts with `m_0` full sets, it makes at most `2*(8 - m_0)` attempts.
- Stage 1 skips families of search bases whose estimated work is below 100,000 candidate/set evaluations.

The choice of these constants affects only performance; the certificate holds for any values.
The selection depends only on the basis and its labels, the current upper bound `U`, the fixed constants listed here, and the disjoint family of search bases and weight divisor from preparation.
The upper bound can come from a row witness, from the other sector of a CSS code ([Section 3.3](#33-sharing-witnesses-between-css-sectors)), or from a distance that is already known.
A different valid upper bound can change which overlapping family is selected, and the certificate is sound for whichever family results.

#### The work estimate

`_get_brouwer_zimmermann_work` estimates the total number of candidate/set evaluations, from layer one onward, needed to certify `U`.
It pessimistically assumes that no lighter witness appears before the certificate reaches `U`.
Let `m` be the number of search bases, and let `t_*` be the smallest layer after which the lower certificate reaches `U`.
This is the certificate of [Section 3.1](#31-overlapping-search-bases) after rounding to a multiple of the weight divisor ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)), for the ranks and maximum coverage of the family being estimated.
The estimate is

$$
W = m\sum_{w=1}^{t_*} \binom{r}{w}.
$$

The sum starts at `w = 1`, so it includes the systematic rows.
The comparison with exhaustive enumeration in [Section 4.5](#45-exhaustive-enumeration-as-an-alternative) happens after layer one has been checked, so it slightly overestimates the remaining BZ work.
Actual runtime also depends on basis construction, the width of packed words, and early improvements of the witness.

### 4.4 Packed enumeration

#### Reconstructing weights without storing pivots

Preparation removes the pivot columns of each basis before packing the remaining columns into `uint64` words.
This deletion is a qLDPC implementation choice; packing bits into machine words is also used by [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2).
The known contribution of the removed columns to a word's weight is easy to restore:

- For a full information set, a support of `w` rows contributes exactly `w` removed pivot bits.
- For a partial block of rank `r_i`, the contribution is the number of selected row indices below `r_i`.

The scan adds this contribution to the population count of the packed XOR of the remaining columns.
The sum is the Hamming weight of the word.

#### Enumerating supports and reusing XOR prefixes

`_scan_brouwer_zimmermann` describes a candidate by its **support**: the selected row indices in increasing order.
Reusing partial XOR sums in this way follows the saved-additions idea of [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2).
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

The scan is compiled with Numba and runs in chunks.
Each call to the compiled scan returns after a bounded number of complete supports, when the cutoff is reached, or when the layer is finished.
Within one `advance()` call, the support indices and XOR prefixes persist across these calls.
`_BrouwerZimmermannSearch.advance` updates the lower certificate and increments `next_weight` only after the **whole layer is complete in each search basis**.
A cutoff, an exception, or an interruption leaves the certificate unchanged.
After an interruption, a new `advance()` call restarts the unfinished layer from its first support, because the support indices and prefixes were local to the earlier call.

`_BrouwerZimmermannSearch` stores the following state:

| State | Meaning |
| --- | --- |
| `generators`, `labels` | Packed bases, with pivot columns deleted, and their quotient labels |
| `ranks` | Rank of each basis's block |
| `widths`, `label_widths` | Number of packed words that are valid for the generators and for the labels of each search basis |
| `best` | Lightest witness known, which is the current upper bound |
| `lower_bound` | Lower certificate from the layers completed so far |
| `next_weight` | Next coefficient weight to search |
| `cutoff` | Cutoff that ends the search early ([Section 4.6](#46-early-exit-with-a-cutoff)) |
| `batch_size` | Maximum number of supports visited by one call to the compiled scan |
| `weight_divisor` | Proven divisor of every candidate Hamming weight ([Section 3.2](#32-rounding-the-certificate-to-a-proven-weight-multiple)) |
| `max_coverage` | Maximum number of blocks containing any column index ([Section 3.1](#31-overlapping-search-bases)) |

A search is finished when its cutoff is reached, its lower certificate reaches `best`, or every coefficient weight has been enumerated.

### 4.5 Exhaustive enumeration as an alternative

For small or low-dimensional problems, the setup cost of BZ exceeds the cost of enumerating every word.
Exhaustive enumeration can therefore be chosen at three points:

- Before building an adapted basis, `get_distance_quantum` can enumerate directly over the supplied rows.
  It does so when `_exhaustive_is_cheaper` approves the number of supplied rows (stabilizers and logical operators together) and those rows are linearly independent.
  Dependent rows take the general path through the adapted basis instead.
- During preparation, `_exhaustive_is_cheaper` can choose exhaustive enumeration when the setup cost of information sets would dominate.
- After the information sets are built, `_brute_force_is_cheaper` compares the estimated BZ work from [Section 4.3](#the-work-estimate) with the number of eligible words, `(2**q - 1) * 2**(r-q)`, where `q` is the number of logical rows.

Neither [arXiv:1603.06757](https://arxiv.org/abs/1603.06757v2) nor [arXiv:2408.10743](https://arxiv.org/abs/2408.10743v2) describes switching to exhaustive enumeration; it is a qLDPC choice.
The exhaustive implementation in [brute_force.py](brute_force.py) enumerates words in vectorized, packed blocks and updates them using Gray-code steps.
This enumeration does not use the BZ certificate.

Callers do not need to supply independent generators.
Classical inputs are first reduced to an independent basis, and quantum exhaustive enumeration uses either the adapted basis from [Section 4.2](#42-testing-membership-in-s-with-quotient-labels) or, in the direct case above, supplied rows that were verified to be independent.
The packed enumerators therefore always receive independent bases, prepared or verified in [core.py](core.py) or [brouwer_zimmermann.py](brouwer_zimmermann.py).

### 4.6 Early exit with a cutoff

The distance functions in `core.py` accept a `cutoff`, which is not part of the published algorithm.
`get_distance_classical` and `get_distance_quantum` default to `cutoff=1`, while `get_distance_css_brouwer_zimmermann` requires its caller to provide one.

With a positive `cutoff`, the search stops as soon as `U` is at most the cutoff.
The result is then an upper bound that is at or below the cutoff, and it can exceed the exact distance.
The default of one still returns the exact distance for a valid nontrivial binary problem, because a positive distance cannot be less than one.
A cutoff of zero or less never triggers an early exit, so it requires a certified result.

Preparation checks `U` against the cutoff after the initial row witness, after the rows of the disjoint family, and after the rows of a replacement overlapping family.
In a joint CSS search, a sector's scan also returns immediately once its upper bound reaches the cutoff.

For non-homogeneous Pauli input on the BZ route, the cutoff is doubled before it is compared with Hamming weights, because of the map of [Section 2.7](#27-pauli-weight-as-hamming-weight), and the final result is halved.
The direct exhaustive route and `method="brute_force"` instead use native symplectic weights and the original cutoff.

## 5. Invariants to preserve when modifying the implementation

- **Preserve the searched quotient.**
  Every row operation on the generators must also be applied to the labels.
- **Keep coordinates identifiable.**
  Compute pivot coverage before each basis deletes its pivot columns and packs its bits.
- **Use the correct rank term.**
  Partial blocks contribute `b_i(t)`, which accounts for their rank deficiency, rather than `t+1`.
- **Count overlap exactly.**
  Coverage is the actual maximum number of blocks that contain a single column index.
- **Advance certificates only after complete work.**
  A cutoff, exception, or interrupted chunk leaves its layer incomplete.
- **Search every basis that a certificate uses.**
  A certificate may use a search basis only through the layers that have been completed in that same basis, including layer one of a replacement overlapping family.
- **Keep bounds in consistent units.**
  The non-homogeneous BZ route uses doubled Hamming weights internally, with a doubled cutoff and a halved result.
  The exhaustive route uses native symplectic weights and the original cutoff.
- **Prove divisibility for the whole space.**
  An even-weight basis proves that the whole space is even.
  Higher divisibility requires a separate proof.
- **Share upper bounds without sharing certificates.**
  The parity, coverage, and completed layers of each sector must remain local to that sector.
- **Make regression tests non-vacuous.**
  Verify that an overlapping family of search bases was actually selected, and use an independent oracle on a case where enumeration improves the starting witness.

## References

- [arXiv:1603.06757v2](https://arxiv.org/abs/1603.06757v2) (Algorithm 1): Section 2 and Algorithm 1 give the classical disjoint and rank-deficient bound.
- [arXiv:2408.10743v2](https://arxiv.org/abs/2408.10743v2): Section 2.4 describes the `Saved_isometry` symplectic-to-Hamming construction.
- [core.py](core.py), [brouwer_zimmermann.py](brouwer_zimmermann.py), [brute_force.py](brute_force.py), and [_binary.py](_binary.py): the implementation.
- [brouwer_zimmermann_test.py](brouwer_zimmermann_test.py) and [core_test.py](core_test.py): executable invariants and independent small-instance oracles.
