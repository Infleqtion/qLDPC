# qLDPC

This package contains tools for constructing and analyzing [quantum low density parity check (qLDPC) codes](https://errorcorrectionzoo.org/c/qldpc).  At least, that was the original motivation for this package.  In practice, the tools here work just as well with more general error-correcting codes.

## 📦 Installation

This package requires Python>=3.10, and can be installed from the Python Package Index (PyPI) with
```
pip install qldpc
```
If you use macOS, you may need to install `cvxpy` manually by following the instructions [here](https://www.cvxpy.org/install/) before installing qLDPC.

To install a local version of qLDPC from source:
```
git clone git@github.com:Infleqtion/qLDPC.git
pip install -e qLDPC
```
You can also `pip install -e 'qLDPC[dev]'` to additionally install some development tools.

## 🚀 Features

Notable features include:
- `ClassicalCode`: class for representing classical linear error-correcting codes over finite fields.
  - Various pre-defined classical code families.
  - Communication with the [GAP](https://www.gap-system.org/)/[GUAVA](https://www.gap-system.org/Packages/guava.html) package for [even more codes](https://docs.gap-system.org/pkg/guava/doc/chap5.html).
- `QuditCode`: general class for constructing [Galois-qudit codes](https://errorcorrectionzoo.org/c/galois_into_galois).
  - Automatic construction of all SWAP-transversal logical Clifford gates in one code block of a qubit code, with `qldpc.circuits.get_transversal_ops` (see [arXiv:2409.18175](https://arxiv.org/abs/2409.18175); requires [GAP](https://www.gap-system.org/) and [GUAVA](https://www.gap-system.org/Packages/guava.html)).
  - `QuditCode.get_logical_ops`: method to construct a complete basis of nontrivial logical Pauli operators for a `QuditCode`.
  - `QuditCode.concatenate`: method to [concatenate](https://errorcorrectionzoo.org/c/quantum_concatenated) `QuditCode`s in various ways.
- `CSSCode`: subclass of `QuditCode` for the special case of constructing a [quantum CSS code](https://errorcorrectionzoo.org/c/css) out of two mutually compatible `ClassicalCode`s.
  - `CSSCode.get_distance`: method to compute the code distance (i.e., the minimum weight of a nontrivial logical operator) of a `CSSCode`.  Includes options for computing the exact code distance by brute force, as well as an estimate (or upper bound) with the method of [arXiv:2308.07915](https://arxiv.org/abs/2308.07915).
  - Includes options for applying local Hadamard transformations, which is useful for tailoring a `CSSCode` to biased noise (see [arXiv:2202.01702](https://arxiv.org/abs/2202.01702)).
- `TBCode`: class for constructing [two-block quantum codes](https://errorcorrectionzoo.org/c/two_block_quantum).
- `BBCode`: class for constructing the [bivariate bicycle codes](https://errorcorrectionzoo.org/c/quantum_quasi_cyclic) in [arXiv:2308.07915](https://arxiv.org/abs/2308.07915) and [arXiv:2311.16980](https://arxiv.org/abs/2311.16980).
  - Includes methods to identify "toric layouts" of a `BBCode`, in which the code looks like a toric code augmented by some long-distance checks, as in discussed in [arXiv:2308.07915](https://arxiv.org/abs/2308.07915).
- `HGPCode`: class for constructing [hypergraph product codes](https://errorcorrectionzoo.org/c/hypergraph_product) out of two `ClassicalCode`s.
- `LPCode`: class for constructing [lifted product codes](https://errorcorrectionzoo.org/c/lifted_product) out of two protographs (i.e., matrices whose entries are elements of a group algebra).  See [arXiv:2012.04068](https://arxiv.org/abs/2012.04068) and [arXiv:2202.01702](https://arxiv.org/abs/2202.01702).
- `QTCode`: class for constructing [quantum Tanner codes](https://errorcorrectionzoo.org/c/quantum_tanner) out of (a) two symmetric subsets `A` and `B` of a group `G`, and (b) two `ClassicalCode`s with block lengths `|A|` and `|B|`.  See [arXiv:2202.13641](https://arxiv.org/abs/2202.13641) and [arXiv:2206.07571](https://arxiv.org/abs/2206.07571).
  - Random `QTCode`s can be constructed out of a choice of group `G` and one `ClassicalCode` only.
- `abstract.py`: module for basic abstract algebra (groups, algebras, and representations thereof).
  - Various pre-defined groups (mostly borrowed from [SymPy](https://docs.sympy.org/latest/modules/combinatorics/named_groups.html)).
  - Communication with the [GAP](https://www.gap-system.org/) computer algebra system and [GroupNames.org](https://people.maths.bris.ac.uk/~matyd/GroupNames/) for constructing [even more groups](https://docs.gap-system.org/doc/ref/chap50.html).
- `objects.py`: module for constructing helper objects such as Cayley complexes and chain complexes, which are instrumental for the construction of various quantum codes.

## 🤔 Questions and issues

This project technically has a [documentation page](https://qldpc.readthedocs.io/en/latest/), but at the moment the documentation is largely auto-generated from source code and comments that were written to be readable in a text editor.  For now, I recommend looking at the source code (and comments therein) directly, as well as the `examples` directory.  Test files (such as `qldpc/codes/quantum_test.py`) also contain some examples of using the classes and methods in this library.

If you have any questions, feedback, or requests, please [open an issue on GitHub](https://github.com/Infleqtion/qLDPC/issues/new) or email me at [mika.perlin@gmail.com](mailto:mika.perlin@gmail.com)!

## ⚓ Attribution

If you use this software in your work, please cite with:
```
@misc{perlin2023qldpc,
  author = {Perlin, Michael A.},
  title = {{qLDPC}},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Infleqtion/qLDPC}},
}
```
This may require adding `\usepackage{url}` to your LaTeX file header.  Alternatively, you can cite
```
Michael A. Perlin. qLDPC. https://github.com/Infleqtion/qLDPC, 2023.
```
