# qLDPC

This package contains tools for constructing and analyzing [quantum low density partity check (qLDPC) codes](https://errorcorrectionzoo.org/c/qldpc).

## 📦 Installation

This package requires Python>=3.10, and can be installed from PyPI with
```
pip install qldpc
```
To install a local version from source:
```
git clone git@github.com:Infleqtion/qLDPC.git
pip install -e qLDPC
```
You can also `pip install -e 'qLDPC[dev]'` to additionally install some development tools.

## 🚀 Features

Notable features include:
- `abstract.py`: module for basic abstract algebra (groups, algebras, and representations thereof).
- `BitCode`: class for representing {classical, linear, binary} error-correcting codes.
- `CSSCode`: general class for constructing [quantum CSS codes](https://errorcorrectionzoo.org/c/css) out of two classical `BitCode`s.
  - Includes options for applying local Pauli transformations (i.e., Pauli deformations of the code), which is useful for tailoring a `CSSCode` to biased noise (see [arXiv:2202.01702](https://arxiv.org/abs/2202.01702)).
  - `CSSCode.get_logical_ops`: method (from [arXiv:0903.5256](https://arxiv.org/abs/0903.5256)) to construct a basis of nontrivial logical operators for a `CSSCode`.
  - `CSSCode.get_distance`: method to compute the code distance (i.e., the minimum weight of a nontrivial logical operator).  Includes options for computing a lower bound (determined by the distances of the underlying `BitCode`s), an upper bound (with the method of [arXiv:2308.07915](https://arxiv.org/abs/2308.07915)), and the exact code distance (with an integer linear program, which has exponential complexity).
- `GBCode`: class for constructing [generalized bicycle codes](https://errorcorrectionzoo.org/c/generalized_bicycle), as described in [arXiv:1904.02703](https://arxiv.org/abs/1904.02703).
- `QCCode`: class for constructing the [quasi-cyclic codes](https://errorcorrectionzoo.org/c/quantum_quasi_cyclic) in [arXiv:2308.07915](https://arxiv.org/abs/2308.07915).
- `HGPCode`: class for constructing [hypergraph product codes](https://errorcorrectionzoo.org/c/hypergraph_product) out of two classical `BitCode`s.  Follows the conventions of [arXiv:2202.01702](https://arxiv.org/abs/2202.01702).
- `LPCode`: class for constructing [lifted product codes](https://errorcorrectionzoo.org/c/lifted_product) out of two protographs (i.e., matrices whose entries are elements of a group algebra).  See [arXiv:2012.04068](https://arxiv.org/abs/2012.04068) and [arXiv:2202.01702](https://arxiv.org/abs/2202.01702).
- `QTCode`: class for constructing [quantum Tanner codes](https://errorcorrectionzoo.org/c/quantum_tanner) out of (a) two symmetric subsets `A` and `B` of a group `G`, and (b) two classical `BitCode`s with block lengths `|A|` and `|B|`.  See [arXiv:2202.13641](https://arxiv.org/abs/2202.13641) and [arXiv:2206.07571](https://arxiv.org/abs/2206.07571).

## 🤔 Questions and issues

If this project gains interest and traction, I'll add a documentation webpage and material to help users get started quickly.  I am also planning to write a paper that presents and explains this project.  In the meantime, you can explore the documentation and explanations in the source code.  `qldpc/codes_test.py` contains some examples of using the classes and methods described above.

In the meantime, if you have any questions or requests, please open an issue!

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
