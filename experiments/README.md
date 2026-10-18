# Code searches

The scripts in [`bivariate_bicycle/`](bivariate_bicycle/) and [`quantum_tanner/`](quantum_tanner/) search for quantum codes with good parameters.
They are research scripts rather than part of the installed `qldpc` package, and the test suite does not cover them.
Each script documents its method, outputs, and options in its module docstring; run it with `--help` to see them.

Commands below are written relative to the repository root.
Search results and caches are saved beside the scripts, where Git ignores them.

## Bivariate bicycle codes

[`bivariate_bicycle/`](bivariate_bicycle/) performs a brute-force search over the bivariate bicycle codes of [arXiv:2308.07915](https://arxiv.org/abs/2308.07915) with generating polynomials `A = 1 + x + x^ax y^ay` and `B = 1 + y + x^bx y^by`.

1. `run_search.py DIM_X` searches over every torus with width `DIM_X` and height between `--min-order` and `DIM_X`, skipping codes that are equivalent by symmetry.
   It saves code parameters `(n, k, d)` to the disk cache `bivariate_bicycle/.code_cache`, and resumes from that cache if interrupted.
2. `collect_results.py` exports every cached code with `k d^2 / n > 1` to `bivariate_bicycle/codes.csv`.

```bash
python experiments/bivariate_bicycle/run_search.py 12 --workers 8
python experiments/bivariate_bicycle/collect_results.py
```

Code distances are upper bounds, each minimized over `--num-trials` randomized trials.
Before symmetry reduction, each torus height contributes roughly `DIM_X^4` candidate codes, so the cost of a search grows rapidly with `DIM_X`.
`collect_results.py` only exports results computed with its own `--num-trials`, so pass the same value to both scripts.

## Quantum Tanner codes

[`quantum_tanner/`](quantum_tanner/) performs a randomized search over quantum Tanner codes built from small finite groups and small classical base codes.

1. `run_randomized_search.py` samples codes for every group of order at most `--max-order` and every base code in `BASE_CODE_IDS`, and saves each code to a text file in `quantum_tanner/codes/`.
   File names identify the group, base code, and random seed of each code, and codes that already have a file are skipped.
2. `collect_results.py` gathers the parameters of saved codes into the disk cache `quantum_tanner/.code_cache`.
3. `verify_codes.py` checks that regenerating every saved code from its seed reproduces that code.

```bash
python experiments/quantum_tanner/run_randomized_search.py --workers 8
python experiments/quantum_tanner/collect_results.py
python experiments/quantum_tanner/verify_codes.py
```

With default settings, the search covers the 18 groups of order at most 10, with up to 100 random codes per group and base code.
Each code requires a randomized distance estimate with `--num-trials` trials.
Pass the same `--save-dir` to all three scripts when saving codes to a nondefault directory.
Verification requires `codes.QTCode.random` to sample the same code for the same seed, so codes saved with a different version of `qldpc` may fail verification.
