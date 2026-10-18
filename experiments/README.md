# Experiments

This directory holds research scripts that use `qldpc`.
They are not part of the installed `qldpc` package, and the test suite does not cover them.

| Directory | What it does | Entry points |
| --- | --- | --- |
| [`bivariate_bicycle/`](bivariate_bicycle/) | Brute-force search for bivariate bicycle codes | `run_search.py`, then `collect_results.py` |
| [`quantum_tanner/`](quantum_tanner/) | Randomized search for quantum Tanner codes | `run_randomized_search.py`, then `collect_results.py` and `verify_codes.py` |
| [`lattice_surgery/`](lattice_surgery/) | Logical error rates of lattice-surgery gadgets versus quantum memory | `collect_ler.py` |

Each script documents its method, outputs, and options in its module docstring; run it with `--help` to see them.
Commands below are written relative to the repository root.
Caches, search results, and progress files are saved beside the scripts by default, where Git ignores them.
The aggregate CSV file of `lattice_surgery/collect_ler.py` is instead written to the path given by `--output`.

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
Both scripts accept `--cache-dir` to keep the cache elsewhere, such as in a scratch directory for a quick trial run:

```bash
python experiments/bivariate_bicycle/run_search.py 6 --num-trials 10 --cache-dir /tmp/bbcodes
python experiments/bivariate_bicycle/collect_results.py --num-trials 10 --cache-dir /tmp/bbcodes --output /tmp/bbcodes/codes.csv
```

## Quantum Tanner codes

[`quantum_tanner/`](quantum_tanner/) performs a randomized search over quantum Tanner codes built from small finite groups and small classical base codes.

1. `run_randomized_search.py` samples codes for every group of order at most `--max-order` and every base code in `BASE_CODE_IDS`, and saves each code to a text file in `quantum_tanner/codes/`.
   File names identify the group, base code, and random seed of each code, and codes that already have a file are skipped.
2. `collect_results.py` gathers the parameters of saved codes into the disk cache `quantum_tanner/.code_cache`.
   Each entry maps `(group_order, group_index, base_code_id)` to a list of tuples `(n, k, d, num_trials, stabilizer_weight)`, with one tuple per saved code that has `k > 0`.
3. `verify_codes.py` checks that regenerating every saved code from its seed reproduces that code.

```bash
python experiments/quantum_tanner/run_randomized_search.py --workers 8
python experiments/quantum_tanner/collect_results.py
python experiments/quantum_tanner/verify_codes.py
```

To try the pipeline cheaply first, write a small search to a scratch directory:

```bash
python experiments/quantum_tanner/run_randomized_search.py --max-order 6 --num-samples 2 --num-trials 10 --save-dir /tmp/qtcodes
python experiments/quantum_tanner/collect_results.py --save-dir /tmp/qtcodes --cache-dir /tmp/qtcodes
python experiments/quantum_tanner/verify_codes.py --save-dir /tmp/qtcodes
```

With default settings, the search covers the 18 groups of order at most 10, with up to 100 random codes per group and base code, which amounts to roughly 11,000 saved codes.
Each code requires a randomized distance estimate with `--num-trials` trials, so a full search can take many CPU-hours.
Distances `d` are upper bounds, and `d` is `nan` for codes with no logical qubits.
To read collected results, open the cache with `qldpc.cache.get_disk_cache(".code_cache", cache_dir=CACHE_DIR)`, where `CACHE_DIR` is the `--cache-dir` of `collect_results.py`, and iterate over its keys.
Pass the same `--save-dir` to all three scripts when saving codes to a nondefault directory, and pass `--cache-dir` to `collect_results.py` to keep its cache out of the repository.
Verification requires `codes.QTCode.random` to sample the same code for the same seed, so codes saved with a different version of `qldpc` may fail verification.

## Lattice surgery

[`lattice_surgery/`](lattice_surgery/) samples logical error rates of the experimental lattice-surgery constructions in `qldpc.experimental.surgery`, and of quantum memory baselines, with [Sinter](https://pypi.org/project/sinter/).
`collect_ler.py --preset PRESET` runs one of the presets `bb72`, `bb18`, or `steane-joint`, and writes an aggregate CSV file to `--output`.
Collection is resumable, and some presets take hours to days at publication scale.
See the [lattice-surgery README](lattice_surgery/README.md) for example commands, provenance tracking, and costs.
