# Agent and contributor guide

This guide is for people and agents changing qLDPC.
For a human-facing explanation of the data model and package relationships, start with the [library map](docs/source/library_map.rst).
For exact signatures and per-object literature, use the source docstrings or the generated [API reference](https://qldpc.readthedocs.io/autoapi/index.html).

## What to trust when documents disagree

For what the code does, use this order:

1. Current implementation and its invariant tests.
2. Public docstrings and the `__all__` lists in package `__init__.py` files.
3. The [library map](docs/source/library_map.rst) and executable [examples](examples/).
4. Historical discussion, issues, or review notes.

For how code should be written, this guide takes precedence over existing code: code that breaks a rule here is debt to fix, not a precedent to copy.
Re-derive behavior from the current branch before documenting or changing it.
Do not copy transient project history, machine-specific paths, or local-session instructions into production files.

## Orientation

The source tree under `src/qldpc/` mirrors the library: each package and module starts with a docstring whose first line summarizes it, and the [library map](docs/source/library_map.rst) explains how the packages relate.
Canonical executable notebooks live in [`examples/`](examples/), Sphinx pages and the library map in [`docs/source/`](docs/source/), local wrappers for CI's quality gates in [`checks/`](checks/), and untested research scripts in [`experiments/`](experiments/).

The main code hierarchy is:

```text
AbstractCode
|- ClassicalCode
`- QuditCode
   `- CSSCode
```

The methods in `codes/common.py` share cached and mutable state for standard form, logical and gauge operators, parameters, and code transformations.
Do not split them into mixins merely to reduce the file length.
Each decoder backend defines its builder and its specification helper (with `decoder_spec` or `observable_decoder_spec`) beside its implementation, under `decoders/external/` or `decoders/custom/`.
Plumbing that several modules share lives in private modules; helpers that one file uses are underscore names in that file.

## Public API and private code

- A name is public if it is broadly useful to researchers and engineers, and private if it is plumbing that makes qLDPC work.
  Where a name is used, such as in other modules or in notebooks, does not decide this.
- Plain names defined in public modules (no underscore anywhere in the import path) are public.
  Keep them working, and retire a released public name through a tested shim rather than breaking its import (see [Deprecations](#deprecations)).
- [`src/qldpc/__init__.py`](src/qldpc/__init__.py) exports subpackages rather than flattening their classes and functions.
- Each stable subpackage lists its public classes and functions (or, for `qldpc` and `qldpc.external`, its public submodules) in the `__all__` of its `__init__.py`.
  Decoder specification helpers such as `bp_osd` are listed only in `qldpc.decoders.__all__`.
  `qldpc.decoders.construction` has no `__all__`: `qldpc.decoders` re-exports its user-facing names, and the helpers for backend authors (`decoder_spec`, `observable_decoder_spec`) are public at `qldpc.decoders.construction.specs`.
  Type aliases and type variables used in public signatures are public without being listed.
  Keep `__all__` complete and preserve these import paths when moving implementation code.
- AutoAPI lists every public symbol except the generated decoder specification helpers, which [`decoders.rst`](docs/source/decoders.rst) documents instead.
  Other human-written docs therefore explain what packages do and show representative tasks rather than catalogue classes.
- A private module has an underscore somewhere in its import path, and is private as a whole.
  It uses plain names for definitions that other modules use, and underscore names for helpers that only its own file uses.
- A module-level underscore name is used only in its own file and that file's `*_test.py`.
  A class's underscore attributes and methods may be used anywhere in the package that defines the class, but not outside it.
  The only exception is [`_deprecations/decoders.py`](src/qldpc/_deprecations/decoders.py), which imports private decoder builders so that deprecated APIs pass options to them unchecked.
- Public objects are defined in public modules.
  A public module may use a private module's implementation, but never re-exports a private module's object as public API.
- Everything under [`src/qldpc/experimental/`](src/qldpc/experimental/) is explicitly unstable, can change without a deprecation period, and is exempt from the private-name rules above.
  Do not infer that these weaker rules apply elsewhere.

## Code style and organization

- Every source file outside `experimental/` starts with `# SPDX-License-Identifier: Apache-2.0`, a blank line, and a module docstring whose first line summarizes the module.
- Docstrings follow the Google convention, and lines are at most 100 columns long.
- Python comments and docstrings use ordinary paragraphs: separate sentences with two spaces and fill lines to the 100-column limit.
- Keep one prose sentence per physical source line in `.rst`, `.md`, and Markdown notebook cells.
  Put `First sentence.` and `Second sentence.` on separate source lines; leave code blocks, equations, and table syntax intact.
- Preserve a module's existing section headings and organization when adding code, and add a section when new functionality does not fit an existing one.
- Start each section with a line of 100 `#` characters followed by a one-line `# title` comment.
- Put important user-facing APIs first within modules and sections, and keep related public entry points together.
- Place file-internal helpers at the bottom of the section that they serve, so that sections group code by topic rather than by visibility.
  Reserve a final helper section for helpers that several sections share.
  Private constants and types may sit near the imports, and helpers that code uses at import time must precede that use.
- In private modules, put plain names (used by other modules) before underscore names (used only within the file) within each section.
- Within a class, put public methods first (for decoders, the decode methods) and private helper methods after them.

## Core invariants

### Fields and arrays

- Code arithmetic happens in a `galois.FieldArray`.
- Extension-field storage integers are not ordinary integers modulo the field order.
  Do not use `% field.order` as field coercion; construct or preserve values through the field class.
- Preserve the concrete field when copying or transforming arrays.
  Code equality and compatibility often require the same field class, not merely arrays with equal integer views.
- A `RingArray` has one base `GroupRing`.
  NumPy operations must reject arrays from incompatible rings, including arrays supplied through keyword arguments such as `out=`.

### Codes and Pauli conventions

- A classical parity-check matrix `H` defines words satisfying `H @ word == 0`.
- A general quantum check is a symplectic row `[X | Z]` of even length.
  Use [`qldpc.math.symplectic_conjugate`](src/qldpc/math.py) rather than hand-writing sign and half-order conventions.
- A CSS code stores X-type and Z-type checks separately.
  Commuting stabilizer checks satisfy `H_x @ H_z.T == 0` in the code's field.
- For a subsystem code, constructor check rows generate the gauge group and need not commute.
  Stabilizers come from the center.
  Do not decode a subsystem code as though all gauge generators were stabilizers.
- `CSSCode(..., promise_equal_distance_xz=True)` is trusted, not verified.
  A false promise can make distance results wrong and dependent on which sector was computed first.
- Built-in families may cache construction- or literature-supplied parameters, so test them against independent oracles (see [Tests](#tests)).
- [`codes/distance/`](src/qldpc/codes/distance/) is explicitly binary.
  Use field-aware code methods and oracles for nonbinary constructions.

### Decoders

- Build decoders from typed specifications, as in `decoders.bp_osd(...).build(pcm_or_dem)` or `decoders.mwpm(...).build_observable_decoder(dem)`.
- Keep error decoders (`decode_errors`, with `decode` as an alias) distinct from observable decoders (`decode_observables`).
  Code that consumes a user-supplied error decoder coerces it with `as_error_decoder` from `decoders/_common.py` and calls `decode_errors`.
- A method that decodes a matrix it constructs itself must reject prebuilt decoders with `reject_prebuilt_decoder` from `decoders/construction/_resolution.py`.
- Code-capacity estimators resolve their `decoder=`, `decoder_x=`, and `decoder_z=` inputs with `get_code_capacity_decoder` from [`codes/_code_capacity.py`](src/qldpc/codes/_code_capacity.py).
  It always yields an observable decoder: on binary codes, specifications that predict observables natively build it from a DEM; otherwise inferred errors are projected onto logical predictions.
- Keep `resolve_decoder` in `decoders/construction/_resolution.py` specific to error decoders.
  Dispatch on explicit capabilities (`compile_decoder_for_dem`, `decode_observables`, the `ErrorDecoder` protocol), never on output length.
- Only decoders that declare erasure support may append an erasure flag.
  They append that flag as the last entry of each inferred error; unsupported decoders must reject `add_erasure_bit=True`.
- Detector-error-model decomposition indices and remaps must remain valid after cancellation and simplification.
  Test malformed components, not only happy-path Stim models.
- Sliding-window decoders infer time from detector coordinates by a heuristic described in [`decoders/sinter/window.py`](src/qldpc/decoders/sinter/window.py).
  Do not change that fallback; pass an explicit `detector_to_time` mapping for models with other layouts.

### Circuits

- Circuit and tableau constructors are qubit-only and should use the existing `restrict_to_qubits` guard.
- `get_encoding_circuit` constructs a valid encoder but is not fault-tolerant.
  Do not present it as a fault-tolerant state-preparation procedure.
- Keep data, check, reference, and ancilla roles explicit through `QubitIDs` and bookkeeping types.
  Do not infer roles from coincident integer indices.
- qLDPC memory-circuit detectors use coordinates such as `(round, 0, check_index)`.
  A later monotone coordinate can enumerate checks rather than time.
- Noise-model operation immunity and qubit immunity are separate controls; preserve both.

### External systems and caching

- GAP helpers may start subprocesses, read standard input, use the clipboard, access the network, or clone missing packages.
  Keep these side effects explicit in the docstrings of the public entry points that can trigger them, and in error messages.
- Expensive reusable computations should use `qldpc.cache.use_disk_cache(cache_name)`.
  Do not hide errors by returning a default value that looks successful.

## Tests

- Every module needs its own co-located `*_test.py`: the coverage gate requires each module to be fully covered by its own test file, and does not gate a module without one.
  Reach a hard-to-reach branch with a focused test rather than a broad scenario.
- Tests run with network sockets disabled.
  Mock only external boundaries such as web, subprocess, clipboard, prompt, and optional-package paths, or a module's own thin wrappers around them, never the logic under test, and do not add a live-service test.
- Disk caches are bypassed while pytest is imported, so tests must not depend on cache persistence.
- Pytest turns every `DeprecationWarning` into an error, so tests use the modern API and test deprecated paths with `pytest.warns(DeprecationWarning, ...)` (see [Deprecations](#deprecations)).
- A test should guard behavior that matters: results, invariants, documented errors, and contracts between modules.
  Do not pin implementation choices such as internal call order, cache layouts, or incidental error-message wording.
  A test may call a private helper of the module that it tests to check the helper's contract or reach coverage, and may set private state of objects to construct a case, but should not assert on private state.
- Keep each test minimal: use the smallest code, field, or circuit that exercises the behavior, do not parametrize over cases that take the same path, and delete tests implied by others.
- Prefer independent oracles (rank, commutation, row-space identities, or brute force on small instances) over values that the code under test computed or cached.
  Checking a stored value against an independent expected value, such as one from the literature, is not tautological; deriving the expected value from the code under test is.
- Keep tests that reproduce results from the literature, and cite the source in the test.
- Tests run in random order with a per-run random seed (pytest-randomly), so they must not depend on order.
  Seed random inputs from `pytestconfig.getoption("randomly_seed")` for reproducibility, and make sure that the assertions hold for every seed; otherwise use a fixed seed.
- Check that a new or rewritten test fails when the behavior it guards is broken.

## Common change recipes

### Add or change a code construction

1. Put a classical family in [`codes/classical.py`](src/qldpc/codes/classical.py) and a quantum or CSS family in [`codes/quantum.py`](src/qldpc/codes/quantum.py), next to its closest base or sibling.
2. Re-export public names from [`codes/__init__.py`](src/qldpc/codes/__init__.py).
3. Add a linked literature reference and document the field, subsystem, distance, and validation assumptions in the class docstring.
4. Add co-located tests with independent invariants: rank-derived dimension, CSS orthogonality, symplectic commutation, row-space identities, known small distances, or exact cross-family equivalence.

### Change code-core algebra

1. Read the cached-state interactions in [`codes/common.py`](src/qldpc/codes/common.py) before changing standard form, logicals, stabilizers, gauges, dimension, or distance.
2. Test both CSS and non-CSS paths, stabilizer and subsystem paths, `k=0`, and at least one odd characteristic when signs matter.
3. Preserve public imports and invalidate or transfer cached values deliberately.

### Add or adapt a decoder

Follow the [adding a decoder guide](docs/source/adding_decoders.rst).
In addition, do not change generic resolution for an ordinary backend, and decide and test batch behavior, nonbinary support, detector-error-model support, and erasure signaling explicitly, with direct syndrome and error reproductions.

### Change a circuit workflow

1. Keep matrix, qubit-role, measurement-record, detector-record, and Stim-circuit bookkeeping in sync.
2. Test noiseless detector determinism and observable behavior before adding noise.
3. Exercise repeated rounds and seam behavior; a locally valid circuit fragment can still become wrong when initialization, QEC cycles, and readout are joined.
4. State computational and fault-tolerance limitations in the relevant function or class docstring.

### Change an external integration

1. Separate parsing/encoding logic from process, network, clipboard, and prompt mechanics.
2. Test both the callable-tool path and the absent/manual/error paths without live resources.
3. For finite fields, round-trip values through the external system's actual element encoding; do not assume integer storage conventions match.
4. Report nonzero return codes, standard error, malformed output, and missing data clearly.

### Add documentation or an example

1. Put explanations that span packages in [`docs/source/library_map.rst`](docs/source/library_map.rst), exact API behavior in docstrings, and executable workflows in [`examples/`](examples/).
2. The notebooks under `docs/source/examples/` are links to the canonical files under `examples/`.
   Edit the canonical notebook, not the documentation-tree link.
3. Keep README and the library map selective.
4. When a public limitation changes, update the relevant function or class docstring and every README or guide that repeats it in the same pull request.
5. Run the strict documentation build before considering the change complete.

## Deprecations

Deprecated APIs remain supported, but the library and its documentation are written as if they do not exist.
Keep deprecated code in one place, so that it is easy to find and to remove.

- All deprecated code and APIs live under the private package [`qldpc/_deprecations/`](src/qldpc/_deprecations/).
  [`core.py`](src/qldpc/_deprecations/core.py) holds the registry of renamed and moved names, `warn_deprecated`, `defunct`, and the deprecated methods and DEFUNCT stubs that `install_deprecated_methods` attaches to their classes; its module docstring explains the mechanics.
  [`decoders.py`](src/qldpc/_deprecations/decoders.py) holds the deprecated decoder API and its translation into decoder specifications.
- Only the following deprecation code lives outside `qldpc/_deprecations/`:
  1. Each module that still serves a deprecated name ends with this identical marked stub, which looks the name up in the registry:

     ```python
     if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

         def __getattr__(name: str) -> object:
             return _deprecations.get_deprecated_attribute(__name__, name)
     ```

  2. Deprecated arguments, modes, and calling conventions of functions that are not themselves deprecated are handled inline, as small blocks that call `_deprecations.warn_deprecated` or raise `_deprecations.defunct(...)`.
     Substantial translation logic, such as that for deprecated decoder keywords, belongs in `qldpc/_deprecations/`, and the inline block calls into it.
  3. [`qldpc/__init__.py`](src/qldpc/__init__.py) calls `_deprecations.install_deprecated_methods()` once, after importing all subpackages.
- Add a shim only for a name that was public in a released version, at its released path: listed in a package `__all__`, or used in released examples or docs.
  Names added since the last release may change without a shim.
- Register a renamed or moved name, or a moved submodule, in `ALIASES`, which warns on access.
  The warning names the replacement if it is public, and otherwise says that the name will be removed from the public API.
  Use `SILENT_ALIASES` instead for deprecated functions that warn when called.
- No `__all__` lists a deprecated name, so star imports do not warn.
  Type checkers must reject deprecated names, so do not add `TYPE_CHECKING` aliases; the stub defines `__getattr__` only under `if not TYPE_CHECKING`.
- `.rst` pages and the README do not mention deprecated APIs.
  Docstrings carry no deprecation notes, except one line in `Args` that calls out a deprecated option and what to use instead.
- The data-driven tests in [`core_test.py`](src/qldpc/_deprecations/core_test.py) cover every registry entry and the attached methods, and [`decoders_test.py`](src/qldpc/_deprecations/decoders_test.py) covers the deprecated decoder API.
  [`decoders/legacy_api_test.py`](src/qldpc/decoders/legacy_api_test.py) checks that the `qldpc.decoders` root names of v0.3.3 and v0.4.1 remain importable.
  Tests of inline deprecations stay beside their modules; name new ones `test_deprecated_*`.
- To remove deprecations, grep for `_deprecations` and, case-insensitively, for `deprecat`.
  Delete the registry entries, attached methods, inline blocks, and `Args` notes, the stubs of modules that no longer serve deprecated names, and the matching tests.

## Validation commands

Install the development environment with `python -m pip install -e '.[dev]'`.
CI's exact environments and commands are in [`.github/workflows/`](.github/workflows/).

| Command | Purpose |
| --- | --- |
| `python checks/pytest_.py [paths]` | Pytest suite, or the tests under the given paths; does not execute notebooks by default |
| `python -m pytest path/to/module_test.py::test_name -v` | One focused test |
| `python checks/pytest_.py --notebook examples/decoders.ipynb examples/basics.ipynb examples/noise_models.ipynb` | Execute the notebooks that CI executes, with their dependencies installed |
| `python checks/format_.py --check` / `--fix` | Check, or apply, Ruff and `pyproject.toml` formatting |
| `python checks/lint_.py` | Ruff lint |
| `python checks/mypy_.py` | mypy with the repository's typing settings, over source and tests |
| `python checks/coverage_.py` | 100% statement coverage of each module by its own test file, as in CI |
| `python checks/build_docs.py` | Strict Sphinx and notebook documentation build |
| `python checks/all_.py` | Combined gate: formatting, lint, mypy, tests with overall 100% coverage, and (on full runs) the docs build; it does not run the per-module coverage gate or `pyproject.toml` formatting |

Run checks on what you changed while iterating, and save repository-wide runs for a finished change.
Every check script except `build_docs.py` accepts explicit files, as in `python checks/coverage_.py src/qldpc/codes/common.py`, or `-i/--incremental [REVISION ...]`.
The incremental mode checks only the files changed since the merge base with `upstream/main`, `origin/main`, or `main` (the first that exists), or with the given revisions.
For example, `python checks/coverage_.py -i` runs the per-module coverage gate for changed modules only, which is much faster than a full run.
`format_.py` always checks, or with `--fix` rewrites, `pyproject.toml` as well, whatever files it is given.
Prefer the individual scripts for incremental runs: `all_.py -i` measures overall coverage over only the selected tests, so shared fixtures in `conftest.py` files can fail it spuriously.
Before merging, run `python checks/all_.py` and `python checks/coverage_.py`.

The docs build renders saved notebook outputs without executing code cells.
Refresh outputs by executing the entire canonical notebook in an offline, editable installation before changing a notebook example.
Pass an absolute path to this checkout's `src` directory in `PYTHONPATH` if a notebook runner starts its kernel from `examples/`.
The TQEC notebook (`examples/logical_error_rates/misc/decoding_tqec_circuits.ipynb`) requires Python 3.11--3.13 because its pinned TQEC dependency does not support Python 3.14.

Some check wrappers discover files through Git.
Add new source and test files to the index before relying on the full gate to include them.
Do not weaken a failure with `noqa`, `type: ignore`, or coverage exclusions unless the exceptional condition is real and documented.
