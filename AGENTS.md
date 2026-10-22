# Agent and contributor guide

This guide is for people and agents changing qLDPC.
For a human-facing explanation of the data model and package relationships, start with the [library map](docs/source/library_map.rst).
For exact signatures and per-object literature, use the source docstrings or the generated [API reference](https://qldpc.readthedocs.io/autoapi/index.html).

## What to trust when documents disagree

When sources disagree, use this order:

1. Current implementation and its invariant tests.
2. Public docstrings and the `__all__` lists in package `__init__.py` files.
3. The [library map](docs/source/library_map.rst) and executable [examples](examples/).
4. Historical discussion, issues, or review notes.

Re-derive behavior from the current branch before documenting or changing it.
Do not copy transient project history, machine-specific paths, or local-session instructions into production files.

## Orientation

The source tree under `src/qldpc/` mirrors the library: each package and module starts with a docstring whose first line summarizes it, and the [library map](docs/source/library_map.rst) explains how the packages relate.
Canonical executable notebooks live in [`examples/`](examples/), Sphinx pages and the library map in [`docs/source/`](docs/source/), and local wrappers for CI's quality gates in [`checks/`](checks/).

The main code hierarchy is:

```text
AbstractCode
|- ClassicalCode
`- QuditCode
   `- CSSCode
```

The methods in `codes/common.py` share cached and mutable state for standard form, logical and gauge operators, parameters, and code transformations.
Do not split them into mixins merely to reduce the file length.
Immediate decoder builders live beside their implementations under `decoders/external/` and `decoders/custom/`, and typed decoder specifications live under `decoders/construction/`.
Plumbing lives in private modules, whose file names start with an underscore, beside the public code that it serves.

## Public API and compatibility

- A name is public if it is broadly useful to researchers and engineers, and private if it is plumbing that makes qLDPC work.
  Where a name is used, such as in other modules or in notebooks, does not decide this.
- [`src/qldpc/__init__.py`](src/qldpc/__init__.py) exports subpackages rather than flattening their classes and functions.
- Each stable subpackage has an explicit `__all__` in its `__init__.py`.
  Preserve those import paths when moving implementation code.
- Treat plain names in public `qldpc.*` modules (no underscore anywhere in the import path) as public, and keep them working.
  Retire a released public name through a tested shim rather than breaking its import (see [Deprecations](#deprecations)).
- Everything under [`src/qldpc/experimental/`](src/qldpc/experimental/) is explicitly unstable and can change without a deprecation period.
  Do not infer that this weaker guarantee applies elsewhere.
- Keep complete lists of public symbols in `__all__` and AutoAPI.
  Human-written docs should explain what packages do and show representative tasks, not duplicate a class catalogue.
- An underscore-prefixed name is used only within its own file and that file's `*_test.py`.
  Anything that another module uses is a plain name, either in a public module, where it is public API, or in a private module.
  The single exception is that [`_deprecations/decoders.py`](src/qldpc/_deprecations/decoders.py) imports private backend builders, so that deprecated APIs pass options to them unchecked.
- A module whose file name starts with an underscore is private as a whole.
  It uses plain names for definitions that other modules use, and underscore names for helpers internal to the file.
- Public objects are defined in public modules.
  A public module may use a private module's implementation, but never re-exports a private module's object as public API, except as a temporary deprecation shim.

## Module organization and ordering

- Preserve a module's existing section headings and organization when adding code.
- Add a section when new functionality does not fit an existing one.
- Start each section with a line of 100 `#` characters followed by a one-line `# title` comment.
- Put important user-facing APIs first within modules and sections.
- Keep related public entry points together.
- Place file-internal helpers at the bottom of the section that they serve, so that sections group code by topic rather than by visibility.
  Reserve a final `# helper methods` section for helpers that several sections share.
  Private constants and types may sit near the imports, and helpers that code uses at import time must precede that use.
- Private modules follow the same rules: within each section, put plain names (used by other modules) before underscore names (used only within the file).
- Keep user-facing decode methods near the top of decoder classes.
- Place private helper methods after the public decode methods.

## Core invariants

### Fields and arrays

- Code arithmetic happens in a `galois.FieldArray`.
  The default field is `GF(2)`.
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
- Built-in families may cache construction- or literature-supplied parameters.
  A test that compares `get_code_params()` only with those same cached values is tautological; use an independent rank, commutation, row-space, or bounded-distance oracle.
- [`codes/distance/`](src/qldpc/codes/distance/) is explicitly binary.
  Use field-aware code methods and oracles for nonbinary constructions.

### Decoders

- Build decoders from typed specifications, as in `decoders.bp_osd(...).build(pcm_or_dem)` or `decoders.mwpm(...).build_observable_decoder(dem)`.
  Builders named `get_decoder_<NAME>`, `get_error_decoder`, and `get_observable_decoder` are deprecated and live only in [`_deprecations/decoders.py`](src/qldpc/_deprecations/decoders.py).
- `decoder=None` selects GUF for a nonbinary `FieldArray` and BP+OSD otherwise (see [`construction/_resolution.py`](src/qldpc/decoders/construction/_resolution.py)).
- Keep error decoders (`decode_errors`, with `decode` as an alias) distinct from observable decoders (`decode_observables`).
  Code that consumes a user-supplied error decoder coerces it with `as_error_decoder` from `decoders/_common.py` and calls `decode_errors`.
- A method that decodes a matrix it constructs itself must reject prebuilt decoders with `reject_prebuilt_decoder` from `decoders/construction/_resolution.py`.
- Code-capacity estimators resolve their `decoder=`, `decoder_x=`, and `decoder_z=` inputs with [`codes._code_capacity.get_code_capacity_decoder`](src/qldpc/codes/_code_capacity.py), which always yields an observable decoder: on binary codes, specifications with native observable support build it from a DEM; otherwise inferred errors can be projected onto logical predictions.
  Keep `resolve_decoder` in `decoders/construction/_resolution.py` error-decoder-specific, and dispatch on explicit capabilities (`compile_decoder_for_dem`, `decode_observables`, the `ErrorDecoder` protocol), never on output length.
- Only decoders that declare erasure support may append an erasure flag.
  They append that flag as the last entry of each inferred error; unsupported decoders must reject `add_erasure_bit=True`.
- Detector-error-model decomposition indices and remaps must remain valid after cancellation and simplification.
  Test malformed components, not only happy-path Stim models.
- Sliding-window time inference is heuristic.
  Preserve the first nondecreasing detector coordinate convention, and use an explicit `detector_to_time` mapping when a model follows another layout.

### Circuits

- Circuit and tableau constructors are qubit-only and should use the existing `restrict_to_qubits` guard.
- `get_encoding_circuit` constructs a valid encoder but is not fault-tolerant.
  Do not present it as a fault-tolerant state-preparation procedure.
- Transversal-gate search enumerates automorphisms and can be exponential.
- Keep data, check, reference, and ancilla roles explicit through `QubitIDs` and bookkeeping types.
  Do not infer roles from coincident integer indices.
- qLDPC memory-circuit detectors use coordinates such as `(round, 0, check_index)`.
  A later monotone coordinate can enumerate checks rather than time.
- Noise-model operation immunity and qubit immunity are separate controls; preserve both.

### External systems and caching

- GAP helpers may start subprocesses, read standard input, use the clipboard, access the network, or clone missing packages.
  Keep these side effects explicit in docstrings and error messages.
- Expensive reusable computations should use `qldpc.cache.use_disk_cache()`.
  Do not hide errors by returning a default value that looks successful.

### Tests

- Tests run with network sockets disabled.
  Mock only external boundaries such as web, subprocess, clipboard, prompt, and optional-package paths, or a module's own thin wrappers around them, never the logic under test, and do not add a live-service test.
- Disk caches are bypassed while pytest is imported, so tests must not depend on cache persistence.
- Statement coverage is gated at 100%.
  Co-located `*_test.py` files are the strong convention, although a thin helper can be covered through its consumer's test module.
  Reach a hard-to-reach branch with a focused test rather than a broad scenario.
- Pytest turns every `DeprecationWarning` into an error, so tests must use the modern API.
  Test deprecated paths as described in [Deprecations](#deprecations).
- A test should guard behavior that matters: results, invariants, documented errors, and contracts between modules.
  Do not pin implementation choices such as internal call order, private attributes, cache layouts, or incidental error-message wording.
  Testing a private helper directly is fine when that is the simplest way to check its contract or reach coverage.
- Keep each test minimal: use the smallest code, field, or circuit that exercises the behavior, do not parametrize over cases that take the same path, and delete tests implied by others.
- Prefer independent oracles (rank, commutation, row-space identities, or brute force on small instances) over values that the code under test computed or cached.
  Checking a stored value against an independent expected value, such as one from the literature, is not tautological; deriving the expected value from the code under test is.
- Keep tests that reproduce results from the literature, and cite the source in the test.
- Seed or construct random inputs so that a test cannot fail by chance.
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

Follow the [adding a decoder guide](docs/source/adding_decoders.rst) for a custom-decoder example and a first-party backend checklist.

1. Put integrations with third-party decoder packages under [`decoders/external/`](src/qldpc/decoders/external/), and qLDPC-owned implementations under [`decoders/custom/`](src/qldpc/decoders/custom/).
2. Keep each immediate builder beside the implementation it constructs.
   Add a typed specification under [`decoders/construction/`](src/qldpc/decoders/construction/) when exposing an algorithm.
   Do not change generic resolution for an ordinary backend.
   Export concrete backend classes at qualified `custom` or `external` paths, and export public specification helpers from [`decoders/__init__.py`](src/qldpc/decoders/__init__.py).
   Derive each specification helper's signature from one typed builder or constructor, write its public docstring explicitly, and add it to the `autofunction` inventory in [`decoders.rst`](docs/source/decoders.rst).
   Keep deprecated keyword translation isolated in `_deprecations/decoders.py`.
3. Decide and test batch behavior, nonbinary support, detector-error-model support, and erasure signaling explicitly.
4. Use direct syndrome/error reproductions in addition to factory-selection tests.

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
   AutoAPI provides the complete list of public symbols.
4. When a public limitation changes, update the relevant function or class docstring and every README or guide that repeats it in the same pull request.
5. Keep one prose sentence per physical source line in `.rst`, `.md`, and Markdown notebook cells.
   Put `First sentence.` and `Second sentence.` on separate source lines; leave code blocks, equations, and table syntax intact.
   Python comments and docstrings instead use ordinary paragraphs: separate sentences with two spaces and fill lines to the 100-column limit.
6. Run the strict documentation build before considering the change complete.

## Deprecations

Keep deprecated code in one place, so that it is easy to find and to remove.

- All deprecated code and APIs live under the private package [`qldpc/_deprecations/`](src/qldpc/_deprecations/).
  [`core.py`](src/qldpc/_deprecations/core.py) holds the generic infrastructure: the registry of renamed and moved names, `warn_deprecated`, `defunct`, and the deprecated methods and DEFUNCT stubs that `install_deprecated_methods` attaches to their classes.
  [`decoders.py`](src/qldpc/_deprecations/decoders.py) holds the deprecated decoder API and its translation into decoder specifications.
  The module docstring of `core.py` explains how the registry serves names, modules, and old pickles without import cycles.
- Only the following deprecation code lives outside `qldpc/_deprecations/`:
  1. Each module that still serves a deprecated name ends with this identical marked stub, which looks the name up in the registry:

     ```python
     if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

         def __getattr__(name: str) -> object:
             return _deprecations.get_deprecated_attribute(__name__, name)
     ```

  2. A package whose `__all__` must keep released names extends it with `__all__.extend(_deprecations.deprecated_names(__name__))`.
  3. Deprecated arguments, modes, and calling conventions of functions that are not themselves deprecated are handled inline, as small blocks that call `_deprecations.warn_deprecated` or raise `_deprecations.defunct(...)`.
     Substantial translation logic, such as that for deprecated decoder keywords, belongs in `qldpc/_deprecations/`, and the inline block calls into it.
  4. [`qldpc/__init__.py`](src/qldpc/__init__.py) calls `_deprecations.install_deprecated_methods()` once, after importing all subpackages.
- Add a shim only for a name that was public in a released version, at its released path.
  A name is public if a package `__all__` listed it, or if examples or docs used it.
  Names added since the last release may change without a shim.
- Register a renamed or moved name or submodule in `ALIASES`, which warns on access and names the replacement.
  Use `SILENT_ALIASES` instead for deprecated functions that warn when called, and for moved objects that old pickles reference.
  If the released `__all__` of a registered name's module did not list the name, also list it in `UNLISTED_NAMES`, so that `deprecated_names` leaves it out of `__all__`.
- Type checkers must reject deprecated names.
  Do not add `TYPE_CHECKING` aliases; the stub defines `__getattr__` only under `if not TYPE_CHECKING`.
- Docstrings carry no deprecation notes.
- Pytest turns every `DeprecationWarning` into an error, so test a deprecated path with `pytest.warns(DeprecationWarning, ...)`.
  The data-driven tests in [`core_test.py`](src/qldpc/_deprecations/core_test.py) cover every registry entry and the attached methods, and [`decoders_test.py`](src/qldpc/_deprecations/decoders_test.py) covers the deprecated decoder API.
  Tests of inline deprecations stay beside their modules and are named `test_deprecated_*`.
- To remove deprecations, grep for `_deprecations` to find every stub, inline block, and `__all__` extension.
  Delete the registry entries, attached methods, and inline blocks, the stubs of modules that no longer serve deprecated names, and the matching tests.

## Validation commands

Install the development environment:

```bash
python -m pip install -e '.[dev]'
```

Use the smallest targeted command while iterating, then the full gate before merging:

| Command | Purpose |
| --- | --- |
| `python checks/all_.py` | Complete repository gate: formatting, lint, strict mypy, tests, 100% coverage, and docs |
| `python checks/pytest_.py` | Full Python pytest suite; does not execute notebooks by default |
| `python checks/pytest_.py src/qldpc/codes/` | Tests for one package |
| `python checks/pytest_.py --notebook examples/decoders.ipynb examples/basics.ipynb examples/noise_models.ipynb` | Execute the three complete, CI-allowlisted notebooks with their dependencies installed |
| `python -m pytest src/qldpc/codes/quantum_test.py::test_name -v` | One focused test |
| `python checks/format_.py --check` | Verify Ruff and `pyproject.toml` formatting |
| `python checks/format_.py` | Apply repository formatting |
| `python checks/lint_.py` | Ruff lint |
| `python checks/mypy_.py` | Strict mypy over source and tests |
| `python checks/coverage_.py` | Modular 100% statement-coverage gate |
| `python checks/build_docs.py` | Strict Sphinx and notebook documentation build |

Run checks on what you changed while iterating, and save repository-wide runs for a finished change.
Every check script accepts explicit files, as in `python checks/coverage_.py src/qldpc/codes/common.py`, or `-i/--incremental [REVISION]`, which checks only the files changed since `origin/main` or the given revision.
For example, `python checks/coverage_.py -i` runs the modular coverage gate for changed modules only, which is much faster than the full gate.

The docs build renders saved notebook outputs without executing code cells.
Refresh outputs by executing the entire canonical notebook in an offline, editable installation before changing a notebook example.
Pass an absolute path to this checkout's `src` directory in `PYTHONPATH` if a notebook runner starts its kernel from `examples/`.
The TQEC notebook requires Python 3.11--3.13 because its pinned TQEC dependency does not support Python 3.14.

Some check wrappers discover files through Git.
Add new source and test files to the index before relying on the full gate to include them.
Do not weaken a failure with `noqa`, `type: ignore`, or coverage exclusions unless the exceptional condition is real and documented.
