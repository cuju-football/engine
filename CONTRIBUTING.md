# Contributing

One rule shapes everything else in cuju. Its outputs equal those of Google Research Football, or
GRF, bit for bit, and every contribution keeps that rule.

## Reports

- **A difference from GRF** is the most valuable report. Use the Difference from GRF issue form,
    which asks for exactly what a reproduction needs.
- **Bugs, installation problems, performance and missing GRF features** have their own forms.
- **Questions** go to [Discussions](https://github.com/cuju-football/engine/discussions).

## Changes to the simulation

The simulation lives in `include/` and `src/`. Testing against GRF is internal. Issues are welcome.

- A change that keeps every output is welcome, such as a speed-up, a simplification or a crash fix.
- A change that moves an output is accepted only toward GRF, with a case that reproduces the
    difference.
- New behaviour beyond GRF is out of scope for now.

Open an issue before starting on a change to the simulation, so the approach is agreed first.

## Layout

The engine is header-only, in `include/cuju/`. `simulation/` is GRF's game, `batch/` the batch and
its outputs, `engine/` one engine of GRF's engine module, `state/` GRF's state format, `cpu/` and
`gpu/` the two backends. Logic is written
once, in the shared headers, and a backend only schedules it. `src/` holds the C API and the two
library sources, and `python/cuju/` the package.

In `python/cuju/`, `__init__.py` exports the public names and each public module sits at the top.
`_batch/` holds the implementation of `BatchEnvironment`, from loading the library to the pickles
of a state. `gfootball/` is GRF's Python package, which `ci/port_gfootball.py` generates from GRF's
sources with a table of import rewrites, and which changes only through that script.
`gfootball_engine/` is GRF's engine module over the engine of `include/cuju/engine/`, and `data/`
holds the bundles, the animations and the digests. `state.py` reads single fields of a state, `_plugins.py` suggests the package of a missing
module `cuju.<name>` and runs the hooks of batches, and `__main__.py` is the command `cuju`, one entry per subcommand.
`packages/cuju-gfootball/` is a separate package with its own version, which maps GRF's module
names onto `cuju.gfootball` through a `.pth` file and releases with cuju.

## Changes elsewhere

Changes to the documentation, the examples, the packaging, the build and the error messages need
only a pull request.

## Before a pull request

1. Run `pytest tests/` on the CPU, and on a GPU where one is available.
2. Run `ruff check` and `ruff format --check` for Python.
3. Run `clang-format-18 --dry-run -Werror` on changed C++ and CUDA files.
4. Keep the compiler flags as they are. The build refuses flags that change floating-point results,
    and a pull request that needs one explains why.
5. Write each commit message as one line in the imperative, at most 72 characters, with no body,
    such as `Port the flight of the ball`.
6. Write documentation in plain, precise sentences, with GRF's names for GRF's things, to
    Wikipedia's Manual of Style, with prose that uses only words, commas and periods.

## Rules for exactness

- Keep every output bit for bit what it is with every option at its default. Make a new feature
    opt-in, with no effect when off.
- Follow GRF's source, with the same operations in the same order, its choice of `float` or
    `double` for each value, and its quirks, each marked with a `GRF:` comment.
- Keep floating-point results fixed. Fast math, contraction and the reordering of a simulation
    expression are all ruled out, so the build keeps `-ffp-contract=off` and `--fmad=false` and
    refuses flags that change results.
- Call the math functions ported from glibc 2.39 in `include/cuju/libm/`, never the system's.
- Make every change work on both backends, so that whatever runs on one runs on the other with the
    same bits.
- A changed digest is a changed result, so `python/cuju/data/digests.json` stays as it is. After a
    change to `python/cuju/data/animations.bin` or a shipped bundle, run
    `python ci/data_checksums.py` to rewrite `SHA256SUMS`.

## Building and testing

```sh
pip install .                                  # both libraries where nvcc is found
CMAKE_ARGS=-DCUJU_CUDA=OFF pip install .       # the CPU library alone
pytest tests/                                  # the public suite, a couple of minutes
cuju check                                     # the shipped self-check
python ci/check_links.py --offline             # the links of the README and the docs
ruff check && ruff format --check              # Python
clang-format-18 --dry-run -Werror <files>      # C++ and CUDA
```

## Style

- Indent with four spaces everywhere, in C++, Python, Markdown lists and code blocks.
- In C++, name constants in `UPPER_SNAKE_CASE`, variables in `camelCase`, functions and types in
    `PascalCase`. Write `//` comments only, sparse, saying why. Prefix GRF facts with `GRF:`,
    exactness notes with `Exact:`, speed notes with `Speed:`.
- In Python, follow PEP 8 through `ruff`, with line length 100, single quotes and Google-style
    docstrings. Raise the classes of `cuju.errors` rather than bare built-ins.
- Start new files with `Copyright 2026 The cuju Authors` and `SPDX-License-Identifier: Apache-2.0`.
- Leave `python/cuju/gfootball/` out of `ruff`, since `ci/port_gfootball.py` generates it.

## Leave alone unless the pull request states a reason

- `python/cuju/data/`, the animations, the bundles, the digests and their checksums.
- `include/cuju/libm/`, LGPL ports of glibc, and `LICENSE`, `NOTICE` and `LICENSES/`.
- The floating-point guards in `CMakeLists.txt` and `include/cuju/floating_point.h`.
- The C API's layout. Append fields of `cuju_configuration` only, raise
    `CUJU_CONFIGURATION_VERSION` with each addition, and start each new function on its own line in
    `src/*_api.h`.
- The ownership of views. A view or buffer that a batch hands out holds `batch._native`, never the
    batch, so that a dropped batch is freed.
