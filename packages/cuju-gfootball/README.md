# cuju-gfootball

cuju-gfootball lets code written for Google Research Football, GRF, import cuju's port of GRF under
GRF's own names. With it installed, `import gfootball` imports `cuju.gfootball`, and
`gfootball_engine` and `_gameplayfootball` import `cuju.gfootball_engine`, so GRF's scripts, GRF's
tests, the `gfootball:` ids of gym and tools built on GRF, such as Kaggle's football environment,
run on cuju with every line as it was.

```
pip install cuju-gfootball
```

`pip install cuju` installs cuju-gfootball next to the engine, and `pip install cuju-gfootball` does
the same. Each release of cuju-gfootball depends on the range of cuju-engine whose module names it
maps, cuju-engine 0.3 for this one.

## What the alias does

cuju already holds GRF's Python package as `cuju.gfootball`, its files GRF's own with only their
import lines changed, over `cuju.gfootball_engine`, an engine module with the names, types and
behaviour of GRF's compiled one. Code that imports `cuju.gfootball` runs on cuju without the alias.
The alias serves the code that cannot change its imports, a library that imports `gfootball` deep
inside, a test suite that should stay byte for byte as GRF ships it, or a pickle written by GRF.

Each of GRF's names is the very module of cuju, not a copy. `gfootball.env.football_env is
cuju.gfootball.env.football_env` holds, so a setting changed through one name is seen through the
other, and `importlib.util.find_spec('gfootball')` and `from gfootball.scenarios import *` work as
they do in GRF.

The classes of the port carry GRF's module names as well. `CoreAction` reports the module
`gfootball.env.football_action_set`, and the enumerations of the engine report `_gameplayfootball`,
as GRF's do. Python's `pickle` writes those names, so a dump or a state written through the port
opens in GRF, and a dump of GRF opens in the port. Without the alias the port's dumps name
`cuju.gfootball`, which GRF cannot import, and GRF's dumps name `gfootball`, which the port's
readers cannot import, so the alias is the piece that lets dumps travel both ways between Python
environments. cuju's own readers, `cuju.dumps.read_dump`, `cuju replay` and `cuju dump-to-txt`, read
either kind with or without it.

## How it activates

The wheel holds a file `cuju_gfootball.pth`, which Python's `site` module runs at every start of
the interpreter. It places one import finder at the front of `sys.meta_path` and imports nothing
else, so the start of Python stays as fast as before. The finder answers only for GRF's names and
for the modules of the port, and it decides whether to activate at the first import of one of them.

An interpreter started with `python -S` skips `.pth` files, and there `import cuju_gfootball`
activates the alias. The modules of the port imported before that point receive GRF's module names
at the activation.

## Beside an install of GRF

GRF and cuju-gfootball both claim the name `gfootball`, and GRF's install always keeps it. At the
first import of one of the names, the alias looks for GRF in three places: a `gfootball`,
`gfootball_engine` or `_gameplayfootball` that any other finder can import, which covers regular
and editable installs and folders on `PYTHONPATH`, and the installed distribution `gfootball`. When
it finds any of them, it activates nothing, `cuju.gfootball` keeps its own module names, and one
`RuntimeWarning` names what it found. Installing GRF after cuju-gfootball has the same effect at the
next start of Python.

The command `python -m cuju_gfootball` prints whether the alias is active and the reason where it
is not, and exits with status 1 in that case. In code, `cuju_gfootball.active()` and
`cuju_gfootball.status()` say the same.

To keep both GRF and cuju in one environment, leave cuju-gfootball out and import the port as
`cuju.gfootball`. To have the alias, uninstall GRF's `gfootball`.

## Uninstalling

`pip uninstall cuju-gfootball` removes the package, its `.pth` file and its metadata, and leaves
cuju as it was. The next start of Python imports `cuju.gfootball` under its own names again.

## Documentation and license

[Migrating from GRF](https://github.com/cuju-football/engine/blob/main/docs/migrating-from-grf.md)
describes the port, the alias and what changes for code written for GRF, and
[Differences from GRF](https://github.com/cuju-football/engine/blob/main/docs/differences-from-grf.md)
lists every behaviour of cuju that departs from GRF. cuju-gfootball is cuju's own code, under the
Apache License 2.0, and holds none of GRF's files.
