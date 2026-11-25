# Changelog

All notable changes to this project are documented in this file. Every change that moves an output
is listed, as is every change to the interface. A change of outputs is a change of results, so runs
compare only within one version, or across versions whose entries say the outputs stayed the same.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Complete family help topics and static values such as built-in difficulty examples.
- Keep batch modules out of family help and completion.
- Name the spaces prerequisite in the single-environment migration guide.
- List self-check cases without loading PyTorch or the batch modules.
- Use two spaces between columns in the self-check case list.
- Complete shipped self-check case names in static shell scripts.
- Complete replay device names and text exporter Boolean values in static shell scripts.
- Explain which dumps accept an explicit frame rate in replay help.

- Windows builds export the C API functions of the CPU and renderer libraries from their headers.
- CPU CI runs GRF's replay and video export tests with OpenCV. Local test runs without OpenCV
    report those cases as skipped.
- The Windows renderer opens media from folders with names outside the system ANSI code page.
- Windows and Linux x86-64 CPU CI run the full test suite with the same test dependencies and
    pinned GRF media.
- CLI errors and hints preserve Windows path separators inside quoted values.
- CLI commands exit quietly when a reader closes the Windows standard output pipe.
- The Windows completion test runs Git Bash from its standard installation path.

## [0.3.0] - 2026-11-20

Every output of a batch and of `cuju.env` stays as it was in 0.2.1, bit for bit, on both backends,
apart from the outputs after `set_state` that two entries under Fixed correct.

### Added

- Install the engine with `pip install cuju-engine`. The Python import stays `import cuju`.
    From 0.3.0 on, `pip install cuju` installs the whole family through the
    [umbrella distribution](https://github.com/cuju-football/cuju).
- The engine discovers family commands through the `cuju.commands` entry point group.
    Other distributions can provide modules under `cuju`, including from editable installs.
    The engine repository is `cuju-football/engine`, and the arena, browser renderer and umbrella
    have their own repositories in the Cuju organization.
- `CUJU_BATCH_HOOKS` selects the hooks that receive batches when they are created. Batches discover
    hooks from the `cuju.batch_hooks` entry point group when the variable is set.
- `cuju.engine_status` and `cuju-repair` check for damage from an in-place upgrade of 0.2.1.
    To avoid the damage, uninstall 0.2.1 with `pip uninstall cuju` before installing 0.3.0.
    If the upgrade already removed engine files, run `cuju-repair` and then the pinned reinstall
    command it prints. For the regular wheel in 0.3.0, that command is
    `pip install --force-reinstall --no-deps cuju-engine==0.3.0`.
    For a CPU-only wheel, it names `cuju-engine-cpu` instead.
- The engine CLI lists family versions and help, generates shell completions, reports check and repair results as JSON, and uses common error messages and exit statuses.
- `cuju.gfootball` is GRF's Python package itself, every module of GRF 2.10.3 ported verbatim with
    only its import lines changed, so code written for GRF runs on cuju with `gfootball` replaced by
    `cuju.gfootball`, its reach into GRF's internals included. It adds every module that 0.2.1 left
    out, among them `football_env_core`, `observation_processor`, the players, `play_game`,
    `eval_server` and GRF's own tests, the `tests` scenarios, and `tracker_setup`.
- `cuju.gfootball_engine`, GRF's engine package over cuju's engine, with the classes, functions,
    conversions and errors of GRF's compiled `_gameplayfootball`. Writes to `GameEnv.config` during
    a match reach the engine as they reach GRF's, and GRF's tracker plays two engines in lockstep
    and compares their states. It runs on the CPU library without PyTorch.
- The errors `SignatureError`, for arguments that a function of `cuju.gfootball_engine` does not
    take, and `TrackerError`, for two tracked engines whose states differ.
- The extras `gfootball-video`, `gfootball-play` and `gfootball-remote`, which add OpenCV, pygame,
    and gRPC with protobuf for GRF's videos, game and remote environments. The extra `test` adds
    psutil for GRF's own tests.
- The package cuju-gfootball, installed with `pip install cuju-gfootball`. It imports `gfootball`, its
    submodules, `gfootball_engine` and `_gameplayfootball` as the modules of `cuju.gfootball` and
    `cuju.gfootball_engine`, and gives the port's classes GRF's module names, so code written for
    GRF runs with no line changed, gym's `gfootball:` ids and Kaggle's football environment
    included, and dumps and states open in GRF and in the port alike. Beside an install of GRF it
    activates nothing and warns once. `python -m cuju_gfootball` reports which holds the names.
- `cuju_environment_set_render` and two fields of `cuju_configuration`, `render_mode` and
    `referee_animations`, in the C API, whose version is now 3. GRF's `animations` flag of
    `GameEnv.reset`, which plays every stoppage in full, reaches the simulation through them, off by
    default.
- GRF's render mode in both engines, which plays the matches of a rendering GRF bit for bit. With
    the graphics present, each change of a node of a player's or official's skeleton recomputes
    the cached transform of its root, while the reset of a position for a set piece moves the root
    without it, and every save or load of a state marks the caches to compute afresh. The engine of
    `cuju.gfootball` plays so whenever its `game_config.render` is on and a renderer opened at
    `start_game`, and a batch in the environments that `cuju_environment_set_render` gives a
    nonzero `render_mode`. With render mode off every output stays as it was. The engine's C API,
    now version 2, adds `cuju_engine_set_render` and `cuju_engine_prepare_render`, which give the
    engine of `cuju.gfootball` GRF's render mode and what each render changes in the match.
- The match's message caption at each of GRF's `SpamMessage` sites, the goals, own goals, offsides,
    fouls, cards and sendings off with GRF's wording and team and player names, and the draws of
    the officials' skin and of the light, kept outside the state for the pictures.
- `RenderState` from the engines, what GRF's graphics read at a render, through
    `cuju_environment_render_states` for chosen environments of a batch, extracted on the device
    on CUDA, and `cuju_engine_render_state` for the engine of `cuju.gfootball`, with each
    environment's count of state saves and loads, after which GRF draws the nets afresh.
    `cuju_engine_render_state_version` and `cuju_engine_render_state_size` describe the structure.
    `cuju_engine_render_start` gives the RenderState of a match's creation for GRF's
    `StartMatch`, and `cuju_engine_internal_render_state` the one of `start_game`'s match.
- GRF's frames in `cuju.gfootball` through cuju's software renderer: `create_environment(
    render=True)`, `env.render()`, rendered videos and the representations `pixels` and
    `pixels_gray` draw GRF's frames byte for byte, with the renderer called where GRF's `GameEnv`
    calls its graphics, `start_game`'s match, each reset, render, save and load. They need the
    extra `pixels`, and `CUJU_RENDER_THREADS` and `CUJU_GRF_MEDIA` choose the threads and the media.
- The representations `pixels` and `pixels_gray` and the option `render_frames` of
    `BatchEnvironment`, which play every environment in render mode, draw each environment's
    renders with a renderer of its own in GRF's order, and return GRF's frames, also through
    `frame()`, recorded on the device on CUDA through `cuju_environment_enable_render_events` and
    `cuju_environment_render_events`.
- `set_state` of `cuju.gfootball` loads the AI's mental images as GRF's loader does, which
    mirrors the ball predictions of each image that the engine did not hold once more when
    `reverse_team_processing` is on, so the matches after GRF's render switch equal GRF's.
- The package cuju-grf-media, GRF's media for cuju's software renderer, 106 files of GRF's commit
    3d9e754 copied byte for byte, 89 MB installed from a wheel of about 8 MB, with a manifest that
    names each file's source, SHA-256 and license, and `SHA256SUMS`. The new extra `pixels` installs
    it, and `gfootball-video` takes it as well. Its license expression is
    `Apache-2.0 AND Unlicense AND OFL-1.1`, and LICENSES holds the Unlicense and the OFL 1.1.
    `ci/make_render_media.py` builds it from a GRF checkout and refuses one whose files differ from
    their pinned SHA-256, and the release builds, checks and publishes its wheel.
- The software renderer libcuju_render, which the build makes beside the CPU library and the wheel
    holds. It draws each frame of a rendering GRF engine from a RenderState, what GRF's
    `Match::Put` reads, byte for byte as GRF 3d9e754 draws it on Mesa 25.2.8's llvmpipe, every pass
    inside the frame included, for every thread count. Its C API in `src/render/render_api.h`
    follows the graphics calls of GRF's `GameEnv` and gives the frame, `pixels` and
    `pixels_gray`, and the private module `cuju._pixels` loads it. `tests/test_render_frames.py`
    checks it against GRF's frames with the digests in `cuju/data/render_digests.json.z`.
- The software renderer draws a 1280 by 720 frame of an eleven a side match in about 0.56 seconds on
    one core and 0.16 seconds with four threads of a laptop at 2.3 GHz, faster than llvmpipe's 0.72
    and 0.22 seconds on the same laptop, by shading and sampling eight pixels at a time with AVX2
    and by binning each pass while the threads draw the last one. Every frame and every pass
    inside it stays as it was.
- The software renderer sets up the triangles of each pass and skins the players on all of its
    threads, shares the decoded images and meshes, the pitch and every texture's mipmap chain among
    the renderers of a process, so a later renderer prepares its first match in about a third of a
    second rather than three, and draws no pixel for a render whose frame nobody reads before the
    next render of the match, a render with `swap` off or an internal step inside a batch's call.
    Every frame that is read stays as it was.
- The software renderer draws the shadow map while the G-buffer draws, since neither reads what
    the other writes, runs the occlusion samples of the ambient pass and the probes of the shadow
    lookups as several independent chains at once, and compares the shadow map's depths as 16-bit
    integers against a threshold found once per pixel, which gives llvmpipe's answers for every
    reference value. A frame takes about a tenth less time on one core and with four threads, and
    about a twentieth less with twelve. Every frame and every pass inside it stays as it was.
- A rendering batch draws its environments side by side, `CUJU_RENDER_WORKERS` at once with
    `CUJU_RENDER_THREADS` threads each, by default one environment per processor, about one and a
    half times the frames a second of drawing them one after another. The frames
    are the same for every split. `cuju._pixels.Renderer.frame`, `pixels` and `pixels_gray` take an
    `out` array.
- The software renderer draws GRF's frames at any width that is a multiple of 4 and at any
    height. GRF's configuration sets the height to 720 before it reads the options it is given,
    so `render_resolution_x=800` alone gives frames of 800 by 720, in GRF and in cuju alike.
- `render_resolution_x` and `render_resolution_y` of `BatchEnvironment`, GRF's options, which set
    the size of a rendering batch's frames, 1280 by 720 by default, any width that is a multiple of
    4 and any height, up to 16,384 each. `pixels` and `pixels_gray` shrink the whole frame, as GRF's
    wrapper does. Internal tests compare a batch's frames, pixels and pixels_gray with GRF's at six
    sizes on both backends, and its frames after `get_state` and `set_state` with GRF's after its
    own.
- `tests/test_render_frames.py` plays fourteen more of GRF's episodes, scripts of calls to
    `disable_render`, `render`, resets, saves and loads, through `cuju.gfootball`, and eight of them
    through batches on both backends, against GRF's raw observations, frames, pixels and
    pixels_gray, whose digests `render_digests.json.z` holds.
- The software renderer keeps two habits of GRF's graphics across loads. The scoreboard changes
    only at a goal, so it reads 0 to 0 after a render switch until the next goal, and a player
    whom a load makes inactive keeps the body, the place and the name tag of the engine's last
    picture of him, where a player sent off leaves the picture. The RenderState's spare byte of
    each humanoid now says whether GRF hid him. `tests/test_render_frames.py` compares twenty
    episodes more with GRF's frames, at nine resolutions, across render switches, loads, goals,
    cards, players sent off, captions and the agents' coloured name tags.
- `cuju.gfootball` follows GRF through `disable_render` and the `render` that turns the rendering
    on again. While the rendering is off, goals leave the scoreboard and messages leave the caption's
    text as they were, as GRF's captions take no text then, and a reset builds GRF's match without
    rendering, on the pitch alone, with no captions drawn and the players' bodies left without the
    pose that GRF fits them to only while rendering. A render of such a match, at which GRF's
    process crashes on a name tag that never took its text, raises `StateError`. The renderer's C
    API, now version 2, adds `cuju_render_set_render`, GRF's `game_config.render`, and
    `cuju._pixels.Renderer` its method `set_render`.
- cuju-grf-media holds the glyph atlas of GRF's caption font, `fonts/caption_glyphs.bin`, which
    `ci/make_glyph_atlas.py` renders with SDL_ttf 2.22.0, FreeType 2.13.2 and HarfBuzz 8.3.0, and
    `ci/make_render_media.py` refuses an atlas whose SHA-256 differs from its pin.
- Step observers receive the actions that each step played, under `actions` in their information,
    `[environment_count, agents]` `int32` on the batch's device. The tensor is the caller's own
    when the step read it in place, and a copy taken at `step_async` for a step split in two. A
    batch without observers copies nothing, and every output stays as it was.
- `roles(index)` of `BatchEnvironment` and `MultiDeviceBatch`, and `cuju_environment_roles` in
    the C API, report the role that each player of one environment plays, as the raw observation
    gives it, in any representation. After a goalkeeper is sent off they show the teammate who
    took over the keeper's formation entry, a hand-over that GRF keeps outside the state, so
    `get_state` and `set_state` stay as they were.
- NOTICE paragraphs and license texts for the software renderer's ports: Mesa's MIT License in
    `LICENSES/MIT-Mesa.txt`, the zlib License of SDL, SDL2_gfx and SDL_ttf in `LICENSES/Zlib.txt`,
    and OpenCV under the Apache License 2.0. cuju's license expression adds `Zlib`.
- The documentation of GRF's frames. Rendering describes what the software renderer draws, the
    reference whose frames it gives on every machine, what internal testing covers and how to check
    a machine, the frames of `cuju.gfootball` and of batches, render mode, the variables of workers,
    threads and media, the memory, the speed and the practice of training on pixels. Differences
    from GRF lists every known difference of the frames, Migrating from GRF maps pixel code onto a
    batch, and Exactness states the claim for the frames.

### Changed

- Family help reads installed command parsers and suggests installed and known family commands
    after a typo. Usage errors print the usage of the command that failed. Replay and text export report
    results and errors as JSON. The self-check aligns its case list, keeps terminal progress
    on one line, and reports when no backend can run. Bash and Zsh completion handle attached
    option values, and Zsh completion can be sourced.
    Usage errors name the expected type for numeric arguments. Help and terminal diagnostics
    wrap long lines for easier reading.
- The guides now cover the engine alone, and the renderer's documentation lives in the [renderer repository](https://github.com/cuju-football/render).
    The Python examples use the same formatting as the engine source.
- The release publishes `cuju-gfootball` from its own job through the environment `pypi-gfootball`, after `cuju-engine`, so each new project on PyPI has its own trusted publisher.
- Every publishing job checks that the tag names the version.
- `absl-py` and `six` are dependencies, since GRF's modules import them.
- `import cuju` registers the `GFootball-*-v0` ids of `cuju.env` with Gymnasium alone. gym gets
    GRF's ids from `import cuju.gfootball`, which gives GRF's environment, on every gym on which GRF
    runs, gym 0.21 to 0.24 and gym 0.25 with numpy 1. On later gyms `cuju.gfootball` runs on a
    copy of gym 0.21's core, where GRF fails.
- `cuju.gfootball` loses what 0.2.1's stand-in added beyond GRF, the arguments `device`, `threads`
    and `api`, levels given as paths, and its import without gym, since GRF's `__init__` imports
    `gfootball.env`. `cuju.env.create_environment` keeps all of them.
- `cuju replay` and `cuju dump-to-txt` run cuju's tools for dumps, which moved out of
    `cuju.gfootball`, where GRF's own scripts `replay` and `dump_to_txt` now stand. The check of a
    dump, `check` of 0.2.1's `cuju.gfootball.replay`, stays with `cuju replay`.
- A custom scenario file that imports its roles from `cuju.gfootball.scenarios` builds as before,
    with the roles of `cuju.gfootball_engine`.
- `cuju.env.create_environment` with GRF's gym interface, the default, gives an environment that
    is also gym's `Env`, with gym's spaces, where a gym older than 0.26 is installed beside
    Gymnasium, so code written for GRF finds the classes it checks. With only Gymnasium installed,
    and with `api='gymnasium'`, the spaces stay Gymnasium's.
- `cuju.scenarios` imports without PyTorch, which now loads only for a device other than `cpu` and
    `cuda` or for `auto`.
- The CUDA kernels take about 6,800 bytes of local memory per thread, where they took about 50 KB,
    so the reservation that the driver makes at the first batch of a process on a GPU falls from
    about 9.2 GiB to about 1.05 GiB on a GPU of 128 multiprocessors, such as the RTX 4090, and from
    about 1.15 GiB to about 135 MiB on a 4 GB RTX 3050 Laptop GPU. The lists that every thread of an
    environment's warp holds alike lie once per environment in a work area of 12 KB of device
    memory, which `memory_bytes` and `memory_estimate` count, and the sorts of the animation
    selection keep their keys in registers. Every output stays the same bit for bit. On the RTX
    3050 Laptop GPU the compact kernels run 1.7% to 5.8% slower than the earlier ones, 3.0% and
    5.8% for `5_vs_5_l4_r4` at 1,024 and 4,096 environments, and 1.7% and 2.3% for
    `11_vs_11_stochastic_l1_r0`. They were proven bit-exact on that GPU, compute capability 8.6,
    and the other compute capabilities of the wheel come from the same source, which `cuju check`
    verifies on any GPU in about a minute.

### Fixed

- Zsh completion works when sourced after `compinit` or loaded as `_cuju` from `fpath`.
- The engine repair command uses the installed distribution and version for both `cuju-engine` and `cuju-engine-cpu`.
- `cuju.env.create_environment` without agents takes `step([])` as GRF's does, and a batch takes an
    empty list of actions, where both raised `TypeError`.
- The error of a GPU without room for the local memory of the kernels names the local memory of one
    thread of the kernels and the memory that the reservation adds. It named the CUDA stack limit of
    1 KB as the memory of each thread.
- `import cuju` works beside gym 0.22 and 0.23, whose registry raised for the ids that cuju
    registered.
- `set_state` of `cuju.gfootball` and of a batch keeps the formation entries of the reset for the
    players, as GRF's `TeamData` keeps them, while the state's entries and order of the teams load
    into the scenario. A state saved in an episode of the other parity of a stochastic scenario,
    whose teams swap formations, so plays as in GRF. cuju took the state's entries for the players
    too, so the kickoff after a goal placed the teams by the wrong formations.
- The two GPU memory checks of `tests/test_memory_estimate.py` count the memory of the whole GPU
    through NVML. On Windows and WSL the free memory that CUDA reports follows a budget that the
    WDDM driver moves with the memory of other processes, so they failed whenever another process,
    the suite's own among them, held a batch, and they now skip only when the GPU is full.
- `set_state` of a batch writes the raw roles through the hand-over of a goalkeeper sent off, as
    GRF's observation after a load reads them. In an environment whose goalkeeper was sent off,
    `left_team_roles` and `right_team_roles` showed the scenario's role of the replacement until
    the next step. The state's bytes and every other output stay as they were.

### Removed

- The engine extras `render` and `gfootball`. Install the whole family with `pip install cuju`,
    the browser viewer with `pip install cuju-render`, or the GRF compatibility alias with
    `pip install cuju-gfootball`. The `spaces`, `test`, `pixels` and `gfootball-*` extras
    remain available on `cuju-engine`.
- The `CUJU_RENDER` integration. Batch observers use the `cuju.batch_hooks` entry point group.
- The outdated `uv.lock`, which described the old `cuju` distribution and extras. The engine
    declares its installation requirements in `pyproject.toml`.

## [0.2.1] - 2026-11-11

Every output stays as it was in 0.2.0, bit for bit, on both backends.

### Added

- Step observers. `BatchEnvironment.add_observer` calls a function after every reset and step of
    a batch, with the done flags, the information and the rewards of each step, and once when the
    batch closes, and `remove_observer` and `observers` manage them. A batch without observers pays
    one check per step, an observer that raises is removed with one warning, and every output stays
    the same, bit for bit. An observer may remove itself during any call, and one that closes the
    batch makes the reset or step that called it raise `ClosedError`.
- `BatchEnvironment.scenario_name`, the names of the scenarios that a batch was created with.
- `cuju.kaggle.AgentRunner`, `accepts_configuration`, `action_index` and `action_indices`, which
    call a Kaggle agent function as Kaggle does and check its answers, and `cuju.kaggle.observation`
    takes a list of agents for Kaggle's multi-agent form and a shared host copy of the raw arrays.

### Changed

- `import cuju` and the commands of `cuju render` that step no batch start without importing
    PyTorch, which loads with the first `BatchEnvironment`, so `team list`, `search`, `doctor`,
    `feed` and the like answer in about a second where they took ten. Every public name of `cuju`
    stays as it was.
- cuju imports its renderer only when `cuju render` runs or `CUJU_RENDER` asks for it. Where the
    renderer is missing, `cuju render`, `python -m cuju.render`, `import cuju.render` and
    `CUJU_RENDER` give one message naming `pip install 'cuju[render]'`, and the batches of a
    process with `CUJU_RENDER` run on after one warning.
- The renderer is its own package, cuju-render, which `pip install 'cuju[render]'` installs
    beside cuju and which keeps `import cuju.render`, `cuju render` and `CUJU_RENDER` as
    they were. The wheel of cuju holds no file of the renderer, and its license names the
    licenses of the engine alone. The renderer's changes are in
    [its changelog](https://github.com/cuju-football/render/blob/main/CHANGELOG.md).
    `import cuju.render` loads the renderer from cuju's folder or from an editable cuju-render,
    and the folder of another cuju elsewhere on the path stays out of the package.

## [0.2.0] - 2026-11-07

With default options, every output stays as it was in 0.1.0, bit for bit, on both backends.

### Added

- Renderer MVP medal entrance and match contribution screen with grades, achievements and charts.
- Renderer Help page, resumable games from Home, team zip imports and page addresses.
- Recording fields for the engine scorer, called offside player and scenario timing, with half
    time in scenarios with a second half and separate episode and recording end cards.
- Automatic kit mapping and editable model colors in the team editor, and a soundscape and a
    crowd for each arena, with glass helmets for the players on the space station.
- Renderer teams with dynamic counts, fixed player indices or roles, goalkeeper choices and
    separate action sets for each side, and six GRF MARL training scenarios. Fixed sides can
    keep their chosen player order through set pieces by routing observations and actions.
- `cuju.state`, which reads single fields of a GRF state by path, through the C functions
    `cuju_state_layout` and `cuju_state_gather`, in the rest frame or the frame of the pitch.
- `cuju.render`, a renderer that shows matches in a browser in three dimensions with sound, from
    recordings that `Recorder` saves or live from a running batch with `CUJU_RENDER` or `watch`,
    reading states without changing them. It serves a local page with a library of recordings, a
    registry of Kaggle submission folders that play requested matches, team customization and
    custom player models, and ships art from GRF, three.js, Fabric.js, eleven KayKit characters with
    KayKit's animation clips, and its arenas' sounds.
- The renderer's page, with a Home page of live sources, recent recordings, links and credits, and
    pages to watch, browse recordings, request matches, customize teams and set the field. It
    presents matches as a broadcast does, with goal and decision replays, cards from the referee,
    restarts that ease into place, a wind-down after the final whistle, an end of match show and a
    broadcast mode without the page's controls, under a sky with a time of day, weather and five
    arenas.
- An ability from 0 to 1 for each side of a requested match, GRF's team difficulty for the players
    that the built-in AI controls, through `cuju.render.play`, `cuju render play` and the Matches
    page.
- The players that a team of the renderer's registry controls, from the active player alone up to
    every controllable player of its format, or the 10 outfield players with the goalkeeper left to
    the built-in AI, played through Kaggle's multi-agent form. `cuju.kaggle.KaggleAgent` takes
    `agents`, a list of agents that one function plays, and keeps its single agent form as it was.
    Each frame of a recording marks the players that agents control on its step.
- The renderer's page Play a game, where a person controls the active player of a side with the
    keyboard or a controller of the Xbox layout against GRF's built-in AI, at real time, through
    GRF's 19 actions, with two people on opposite sides, pause and resume, and saving to Recordings
    enabled by default.
- The environment variables `CUJU_RENDER`, `CUJU_RENDER_BACKLOG` and `CUJU_RENDER_HOME`.
- The command `cuju`, with `cuju render`, which serves the renderer's page, registers teams and
    plays requested matches, `cuju check`, the self-check of `python -m cuju.check`, `cuju replay`
    and `cuju dump-to-txt`, GRF's scripts for dumps, `cuju help` and `cuju --version`. The modules
    still run with `python -m`, and `python -m cuju check` stands in where the scripts folder of a
    user installation is missing from the path.
- The extra `render`, which names the renderer and adds no dependency beyond cuju's own.
- Sample matches of eleven and of five a side, which the renderer's library receives on the first
    start of the server, and `cuju render sample`, which adds them again.

## [0.1.0] - 2026-10-27

The first public release.

### Added

- A bit-exact port of Google Research Football 2.10.3 at commit 3d9e754. For the same seed and
    actions, every state, observation, reward and done flag equals GRF's, built by GCC for x86-64
    Linux with glibc 2.39.
- One C++ source, compiled for NVIDIA GPUs with one warp per environment and for the CPU with a
    thread pool, with the same bits on both and for any number of threads.
- All 19 of GRF's scenarios, any number of agents on either side, several scenarios in one batch,
    GRF's own scenario files without GRF, and scenarios that set `control_all_players`.
- GRF's representations `simple115`, `simple115v2`, `extracted` and `raw`, its rewards `scoring`
    and `checkpoints`, its action sets `default`, `v2` and `full`, its stacks and its difficulties.
- PyTorch tensors that view the batch's buffers, and steps that never wait for the GPU and that a
    CUDA graph or `torch.compile` can capture, through the operators `cuju::step` and
    `cuju::reset_done`.
- Resets of chosen environments with `reset(mask=)`, and seeds per environment with
    `reset(seeds=)`, as a list, a tuple, an array, a tensor or one integer, with -1 to draw.
- States in GRF's byte format in both directions, stacked or not. `set_state` rewrites the returned
    tensors in place and names the fields of a state that fails to fit.
- `batch.difficulty`, the difficulties that play, the scenario's where nothing overrides them.
- Exception classes in `cuju.errors`, each also a built-in type, and `cuju.__version__`.
- Memory freed for a batch dropped without `close()`.
- CPU batches that survive `os.fork()`, and `threads=0` within the cgroup's CPU quota.
- A clear message when a batch runs out of memory, naming the memory it requested and the memory
    that was free.
- `cuju.memory_estimate`, the memory that a batch requests at creation, counted without creating
    it, and `batch.memory_bytes`, the memory that a batch has requested.
- A warning when a GPU batch requests more than the free memory of its GPU, naming both amounts,
    which on Windows and WSL2 signals a spill into shared system memory.
- `check_actions=True`, which checks actions in GPU memory before each step and raises
    `ActionRangeError` before any environment steps, where such an action otherwise plays as `idle`.
- `debug=True`, which makes a GPU batch wait after each step and reset, so that a failure on the
    device raises at the call that caused it.
- The CMake option `CUJU_NVTX`, off by default and in the wheel, which wraps the calls of the C API
    in NVTX ranges for Nsight Systems.
- `with cuju.BatchEnvironment(...) as batch`, which closes the batch at the end of its block.
- Devices `cuda:N`, `torch.device` and `auto`, `first_index` to split a run, and one batch over
    several devices with `MultiDeviceBatch`.
- Options that change no output, `allocator='torch'`, asynchronous CPU steps with `step_async`,
    pinned threads, NUMA groups and page-locked outputs. Host inputs, page-locked ones included,
    may be reused as soon as a call returns.
- `cuju.env.create_environment` with GRF's arguments, defaults and gym 0.21 interface, Gymnasium's
    interface with `api='gymnasium'`, and the `GFootball-*-v0` ids.
- `cuju.gfootball`, GRF's import paths, so GRF code runs unchanged with its imports switched, with
    `write_dump`, custom players in `extra_players`, replays of dumps without rendering, and
    `Config.ScenarioConfig()`, `NewScenario` and `set_scenario_value`. As in GRF,
    `import cuju.gfootball` binds `env`, `scenario_builder.Scenario(config)` builds a
    configuration's scenario, `FootballEnv` plays a configuration without an `agent` through
    `step([])`, the dump options list GRF's dictionaries in `info['dumps']` under GRF's file names,
    and `control_all_players` combines with `extra_players`.
- GRF's wrapper classes in `cuju.gfootball.env.wrappers`, whose states carry the counts of
    `CheckpointRewardWrapper` and the frames of `FrameStack` as GRF's do, in the same bytes as the
    states of `create_environment` with `rewards='scoring,checkpoints'` and `stacked=True`.
- GRF's `bot` and `lazy` players in `cuju.players`, its `MultiAgentToSingleAgent` wrapper in
    `cuju.wrappers`, and its raw dictionaries in `cuju.raw`.
- Kaggle's observations and agents, and matches between Kaggle agents, in `cuju.kaggle`.
- GRF's dumps in `cuju.dumps`, which GRF replays and renders.
- `CUJU_LOG=1`, a line for each batch created, and `CUJU_BUNDLE_CACHE`, a cache of the bundles
    built from scenario files.
- `python -m cuju.check` in the wheel, with diagnostics, the shipped data against `SHA256SUMS`, and
    the digests on the CPU and every GPU.
- `cuju.digest`, action logs that replay a run bit for bit with `cuju.ActionLog` and `cuju.replay`,
    leaving every option that changes no output to the replaying machine, and
    `examples/compare_with_grf.py` for a side-by-side run with GRF.
- A quickstart notebook, `examples/cuju_quickstart.ipynb`, that runs on Colab.
- Wheels with both backends for Linux on x86-64, manylinux 2.28, and for Windows on x86-64, with
    machine code for compute capabilities 7.5, 8.0, 8.6 and 8.9. The wheel for Linux serves WSL2.
- Wheels with the CPU backend for Linux on Arm, for macOS 11 or later on Apple silicon, and for
    macOS 10.15 or later on Intel processors, with PyTorch 2.2 there.
- The same digests on every platform, from GCC and Clang on Linux, MSVC on Windows and Apple Clang
    on macOS.
- Builds from source for other GPU architectures.
- GRF's ids registered with Gymnasium, and with gym before 0.26, as soon as either is imported, so
    that `import cuju` leaves both unloaded.

[Unreleased]: https://github.com/cuju-football/engine/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/cuju-football/engine/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/cuju-football/engine/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cuju-football/engine/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cuju-football/engine/releases/tag/v0.1.0
