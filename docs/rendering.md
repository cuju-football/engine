# Frames and videos

The engine's software renderer draws the frames of Google Research Football, or GRF, byte for byte
as GRF 3d9e754 draws them on Mesa 25.2.8's llvmpipe. These frames supply `env.render()`, rendered
videos and the representations `pixels` and `pixels_gray` in `cuju.gfootball`. A `BatchEnvironment` draws
the same frames with `pixels`, `pixels_gray` or `frames=True`. Models that learn from pixels
receive the same inputs as in GRF. Rendering also activates GRF's
[render mode](#render-mode-and-the-match), which changes how the match plays.

## The software renderer

cuju's software renderer, the library libcuju_render, is a port of GRF's graphics and of the parts of
Mesa's llvmpipe that GRF's shaders run on. It needs no GPU, no OpenGL driver and no display, and it
draws into memory, so a frame is available on any machine where cuju runs, a container in the cloud
included. The library ships in cuju's wheel, and GRF's media, the models, textures and font that it
draws from, come in the package cuju-grf-media, as [GRF's media](#grfs-media) describes.

### What it draws

A frame of GRF is a broadcast view of the match. The camera follows the play from the side of the
pitch, and the scene holds the stadium with its walls, crowd and floor, the 23 adboards with images
that a seeded draw assigns when the engine builds its first match, the procedural pitch with its
grass and markings, the two goals with netting that bulges where the ball strikes it, the ball,
every player and official on the pitch, skinned from the 13 joints of the skeleton with the kits,
skins and hairstyles of the two teams, and the cards that the referee shows. The light of the sun
has a direction and a colour that each match draws at random, and it casts shadows.

Over the scene lies GRF's two-dimensional overlay. A scoreboard shows the teams' names, the goals and
the clock, a radar at the bottom shows the players and the ball from above, a name tag floats
over each player, and a caption announces goals, offsides, fouls, cards and sendings off in GRF's
wording, with the names of the team and the player. The first reset of an engine draws GRF's loading
page, and every reset draws one frame between two matches, before the new match starts.

The renderer draws each frame through the passes that GRF's deferred shading runs, in GRF's order. A
shadow pass draws the depth of the scene as the sun sees it, a geometry pass writes the colour,
normal and other properties of every visible surface into three buffers of half-precision floats and
a depth buffer, an ambient pass adds the ambient light with screen-space ambient occlusion, a
lighting pass adds the light of the sun with its shadows, a post-processing pass turns the lit scene
into the 10-bit picture of the screen, and the overlay pass draws the menus over it. The frame that
GRF's Python receives is the picture read back as eight bits per channel, three bytes per pixel,
with the rows from the bottom, as GRF's `glReadPixels` reads it. GRF's Python turns it into the
observation's `frame`, rows from the top, and GRF's wrappers shrink it to 96 by 72 pixels with
OpenCV's area resampling, after a conversion to gray for `pixels_gray`. The renderer does these last
two steps with ports of OpenCV's own arithmetic, so its `pixels` and `pixels_gray` equal GRF's as
well.

### The RenderState

A frame depends only on what the match shows, so the renderer reads one **RenderState** per render,
a plain record of the camera, the ball, the poses of the players and officials, the cards, the
clock, the score and the message caption, which is what GRF's `Match::Put` hands to its graphics. At
the start of each match the RenderState also carries the match's looks, the sun, the teams' kits and
the players' skins and hair. cuju's engine fills the RenderState from its own simulation, on the GPU
for a batch that runs there, so drawing a frame never touches the match, and every observation,
reward and state of a rendering environment stays what it is without the renderer.

The renderer follows the calls that GRF's `GameEnv` makes into its graphics. A reset clears the
resources nothing holds anymore, stops the running match, shows the loading page on the first reset
of an engine, draws one frame without a match, and starts the next match. A render prepares the
frame from the RenderState and draws it, and with `swap` it reads the picture back. A save or a load
of a state marks the goal nets for drawing afresh, as GRF's do. Because the renderer sees the same
calls in the same order as GRF's graphics, it carries the same history from frame to frame, the
field of view of the last render, the uploads of the players' meshes and nets, the shown score and
the cards, and each frame follows from that history as GRF's does.

Two habits of GRF's graphics outlive a load, and the renderer keeps both. The scoreboard changes
only at a goal, where it takes the scoring team's count, so a fresh rendering engine shows 0 to 0
after the render switch of `env.render()` in the middle of an episode, whatever the score, and a
load into a rendering environment leaves the scoreboard as it was, until the next goal brings the
scoring team's figure up to date. A player sent off leaves the picture at once, his dot leaving the
radar with him, while a player whom a load makes inactive keeps the body, the place and the name
tag of the engine's last picture of him. After a render switch past a sending off, the rendering
engine never drew him, so he stands in the resting pose of the player model at the centre spot for
the rest of the match, with his name at the top left of the picture, as he does in GRF.

### Frames equal to GRF's

Each frame equals, byte for byte, the frame that GRF 3d9e754 draws on Mesa 25.2.8's llvmpipe at the
same resolution. The comparison covers more than the finished picture. Every pass inside a frame,
the shadow map, the geometry buffers, the lights, the post-processing and the overlay, equals GRF's
pass as well, and so do the mipmaps of every texture, the procedural pitch, the SSAO kernel and the
glyphs of the captions.

#### The reference

GRF draws its frames with OpenGL, so they are the work of whatever driver runs its shaders. Two
drivers rasterize the same triangles to slightly different pixels, round their arithmetic in their
own ways and filter textures differently, and so do two versions of one driver. GRF's frames are
therefore a family, one per driver and machine, rather than one picture, and a claim of equal
frames names one member of the family.

cuju's reference is GRF at commit 3d9e754, built as the [reference build](exactness.md#reference-build)
of the engine is, drawing on the llvmpipe driver of Mesa 25.2.8 with LLVM 20.1.2, on an x86-64
processor with AVX2, FMA and F16C, an AMD Ryzen 5 7535HS of the Zen 3 generation. llvmpipe is
Mesa's software rasterizer, which compiles each shader to machine code for the processor at hand
and which GRF draws with wherever no GPU driver is present, as on a server or in a container. The
reference is therefore the picture that a headless GRF draws on such a machine with that Mesa.

The renderer reproduces the reference from its arithmetic upward. It carries a transcription of the
machine code that llvmpipe compiles GRF's shaders to, and runs each operation of it in the same
order and precision, fused multiply-adds exactly where llvmpipe fuses them, the minimum and maximum
of x86 with their order of operands, llvmpipe's own polynomials for logarithms, exponentials, cosines
and the sRGB curve in place of a math library, and llvmpipe's flushes of tiny numbers to zero at the
stages where llvmpipe sets them. The rasterizer follows llvmpipe as well, in its snapping of
vertices to the pixel grid, its rule for the pixels on a shared edge, its tiles and its blocks of
pixels, and the overlay follows SDL, SDL_ttf and SDL2_gfx, the libraries that draw GRF's menus.

#### Other machines and drivers

On every machine that cuju runs on, the renderer draws the reference's frames. A machine with an
NVIDIA or AMD GPU, where GRF itself would draw with the GPU's driver and other pixels, gets the
reference's frames from cuju, and so does a machine with another Mesa or without OpenGL, Windows
and macOS included.

One step of llvmpipe depends on the processor itself. When GRF generates the mipmaps of an sRGB
texture, llvmpipe encodes each averaged texel back to sRGB with the processor's approximate
reciprocal square root, an instruction whose last bits differ between AMD and Intel processors. A
GRF on llvmpipe on an Intel processor can therefore draw mipmaps, and from them frames, that differ
from the reference's in the last bit of a few texels. cuju carries the result of that instruction on
the reference's AMD Zen 3 processor as a table, measured for every one of its 2^32 inputs, so every
host draws the reference's mipmaps, an Intel processor and an Arm one included.

The renderer sets its own floating-point modes for each step of a frame, round to nearest, with the
flush to zero exactly where llvmpipe flushes, so a library in the same process that changes those
modes leaves the frames as they are. On processors other than x86-64, such as Arm ones, the
renderer computes the same flushes in software and the same fused multiply-adds through
correctly rounded functions, a portable path that gives the same bits more slowly.

#### What testing covers

cuju's renderer is tested against GRF internally, and a frame that differs from the reference is a
bug, so issues reporting one are welcome. The tests compare bytes and never an approximate measure
of similarity, and today they cover the following.

| Area | Covered |
| --- | --- |
| recorded episodes of GRF | 11 rendered episodes, 1,337 frames, each with every call into OpenGL and every pass inside the frame, including a goal with its caption and the ball in the net, reversed team processing and GRF's render switch in the middle of an episode |
| states from GRF | 1,015 frames drawn from the RenderState that cuju extracts from GRF's state of the same moment, loaded into cuju |
| end to end | six episodes of `11_vs_11_stochastic`, `5_vs_5` and `academy_run_to_score_with_keeper`, every observation's `frame`, `pixels` and `pixels_gray`, 541 in all, played by cuju's own simulation through `cuju.gfootball` and through batches on the CPU and on CUDA |
| GRF's scenarios | twenty further episodes, compared frame by frame with GRF's, at nine resolutions, across render switches, loads, goals, cards, sendings off, captions and the agents' coloured name tags, as described below |
| turning the rendering off and on | six episodes of calls to `disable_render`, `render`, resets, saves and loads in `cuju.gfootball`, 448 observations, each with its raw observation and, where it has one, its frame, `pixels` and `pixels_gray`, and 1,044,665 calls into OpenGL with 465 frames, including goals and messages while the rendering is off, matches built without rendering and the render at which GRF's process crashes |
| batches | frames, `pixels` and `pixels_gray` of batches at six sizes and after loads of earlier states, 201 observations of each kind on the CPU and on CUDA |
| resolution | 1280 by 720, 1920 by 1080, 800 by 600, 800 by 450, 640 by 480, 640 by 360, 1364 by 767, 96 by 72 and 96 by 54, the last two being the size of the `pixels` representation itself, at which GRF's menus shrink to a few pixels |
| threads | 1, 2, 4, 8 and 12 threads per frame, and several splits of a batch into workers and threads |
| builds | GCC for x86-64 with AVX2, FMA and the hardware's flush to zero, and the portable path with the flush in software, both on the reference processor |
| pieces on every input | the grayscale conversion of all 2^24 colours, the reciprocal square root of all 2^32 floats, and every path of the area resampling at nine sizes over 211 frames |
| captions | 1,122,336 renderings of text, GRF's own strings and random ones over every character of the font, against SDL_ttf |

The episodes of GRF's scenarios seek out the moments that a short episode rarely shows, by
scripted agents that chase the ball carrier and slide in, by long episodes rendered from a late
step, and by states edited before a load, one of them in the middle of a stoppage. A red card needs
such an edit, since GRF's referee shows one only for a tackle whose severity its formula reaches
through rounding alone. The tests cover goals that take a team past one on the scoreboard, own
goals, fouls and offsides with their captions, the referee raising a yellow or a red card in the
frames of a stoppage that the observations of an episode skip, second yellow cards that send a
player off with each of the three wordings of the caption, the player's dot leaving the radar,
agents' name tags in the colours of their controllers, which pulse while a pass or a shot waits for
the ball, `env.render()` at a step of play, at a stoppage and right after a reset, with either team
processed first, and loads of earlier states into an environment that renders. GRF makes no
substitutions, so a match keeps its players to the end.

Each speedup of the renderer, the eight-pixel shading, the shared mipmaps, the threads that prepare
triangles, the passes that overlap and the renders that draw no pixel, was checked against these
frames and every pass inside them before it was kept.

The renderer is written to give the same bits beyond what the table lists, at other resolutions,
with other compilers, on Intel and Arm processors and on Windows and macOS, and internal testing
widens the list release by release. Frames from these, or from a situation that the tested episodes
do not reach, are the most useful reports of all.

#### Checking a machine

The repository's tests check the renderer on any machine, against digests of the reference's
frames that ship in `cuju/data/render_digests.json.z`. They need the renderer's library and GRF's
media, and they skip with a message that names what is missing.

```sh
pytest tests/test_render_frames.py tests/test_render_media.py
CUJU_LONG=1 pytest tests/test_render_frames.py
```

The first command takes about half a minute. It draws the first 26 frames of a stored episode of GRF
from their RenderStates, with their `pixels` and `pixels_gray`, and checks the fixed data, the
procedural pitch, the SSAO kernel and the glyph atlas, along with the media's checksums. With
`CUJU_LONG=1` the tests also draw the whole episode with the mipmaps of every texture, at several
thread counts, with and without the renders that nobody reads, play the six episodes of the table
end to end through `cuju.gfootball` and through batches, and play GRF's scenarios at their
resolutions, which takes several minutes. A
failure on a machine where the reference's frames are expected is worth an issue with the test's
output and the output of `cuju check`.

## Installing the media

The renderer's library is part of cuju's wheel on every platform, and GRF's media come in the
package cuju-grf-media, which the extra `pixels` installs.

```sh
pip install 'cuju-engine[pixels]'
```

The extra `gfootball-video` installs the media too, together with OpenCV, since GRF's Python writes
its videos with OpenCV, so a project that renders videos in `cuju.gfootball` takes that extra
alone. `CUJU_GRF_MEDIA` names a folder of GRF's media in place of the installed package, such as the
output of `ci/make_render_media.py`, as [GRF's media from a clone](installation.md#grfs-media-from-a-clone)
shows, and `CUJU_RENDER_LIBRARY` names another build of the library. Neither changes a frame of the
same media. The settings of speed, `render_threads` and `render_workers`, are keys of the
[machine config](installation.md#machine-config).

On Windows, the renderer opens media from folders with Unicode names, including names with
characters outside the system ANSI code page. Set `CUJU_GRF_MEDIA` to the folder that holds the
media, including its caption fonts, when using a copy outside the installed package.

Where the library or the media are missing, a rendering engine of `cuju.gfootball` raises
`RendererMissingError` at `start_game`, and a rendering batch raises `cuju.errors.MissingFileError` at
creation, each with a message that names `pip install 'cuju-engine[pixels]'`. `RendererMissingError` is a
`cuju.Error` and a `ModuleNotFoundError`. Every part of cuju that trains without rendering runs as
it stands without either.

## GRF's rendering in `cuju.gfootball`

GRF's Python package in `cuju.gfootball` renders through its engine as GRF's does. Code written for
GRF that renders runs on it as it stands, the representations `pixels` and `pixels_gray`,
`create_environment(render=True)`, `env.render()`, `write_video=True`, `PixelsStateWrapper`, GRF's
`dump_to_video` and its rendering `replay` included. A program from GRF's documentation turns on
the rendering and reads a frame.

```python
import cuju.gfootball.env as football_env

env = football_env.create_environment('11_vs_11_stochastic', representation='pixels', render=True)
observation = env.reset()  # [72, 96, 3] uint8, GRF's pixels
frame = env.render(mode='rgb_array')  # [720, 1280, 3] uint8
```

As in GRF, `pixels` and `pixels_gray` need `render=True`, and `create_environment` with it calls
`env.render()` once the environment exists, which moves the match into a rendering engine through
GRF's render switch. `env.render()` in the middle of an episode switches as well, by saving the
state, starting a rendering engine, resetting it and loading the state, exactly as GRF's does, the
reload of the AI's mental images in GRF's way included. GRF calls the engine's render twice at the
switch, the first time without reading the frame back, and cuju's renderer sees both. The raw
observation then holds `frame`, `[720, 1280, 3]` `uint8` with the rows from the top and the colours
in the order red, green, blue. `env.render(mode='rgb_array')` returns the same picture with the
first and last channels exchanged, since GRF's Python reverses them there, and `mode='human'`
returns `True`.

GRF allows one rendering environment in a process, and `cuju.gfootball` keeps that rule, since it
runs GRF's Python as it stands. When a rendering environment closes, GRF keeps its engine for the
next environment of the process that renders, which takes it with its resolution and the history of
its renders, and `cuju.gfootball` does the same. A batch, described below, holds as many rendering
environments as a process has memory for.

A video with `render` off is GRF's two-dimensional view, which GRF's Python draws with OpenCV and
which needs no renderer. With `render` on, `write_video=True` writes the rendered frames of every
step, the internal steps that repeat while the ball is out of play included, as GRF's video does,
with GRF's statistics drawn over them when `display_game_stats` is on.

### The order of the calls

The engine calls the renderer where GRF's `GameEnv` touches its graphics. `start_game` builds the
graphics of GRF's own first match, the reset to `ScenarioConfig::make()` with the seed 42, whose
graphics also draw the adboards, and the first rendering match of a process generates the pitch.
Each reset draws its frame between two matches and starts the next match's graphics, every render
draws the RenderState of the match as it stands, and every `get_state` and `set_state` marks the
goal nets for drawing afresh, as GRF's do. The frames therefore equal GRF's, byte for byte, from the
first reset of an engine on, including the frames of the internal steps that a video records while
the match waits for a restart.

### Resolution, threads and media

The frames take the size of GRF's options `render_resolution_x` and `render_resolution_y`, 1280 by
720 by default, which GRF's `create_environment` passes through `other_config_options`. The renderer
draws at any width that is a multiple of 4 and at any height, up to 16,384 each, and `start_game`
raises `cuju.errors.ArgumentError` for another width. The rule for the width is GRF's own. GRF reads each
frame back with three bytes per pixel in rows that OpenGL pads to whole words of four bytes, while
its Python takes the bytes as unpadded rows, so a width that leaves a row short of a whole word
gives GRF a picture whose rows slide into each other. GRF's configuration sets the height to
720, 0.5625 times its default width, before it reads the options it is given, so
`render_resolution_x=800` alone gives frames of 800 by 720 pixels, in GRF and in cuju alike, and
a frame of 800 by 450 takes both options. Internal tests compare whole episodes at nine sizes, as
[What testing covers](#what-testing-covers) lists, from 1920 by 1080 down to 96 by 54, a frame no
larger than a `pixels` observation, at which GRF's menus shrink to a few pixels.

The key `render_threads` of the [machine config](installation.md#machine-config) sets the threads
that draw each frame, by default one per processor that the process may run on, four at most. No number of threads changes a frame, since the renderer splits
each pass into fixed tiles of 64 by 64 pixels and each tile draws its triangles in their order. A
rendering engine draws 1280 by 720 frames at about six a second with four threads, and the
section on [speed](#speed) gives the measured figures.

A render with `swap` off, whose frame GRF never reads back, costs only its preparation, a few
milliseconds. The renderer still makes every upload that a later frame depends on, the players'
meshes, the nets and the menus, and keeps the field of view that GRF's next render reads, but it
rasterizes no pixel, since the next render draws every pixel again.

### Real time and disabling the render

GRF's option `real_time` draws on a schedule of the wall clock. A step of ten ticks spreads as many
renders over its ticks as the step before, one fewer when that step took longer than about nine
milliseconds a tick and one more when it took less, so the count of renders in each step follows
the speed of the machine. In render mode every render also moves the cached skeleton positions that
the next ticks read, as the [next section](#render-mode-and-the-match) explains, so the match itself
follows that count. `cuju.gfootball` applies the same rule with its own clock, and its frames equal
GRF's whenever both draw the same renders in each step, which depends on how long each machine
takes to draw a frame. An episode that should repeat exactly is played with `real_time` off, the
default.

`env.disable_render()` turns GRF's `game_config.render` off, and a later `env.render()` turns it
on again, and cuju follows GRF through both. While the rendering is off the engine draws nothing
and plays the match as a rendering GRF plays it with its rendering off. The referee keeps the
animations of the match's reset, and the cached skeleton positions that the graphics computed stay
until each one's next change, as [Render mode and the match](#render-mode-and-the-match) explains.
GRF's captions take no new text while the rendering is off. A goal scored then leaves the scoreboard
as it stood until the next goal scored with the rendering on, and a message of the referee or of a
goal spammed then still shows, when the rendering returns before the message times out, but with
the caption's last text. cuju's renderer keeps its captions the same way, so the frames after
`env.render()` equal GRF's, which internal tests check over goals, messages, saves and loads made
while the rendering is off.

A reset while the rendering is off builds the match as a GRF without rendering builds it, on a
stadium of the pitch alone, with no caption drawn, and with players whose bodies miss the standing
pose that GRF fits them to only while rendering. GRF cannot draw such a match. Its first render,
from `env.render()` or from a step after the rendering is turned on, redraws a player's name tag
that never took its text, and GRF's process crashes on the empty caption. cuju raises `StateError`
at that render, with a message that names the cause, and the environment is then of no further use,
as GRF's process is gone. The match itself plays, saves and loads with the rendering off as GRF's
does, and a reset with the rendering on builds a full match again, which GRF's Python never does by
itself, since it turns the rendering on only through `env.render()`, while a write of
`game_config.render` on the engine and a reset do. GRF's `PeriodicDumpWriter` with `render=True`
and a `dump_frequency` above 1 makes exactly the crashing calls, since it turns the rendering off
before the reset of each episode that it does not record and on again before the reset of the
next one that it does, so it stops at its first recorded episode after an unrecorded one, in GRF
with the crash and in cuju with the error. A `dump_frequency` of 1 renders every episode and runs
as long as wanted.

## Rendering batches

A `BatchEnvironment` renders as well, which suits evaluation and recording at scale and training on
pixels. `representation='pixels'` and `representation='pixels_gray'` return GRF's frames shrunk to
`channel_dimensions`, 96 by 72 by default, and `frames=True` adds the full frames, which the method
`frame()` returns, with any representation.

```python
import cuju
import torch

batch = cuju.BatchEnvironment(
    '11_vs_11_stochastic',
    environment_count=8,
    device='cpu',
    representation='pixels',
    frames=True,
)
observation = batch.reset()  # [8, 1, 72, 96, 3] uint8
actions = torch.zeros((8, 1), dtype=torch.int32)
observation, reward, done, information = batch.step(actions)
frames = batch.frame()  # [8, 720, 1280, 3] uint8, rows from the top
```

The outputs have these shapes, all `uint8`, on the device of the batch.

| Output | Argument | Shape | Content |
| --- | --- | --- | --- |
| observation | `representation='pixels'` | `[environment_count, agents, height, width, 3]` | GRF's `pixels`, the same frame for every agent of an environment, as GRF's `PixelsStateWrapper` gives it |
| observation | `representation='pixels_gray'` | `[environment_count, agents, height, width, 1]` | GRF's `pixels_gray` |
| `frame()` | `frames=True` or a pixel representation | `[environment_count, frame height, frame width, 3]` | the frame of the last render before each observation, as GRF's observation `frame` holds it, after the last reset or step |

The observation keeps its agent axis, and `final_observation` holds the last pixels of each ended
episode. The batch draws its frames at the size of its argument `frame_size`, `(width, height)` with
1280 by 720 as the default, whatever its `channel_dimensions`, and `pixels` and `pixels_gray`
shrink the whole frame to `channel_dimensions`, as GRF's wrapper does. The width is a multiple of 4,
and each side is at most 16,384, as in `cuju.gfootball`. As in GRF, the height stays 720 when only
the width is given, so `frame_size=800` gives frames of 800 by 720 and a frame of 800 by 450 takes
`frame_size=(800, 450)`. Internal tests compare a batch's frames, pixels and pixels_gray at six
sizes from 1920 by 1080 down to 96 by 72 with GRF's, on the CPU and on CUDA. `stacked=True` raises
`ConfigurationError` with the pixel representations, so a stack of frames is kept by the training
code, from copies of the observations, since each step overwrites them. A call of `frame()` on a
batch without `frames=True` and without a pixel representation raises `cuju.errors.LibraryError`.

`get_state` and `set_state` work on a rendering batch as on GRF's rendering engine. Each marks the
goal nets for drawing afresh at the next render, as GRF's do, the frame stays the last one drawn
until the next step, and the scoreboard keeps what it showed, since GRF's changes only at a goal.
Internal tests compare the frames of steps after loads of earlier states with GRF's frames after
the same loads, on both backends.

Either argument plays every environment in render mode with the referee's animations, as a rendering
GRF plays it, and gives each environment a renderer of its own, which draws every render of the
environment in GRF's order after each call, the internal steps of a stoppage and of a reset
included, so that each frame follows the same history of renders as GRF's. Internal tests find each
environment's frames equal, byte for byte on both backends, to the frames of GRF's same episodes
rendered in GRF's own environment after `env.render()`, the frame of every observation of six
episodes of eleven a side, five a side and an academy scenario.

Of the renders that a reset or a step records, only the last one before each observation is read.
The others, the internal steps of a stoppage or of a reset, still make every upload that a later
frame depends on, but draw no pixel, and such a render costs about two milliseconds where a drawn
frame costs a few hundred. A stoppage with a card renders about 150 times before play restarts, so
it costs about a third of a second of preparation besides its one drawn frame. A batch's frames
therefore hold one frame per step, where GRF's video of the same episode also holds the frames of
the internal steps, and a video with every frame comes from `write_video` in `cuju.gfootball`.

### Workers and threads

The environments of a batch draw side by side, each renderer on a worker thread of its own.
The key `render_workers` of the [machine config](installation.md#machine-config) sets how many
environments draw at once, and `render_threads` the threads of each renderer. By default every environment gets a worker, up to one worker per processor that the
process may run on, and the processors left over go to the threads of each renderer, so a batch of
four environments on twelve processors draws four frames at once with three threads each. Several
frames drawn at once keep the processors busier than one frame split into tiles, since each pass of
a frame ends on the slowest of its tiles. No split changes a frame, which internal tests check for
several of them.

| Setting | Default | Meaning |
| --- | --- | --- |
| key `render_workers` | 0, which gives one per environment, up to the processors, or the processors divided by `render_threads` | the renderers of a batch that draw at once |
| key `render_threads` | 0, which gives the processors left to each worker, or for `cuju.gfootball` the processors up to four | the threads that draw each frame |
| variable `CUJU_GRF_MEDIA` | the installed cuju-grf-media | a folder of GRF's media |
| variable `CUJU_RENDER_LIBRARY` | the library in cuju's folder | another build of libcuju_render |

The two keys take effect when a batch or a rendering engine creates its renderers, and
`CUJU_RENDER_LIBRARY` when the process first loads the library. A TOML snippet in the file that
`CUJU_CONFIG` names, or in the default path that [Installation](installation.md#machine-config)
gives, sets the keys.

```toml
[run]
render_workers = 4
render_threads = 3
```

### Memory

Each environment of a rendering batch holds about 200 MB, the buffers of its passes, among them the
shadow map of 2048 by 2048 pixels and the geometry and light buffers of eight bytes per pixel, and
the textures and meshes of its match. A batch of 16 environments therefore takes about 3.2 GB of host
memory, and the number of environments that fit is usually set by the memory before the processors.
The renderers of a process share what never changes, GRF's images and meshes as decoded, the
procedural pitch, the glyphs and the mipmap chains of every texture, which each process makes once
and holds once for all its renderers.

The first match of a process takes about three seconds to prepare, most of it for the pitch and
the mipmaps, and the first match of each later renderer about a third of a second, so the creation
of a batch of 16 environments takes a few seconds before its first frame.

### On the GPU

On the GPU the simulation and the extraction of each render's RenderState run on the device, and
one copy per environment brings the states of a step to the host, where the renderers draw. The
frames do not depend on the backend, and the pixel observations and frames are copied to the device
of the batch after each step. A batch that renders steps with `step`, and refuses `step_async` and
`torch.compile`, since every step waits for its frames.

## Render mode and the match

A rendering GRF plays other matches than one that does not. Its referee acts out every stoppage,
and its graphics keep cached positions of the players' skeletons that a set piece leaves behind for
a tick. cuju's engine plays those matches whenever it renders, so that a frame of cuju shows the
moment that GRF's frame shows.

**The referee's animations.** A rendering GRF resets every match with the flag `animations` on. The
referee then lets every stoppage play out in full, 2000 milliseconds before a set piece, 500 after
a goal and 10000 more for a card, where without the flag the clock skips that wait. The waits are
internal steps of the match, each simulated and, in render mode, rendered, so a rendering episode
holds more ticks of simulation than an unrendered one of the same length in steps.

**The cached roots.** GRF's graphics observe the nodes of every player's and official's skeleton,
and each change of a node recomputes the cached position and rotation of the node and of its root
at once, while the reset of a player's position for a set piece moves the root without that
recomputation. Until the next change of the root, the joints of the team processed first and of the
officials derive from the root where it stood, and a throw-in taker of that team can place the ball
from there for one tick. Every render point of GRF, after each step and at each render, refreshes
or reads these caches, so in render mode the renders take part in the match.

**Saves and loads.** A save or a load of a state marks every cache to compute afresh and makes the
next render draw the nets again, so `get_state` in a rendering match changes the match that
follows, as in GRF.

**The render switch.** GRF's `render()` on an engine that does not render saves the state, starts a
rendering engine, resets it and loads the state into it. GRF's loader mirrors the ball predictions
of the AI's mental images once more than the rest of the match when `reverse_team_processing` is on,
for each image the fresh engine did not hold, and `cuju.gfootball` loads states as GRF does, so its
switched matches equal GRF's.

The engine of `cuju.gfootball` plays render mode whenever its `game_config.render` is on and a
renderer opened at `start_game`, the rendering engine of GRF's render switch included. A batch plays
it in every environment when it renders. Internal tests find the matches of the clean reference
build of GRF, rendered, equal to cuju's in every state, observation, reward and done flag, on the
switch at the first reset and on the switch during an episode.

Two practical consequences follow. A rendering batch and an unrendered one with the same seed play
different matches, which part once the referee acts out a stoppage, so a policy trained on pixels
and evaluated on vectors meets the same game in other matches, and a comparison of two runs keeps
both in the same mode. The simulation of render mode costs little next to the drawing, a few more
ticks per stoppage and the work of the caches, so the cost of a rendering batch is almost entirely
the frames.

## Speed

On one core of a laptop's AMD Ryzen 5 7535HS, running at about 2.3 GHz, the renderer draws a frame
of 1280 by 720 pixels of an eleven a side match in about 0.56 seconds, 1.8 frames per second. With
two threads a frame takes about 0.30 seconds, with four about 0.16, and with eight or twelve about
0.10. These are averages over every frame of an episode, measured while the laptop was otherwise
idle. On the same laptop at the same speed, GRF's own frames on llvmpipe take about 0.72 seconds on
one core and 0.22 with four threads, against 13 milliseconds for a step without rendering, so the
renderer draws frames about 1.3 times as fast as llvmpipe on one core and 1.4 times as fast with four
threads. The laptop's processor runs anywhere from about 2.3 to 4.5 GHz, depending on its power
settings and temperature, and every figure here scales with that speed, so measurements taken at
different moments can differ by as much as half.

On one thread about three fifths of a frame goes to the accumulation of light, the ambient pass
with its 32 occlusion samples per pixel and the sun's light with its filtered lookups in the shadow
map. About a fifth goes to the G-buffer, whose materials take anisotropic samples of their
textures, and the shadow map, the post-processing and the menus take most of the rest. The work
that only one thread does for a frame, the menus, the uploads, the bookkeeping of the scene and the
start of each pass, takes about five milliseconds with twelve threads.

Most of the speed comes from shading eight pixels at a time with the AVX2 instructions that
llvmpipe's own code runs, in the texture samplers and in the geometry, lighting and post-processing
passes. Each pixel goes through the same operations in the same order as it would one at a time, so
the bits do not move, and the texture samplers also skip the colour channels that a pass never reads.
The occlusion samples of the ambient pass and the probes of the shadow lookups run as several
independent chains at once, each step for all of them before the next, so that the processor
overlaps their long runs of dependent divisions and loads. The shadow lookups compare each texel's
16 bits with a threshold found once per pixel, which gives the same answers as llvmpipe's
comparison of floating-point depths for every possible reference value, a fact checked over all of
them.

The renderer also prepares the triangles of one pass while its threads still draw the previous pass,
and it splits that preparation, the vertex shading, the setup and the binning of the triangles,
among its threads in runs of draws, each into a list of its own that the tiles then read in the
order of the draws. A pass also starts drawing while an earlier one still draws, when neither reads
what the other writes, as the shadow map and the G-buffer do. The players' skinning runs on the
threads as well. The laptop has six cores, each running two threads, and beyond six threads the
second thread of a core adds little, since the shading keeps the core's arithmetic busy, so twelve
threads draw a frame about as fast as eight. A batch of several environments uses the processors
better than one frame split into more tiles. Processors without AVX2 take a portable path that
gives the same frames, more slowly. The vector code stops at AVX2, the widest that the laptop of
these figures has, since only a machine with AVX-512 could prove such code against GRF's bits.

On the same laptop, a batch of eight environments draws about nine to ten frames a second in all
and a batch of 32 about ten, about one and a half times what the same environments draw one after
another with four threads each. The default split of workers and threads does as well as any other
there, within the spread of the measurements, for both sizes. The renders that nobody reads cost
about two milliseconds each, and the first match of a process about three seconds of preparation.

## Training on pixels

Pixels cost far more than the other representations, and a plan for training on them starts from the
cost of a frame. At about ten frames a second on a laptop of 12 processors, a million
observations take more than a day, and a machine with more cores scales the rate with its processors,
since each environment draws on a worker of its own. The simulation of the same million steps takes
minutes. Pixel observations therefore suit experiments whose question is the picture itself, such as
a policy that reads GRF's frames, an evaluation of such a policy on GRF's exact inputs, or a study
of what a network learns from pixels.

Several choices keep the cost down.

- **The SMM in place of pixels.** `representation='extracted'` gives GRF's super mini map, a picture
    of the pitch drawn on a grid of 96 by 72 cells, at the speed of the simulation. It suits a
    convolutional network as pixels do, and GRF's own baselines trained on it.
- **Environments per processor.** With at least as many environments as processors, the default
    split draws one frame per processor at a time, one thread each, which uses the processors best,
    while a smaller batch spreads the spare processors over the threads of each frame.
- **Memory first.** At about 200 MB per environment, the host memory usually bounds the batch, so
    the count of environments is chosen from the memory and the processors are then shared out.
- **Gray frames.** `pixels_gray` holds one byte per pixel where `pixels` holds three, a third of
    the memory of a rollout buffer, from the same frames at the same cost of drawing.
- **One mode throughout.** Training, evaluation and comparison run in the same mode, all rendering
    or all unrendered, since the two play different matches, as the
    [section above](#render-mode-and-the-match) explains.
- **Frames for recording.** `frames=True` with a vector representation records evaluation
    episodes for viewing while the policy reads vectors, and the matches are then those of a
    rendering GRF.

A stack of the last four frames, which GRF's `stacked=True` builds for pixels, is kept by the
training code, as in this loop, which also writes the full frames of environment 0 as a video with
OpenCV.

```python
import cv2
import torch

import cuju

batch = cuju.BatchEnvironment(
    '11_vs_11_stochastic',
    environment_count=12,
    device='cpu',
    representation='pixels_gray',
    frames=True,
)
observation = batch.reset()  # [12, 1, 72, 96, 1]
stack = observation.repeat(1, 1, 1, 1, 4)  # [12, 1, 72, 96, 4], the oldest first
video = cv2.VideoWriter('match.avi', cv2.VideoWriter_fourcc(*'MJPG'), 10, (1280, 720))
for _ in range(300):
    actions = policy(stack)  # [12, 1] actions
    observation, reward, done, information = batch.step(actions)
    stack = torch.cat([stack[..., 1:], observation], dim=-1)
    stack[done] = observation[done].repeat(1, 1, 1, 1, 4)
    video.write(cv2.cvtColor(batch.frame()[0].cpu().numpy(), cv2.COLOR_RGB2BGR))
video.release()
```

The frames are the same for every number of workers, threads and environments and on both backends,
so a run on pixels reproduces on another machine with the same seed, as any run of cuju does.

## GRF's media

cuju's software renderer draws GRF's frames from GRF's own media, the models, textures, menu
images and font that GRF's engine reads when it renders. They come in the package cuju-grf-media,
which the extras `pixels` and `gfootball-video` install, as [Installation](installation.md#wheel-from-pypi)
shows.

```sh
pip install 'cuju-engine[pixels]'
```

### What the package holds

The package holds 106 files of GRF's commit 3d9e754, copied byte for byte, 89,209,401 bytes in
all, and one file made from GRF's font, the glyph atlas `fonts/caption_glyphs.bin` of 3,850,034
bytes. The set of GRF's files is exactly what GRF reads for a rendered match. It follows GRF's load
paths in its engine sources, and internal tests checked it against every texture that GRF uploads
to OpenGL during rendered episodes, where each upload of an image at its own size matched one of
the files byte for byte once decoded. It covers the player model with its skeleton, kits, skins and
five hairstyles in three colours, the ball, the goals and their netting, the referee's cards, the
stadium with its walls, crowd and floor, the light of the sun, the twelve placeholder maps and the
two sources of the procedural pitch, the 16 adboards, the two team kits and logos, the radar, the
scoreboard and the league badge of the 2D overlay, the picture of the loading page that a reset
draws in its first frame, the noise texture of the ambient occlusion, and Alegreya Sans SC
ExtraBold, the font of every caption.

GRF draws its captions with SDL_ttf, which shapes the text with HarfBuzz and rasterizes the glyphs
with FreeType. The glyph atlas holds every glyph of the font as those libraries draw it, filled and
outlined, with the advances, the kerning of every pair of characters, and the ligatures and
contextual forms that HarfBuzz applies, so cuju's captions equal GRF's without the three libraries.
The release makes the atlas with `ci/make_glyph_atlas.py` from the font and the versions of SDL_ttf,
FreeType and HarfBuzz that GRF's reference build links, and the script that fills the package
refuses an atlas whose SHA-256 differs from its pin.

Each of GRF's files keeps GRF's path below the package's folder, so
`data/media/textures/adboards/ad_stark01.bmp` is GRF's
`third_party/gfootball_engine/data/media/textures/adboards/ad_stark01.bmp`, and
`fonts/AlegreyaSansSC-ExtraBold.ttf` is GRF's `third_party/fonts/AlegreyaSansSC-ExtraBold.ttf`. The
images are the BMP files that GRF ships, which GRF's loader decodes by reordering the colour
channels and nothing more, so a texel that cuju samples is the texel that GRF samples. Keeping the
files as they are, rather than converting them to a smaller format, lets their checksums prove
that they are GRF's, and leaves no decoder between GRF's bytes and cuju's frames. The price is the
installed size, about 93 MB, of which the pitch's overlay image alone is 33.5 MB and the six kit
textures 25 MB, while the wheel, which compresses each file, downloads at about 9 MB.

GRF's animations stay in cuju's engine, which carries the poses in its own tables, and the
arithmetic of GRF's shaders is part of cuju's renderer, which runs its own transcription of what
GRF's shaders compile to. Neither is therefore among the files. Files of GRF's data that no
rendered match reads, such as the hairstyles and hair colours that neither built in team uses,
stay out as well.

### Adboards

GRF draws the 23 adboards of its stadium from the images of one folder. It sorts the 16 file
names, and each board takes the image at an index drawn from the match's seeded random numbers, so
the number of files and their order decide which image shows on which board. The draw happens when
an engine builds its first match, the internal match of `start_game`, so the boards stay as they are
for every later match of the engine. The package ships all
16 boards exactly as GRF does, and its manifest lists them in GRF's order, so every board of
cuju's frame shows the image that GRF's frame shows. Most of the boards show the names and logos of
third parties, game studios, record labels and a city among them, which may be trademarks of their
owners. The dedication to the public domain concerns copyright alone, and cuju ships the boards for
the parity of its frames with GRF's, with no affiliation to those owners, as the package's NOTICE
says.

### Licenses

The models, textures, scene files and menu images descend from Gameplay Football by Bastiaan
Konings Schuiling, who dedicated it to the public domain under the Unlicense. Google Research
Football modified that game and converted its images to BMP, and Google's changes are licensed
under the Apache License 2.0, as are the two team logos, which Google added. The font, Alegreya
Sans SC ExtraBold 2.003, Copyright 2013 The Alegreya Sans Project Authors, is licensed under the
SIL Open Font License 1.1, which lets it be bundled with software and asks that it never be sold
on its own, and the glyph atlas, drawn from the font, carries the same license. The wheel's license
expression is therefore `Apache-2.0 AND Unlicense AND OFL-1.1`, and it carries the three license
texts and a NOTICE. The manifest names the license of every file, so a project that redistributes
some of the files can see which terms each one carries.

The renderer itself, in cuju's wheel, ports code of Mesa, under the MIT License, of SDL, SDL2_gfx
and SDL_ttf, under the zlib License, and of OpenCV, under the Apache License 2.0, and cuju's NOTICE
and `LICENSES` folder carry their notices and texts.

### Checking the files

`cuju_grf_media/manifest.json` names each file, its path in GRF or the script that made it, its
size, its SHA-256, its license and its role, and `cuju_grf_media/SHA256SUMS` lists the SHA-256 of
every file and of the manifest in the format of `sha256sum`. Both sit in the folder of the installed
package, which Python prints, and `sha256sum -c` checks every file there without cuju.

```sh
cd "$(python -c 'import cuju_grf_media, os; print(os.path.dirname(cuju_grf_media.__file__))')"
sha256sum -c SHA256SUMS
```

The checksums of the originals are pinned in cuju's repository, in
`packaging/cuju-grf-media/sources.txt`, and the script that fills the package refuses a GRF
checkout in which any file differs from its pin. The release builds the wheel twice and compares
the two byte for byte, as it does for cuju's own wheels, so the published wheel can be rebuilt
from the tagged commit and a checkout of GRF, as [GRF's media from a clone](installation.md#grfs-media-from-a-clone)
shows.

Where the package is missing, every part of cuju that reads the media raises
`cuju.errors.MissingFileError`, which is also a `FileNotFoundError`, with one message that names
`pip install 'cuju-engine[pixels]'`. An installed package whose manifest is of another format than the
one that the installed cuju reads raises the same error, and the same command then installs the
version that fits.

## See also

- [Differences from GRF](differences-from-grf.md#grfs-frames), the points where rendering differs from GRF
- [Migrating from GRF](migrating-from-grf.md#pixel-observations-and-frames), moving code that renders
- [Exactness](exactness.md), the claim for the matches, which render mode extends
- [Performance](performance.md), the throughput of the simulation

See the documentation for the browser viewer in the [render repository](https://github.com/cuju-football/render).
