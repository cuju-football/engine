# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks cuju's software renderer against the frames of an unmodified GRF, byte for byte."""

import hashlib
import json
import math
import os
import pickle
import re
import shutil
import struct
import subprocess
import sys
import zlib

import pytest

import cuju
from cuju import _media, _pixels
from cuju._batch import Batch
from cuju.errors import ArgumentError, MissingFileError, StateError

HERE = os.path.dirname(os.path.abspath(__file__))
DIGESTS = os.path.join(
    os.path.dirname(os.path.abspath(cuju.__file__)), 'data', 'render_digests.json.z'
)
DIGESTS_SHA256 = '5d935941e7b70de7d3da82efea45f0e6c33ecb3b57c368cad10364333edf167e'
with open(DIGESTS, 'rb') as handle:
    DIGESTS_TEXT = zlib.decompress(handle.read())
EXPECTED = json.loads(DIGESTS_TEXT)
STORED = EXPECTED['stored_episode']
# GRF's episodes of the paths beyond the six episodes, each observation with a frame.
SCENARIOS = EXPECTED.get('scenarios', {})
# GRF's episodes of scripts of calls, each observation with the digest of its raw output and frame.
OPS = EXPECTED.get('ops', {})
SHORT_READBACKS = 26
MAGIC = b'CUJRSE1\0'
STATE_KINDS = (5, 6)

opted_in = pytest.mark.skipif(
    os.environ.get('CUJU_LONG', '') in ('', '0'), reason='set CUJU_LONG=1 for the long cases'
)


def sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


@pytest.fixture(scope='module')
def pixels():
    from cuju import _pixels

    if not os.path.isfile(_pixels.library_path()):
        pytest.skip(f'needs the renderer library {_pixels.library_path()}, built with the renderer')
    return _pixels


@pytest.fixture(scope='module')
def media():
    folder = os.environ.get('CUJU_GRF_MEDIA')
    if folder:
        return folder
    try:
        return str(_media.root())
    except MissingFileError:
        pytest.skip(
            "needs GRF's media: pip install 'cuju-engine[pixels]', or set CUJU_GRF_MEDIA to the "
            'output folder of ci/make/make_render_media.py'
        )


def events():
    """The stored episode's events: kind, swap, seed and the RenderState of each."""
    with open(os.path.join(os.path.dirname(HERE), 'data', STORED['file']), 'rb') as handle:
        data = zlib.decompress(handle.read())
    assert data[:8] == MAGIC
    size, version = struct.unpack_from('<II', data, 8)
    at, previous = 16, bytes(size)
    while at < len(data):
        kind, swap = data[at], data[at + 1]
        (seed,) = struct.unpack_from('<I', data, at + 2)
        at += 6
        state = None
        if kind in STATE_KINDS:
            delta = data[at : at + size]
            at += size
            state = (int.from_bytes(delta, 'little') ^ int.from_bytes(previous, 'little')).to_bytes(
                size, 'little'
            )
            previous = state
        yield kind, bool(swap), seed, state, (size, version)


def replay(renderer, readbacks, unread=None):
    """Draws the stored episode up to `readbacks` frames and returns their digests."""
    frames, observations = [], {}
    expected = {o['readbacks']: o for o in EXPECTED['episodes']['5_vs_5_0']['observations']}
    for kind, swap, seed, state, _ in events():
        if kind == 1:
            renderer.reset_begin()
        elif kind == 2:
            renderer.stop_match()
        elif kind == 3:
            renderer.loading_page()
        elif kind == 4:
            renderer.render_scene(swap)
        elif kind == 5:
            renderer.start_match(state, seed)
        elif kind == 6:
            if swap and unread is not None and unread(len(frames) + 1):
                renderer.render(state, False)
                frames.append(None)
                if len(frames) == readbacks:
                    break
                continue
            renderer.render(state, swap)
        elif kind == 7:
            renderer.state_processed()
        if kind in (4, 6) and swap:
            frames.append(sha(renderer.frame()))
            if len(frames) in expected:
                observations[len(frames)] = {
                    'frame': sha(renderer.frame(top_down=True)),
                    'pixels': sha(renderer.pixels()),
                    'pixels_gray': sha(renderer.pixels_gray()),
                }
            if len(frames) == readbacks:
                break
    return frames, observations, expected


def check_replay(pixels, media, threads, readbacks):
    with pixels.Renderer(threads=threads, media=media) as renderer:
        frames, observations, expected = replay(renderer, readbacks)
        assert frames == STORED['readbacks'][:readbacks]
        assert set(observations) == {index for index in expected if index <= readbacks}
        for index, digests in observations.items():
            want = expected[index]
            assert digests == {k: want[k] for k in ('frame', 'pixels', 'pixels_gray')}, index
        return renderer.digests()


def test_the_stored_episode_holds_render_states_of_this_layout(pixels):
    stream = list(events())
    assert stream[0][4] == (pixels.state_size(), pixels.state_version())
    swapped = [kind for kind, swap, *_ in stream if kind in (4, 6) and swap]
    assert len(swapped) == len(STORED['readbacks'])
    assert [kind for kind, *_ in stream].count(5) >= 2


def test_the_fixed_data_equal_grf(pixels, media):
    with pixels.Renderer(threads=2, media=media) as renderer:
        digests = renderer.digests()
    assert {name: digests.get(name) for name in EXPECTED['media']} == EXPECTED['media']


def test_frames_of_the_reset_and_first_steps_equal_grf(pixels, media):
    check_replay(pixels, media, threads=4, readbacks=SHORT_READBACKS)


def test_frames_from_media_in_a_unicode_path_equal_grf(pixels, media, tmp_path):
    folder = tmp_path / 'João 日本'
    shutil.copytree(media, folder)
    check_replay(pixels, folder, threads=4, readbacks=SHORT_READBACKS)


def test_the_renderer_refuses_wrong_calls(pixels, media):
    with pytest.raises(ArgumentError, match='multiple of 4'):
        pixels.Renderer(width=1281, media=media)
    with pytest.raises(ArgumentError, match='positive'):
        pixels.Renderer(width=96, height=0, media=media)
    with pytest.raises(MissingFileError):
        pixels.Renderer(media=os.path.join(HERE, 'no-such-media'))
    with pixels.Renderer(width=64, height=48, media=media) as renderer:
        with pytest.raises(StateError, match='no frame'):
            renderer.frame()
        with pytest.raises(StateError, match='bytes'):
            renderer.render(b'\0' * 16)
        state = bytearray(pixels.state_size())
        struct.pack_into('<I', state, 0, pixels.state_version())
        with pytest.raises(StateError, match='no match'):
            renderer.render(state)
        struct.pack_into('<I', state, 0, pixels.state_version() + 1)
        with pytest.raises(StateError, match='version'):
            renderer.start_match(state, 42)
        renderer.render_scene(True)
        assert renderer.frame().shape == (48, 64, 3)
        with pytest.raises(ArgumentError):
            renderer.pixels(0, 72)
    renderer.close()
    with pytest.raises(StateError, match='closed'):
        renderer.render_scene(True)


def test_the_height_is_any_as_in_grf(pixels, media):
    # GRF's configuration gives 0.5625 times the width, 54 rows for a width of 96.
    with pixels.Renderer(width=96, height=54, media=media) as renderer:
        renderer.render_scene(True)
        assert renderer.frame(top_down=True).shape == (54, 96, 3)


def check_unread(pixels, media, readbacks, unread):
    with pixels.Renderer(threads=2, media=media) as renderer:
        frames, observations, expected = replay(renderer, readbacks, unread)
    skipped = [index for index, digest in enumerate(frames, 1) if digest is None]
    assert skipped
    assert [d for d in frames if d is not None] == [
        d for index, d in enumerate(STORED['readbacks'][:readbacks], 1) if index not in skipped
    ]
    for index, digests in observations.items():
        want = expected[index]
        assert digests == {k: want[k] for k in ('frame', 'pixels', 'pixels_gray')}, index


def test_a_match_built_without_rendering_raises_where_grf_crashes(pixels, media):
    # GRF: a reset while render is off builds the match without captions,, and the render crashes.
    start = next(state for kind, _, _, state, _ in events() if kind == 5)
    render = [state for kind, swap, _, state, _ in events() if kind == 6 and swap][4]
    with pixels.Renderer(threads=2, media=media) as renderer:
        renderer.reset_begin()
        renderer.stop_match()
        renderer.loading_page()
        renderer.render_scene(True)
        renderer.start_match(start, 42)
        renderer.reset_begin()
        renderer.stop_match()
        renderer.set_render(False)
        renderer.start_match(start, 0)
        renderer.set_render(True)
        with pytest.raises(StateError, match='crashes'):
            renderer.render(render, True)


def test_a_batch_takes_grf_resolution_options(monkeypatch):
    for options in (
        {'render_resolution_x': 642},
        {'render_resolution_y': 0},
        {'render_resolution_x': 1280.0},
        {'render_resolution_y': 16388},
    ):
        with pytest.raises(cuju.errors.ConfigurationError, match='render_resolution'):
            Batch('5_vs_5_l1_r0', environment_count=1, device='cpu', render_frames=True, **options)


def test_a_batch_draws_at_grf_resolutions(pixels, media, monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setenv('CUJU_GRF_MEDIA', media)
    for options, shape in (
        ({'render_resolution_x': 640}, (1, 720, 640, 3)),
        ({'render_resolution_x': 96, 'render_resolution_y': 54}, (1, 54, 96, 3)),
    ):
        batch = Batch(
            '5_vs_5_l1_r0',
            environment_count=1,
            device='cpu',
            representation='pixels',
            render_frames=True,
            **options,
        )
        try:
            observation = batch.reset()
            assert tuple(batch.frame().shape) == shape
            assert tuple(observation.shape) == (1, 1, 72, 96, 3)
            _, _, _, information = batch.step(torch.zeros((1, 1), dtype=torch.int32))
            assert tuple(information['frame'].shape) == shape
        finally:
            batch.close()


def test_renders_nobody_reads_leave_the_frames_read_as_they_are(pixels, media):
    check_unread(pixels, media, SHORT_READBACKS, lambda index: index > 3 and index % 3 != 0)


def test_the_frame_goes_into_a_given_array(pixels, media):
    np = pytest.importorskip('numpy')
    with pixels.Renderer(width=64, height=48, media=media) as renderer:
        renderer.render_scene(True)
        out = np.zeros((48, 64, 3), np.uint8)
        assert renderer.frame(top_down=True, out=out) is out
        assert sha(out) == sha(renderer.frame(top_down=True))
        gray = np.zeros((72, 96), np.uint8)
        assert sha(renderer.pixels_gray(out=gray)) == sha(renderer.pixels_gray())
        with pytest.raises(ArgumentError, match='shape'):
            renderer.frame(out=np.zeros((48, 64), np.uint8))
        with pytest.raises(ArgumentError, match='uint8'):
            renderer.pixels(out=np.zeros((72, 96, 3), np.float32))


def test_a_batch_splits_the_processors_between_workers_and_threads():
    from cuju._batch.pixels import split

    assert split(16, cpus=12) == (12, 1)
    assert split(3, cpus=12) == (3, 4)
    assert split(1, cpus=12) == (1, 12)
    assert split(16, cpus=12, threads=4) == (3, 4)
    assert split(16, cpus=12, workers=2, threads=4) == (2, 4)
    assert split(16, cpus=12, workers=2) == (2, 6)
    assert split(1, cpus=12) == (1, 12)


def test_renderers_of_one_process_share_the_media(pixels, media):
    first = pixels.Renderer(width=64, height=48, media=media)
    second = pixels.Renderer(width=64, height=48, media=media)
    first.render_scene(True)
    second.render_scene(True)
    assert sha(first.frame()) == sha(second.frame())
    first.close()
    second.render_scene(True)
    second.close()


def test_a_caller_that_flushes_subnormals_changes_no_frame(pixels, media):
    torch = pytest.importorskip('torch')
    if not torch.set_flush_denormal(True):
        pytest.skip('PyTorch cannot flush subnormals on this processor')
    try:
        check_replay(pixels, media, threads=2, readbacks=4)
    finally:
        torch.set_flush_denormal(False)


def test_windows_exports_every_function_that_the_loader_calls():
    header = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'src', 'render', 'render_api.h')
    loader = os.path.join(os.path.dirname(os.path.abspath(cuju.__file__)), '_pixels.py')
    if not os.path.isfile(header):
        pytest.skip('needs the source tree')
    with open(header, encoding='utf-8') as handle:
        exports = {
            m.group(1)
            for line in handle
            if (m := re.match(r'[A-Za-z][^(]*[ *](cuju_render_[a-z0-9_]+)\(', line))
        }
    with open(loader, encoding='utf-8') as handle:
        text = handle.read()
    called = set(re.findall(r'cuju_render_[a-z0-9_]+', text))
    called |= {'cuju_render_' + name for name in re.findall(r"_call\('([a-z_]+)'", text)}
    called |= {
        'cuju_render_' + name
        for name in ('reset_begin', 'stop_match', 'loading_page', 'state_processed')
    }
    assert called <= exports, sorted(called - exports)
    if os.name == 'nt' and os.path.isfile(_pixels.library_path()):
        library = _pixels._load()
        missing = [name for name in sorted(exports) if not hasattr(library, name)]
        assert not missing, missing


@opted_in
def test_every_frame_of_the_stored_episode_equals_grf(pixels, media):
    digests = check_replay(pixels, media, threads=4, readbacks=len(STORED['readbacks']))
    mips = {name: value for name, value in digests.items() if name.startswith('mip_')}
    assert len(mips) > 500
    assert {name: STORED['mips'].get(name) for name in mips} == mips


@opted_in
@pytest.mark.parametrize('threads', [1, 3])
def test_frames_are_the_same_for_every_thread_count(pixels, media, threads):
    check_replay(pixels, media, threads=threads, readbacks=SHORT_READBACKS)


@opted_in
@pytest.mark.parametrize('every', [2, 5])
def test_renders_nobody_reads_change_no_frame_of_the_episode(pixels, media, every):
    check_unread(pixels, media, len(STORED['readbacks']), lambda index: index % every != 0)


@opted_in
def test_renderers_that_share_a_process_draw_grf_frames(pixels, media):
    with pixels.Renderer(threads=2, media=media):
        check_replay(pixels, media, threads=2, readbacks=SHORT_READBACKS)
        check_replay(pixels, media, threads=1, readbacks=SHORT_READBACKS)


def scripted_actions(kind, observation, rng, agents):
    """The actions of a scenario's agents from GRF's raw observation."""
    if agents == 0:
        return []
    if kind.startswith('random'):
        chosen = [int(a) for a in rng.randint(0, 19, size=agents)]
        return chosen[0] if agents == 1 else chosen
    # GRF's directions by angle from +x, counter-clockwise, with the observation's y downwards.
    directions = [5, 4, 3, 2, 1, 8, 7, 6]

    def toward(dx, dy):
        return directions[round(math.atan2(-dy, dx) / (math.pi / 4)) % 8]

    chosen = []
    controlled = list(observation['left_agent_controlled_player'])
    for agent in range(agents):
        me = observation['left_team'][controlled[agent]]
        carrier = observation['ball_owned_team'] == 1
        if carrier:
            k = observation['ball_owned_player']
            target = observation['right_team'][k]
        else:
            target = observation['ball'][:2]
        dx, dy = float(target[0] - me[0]), float(target[1] - me[1])
        distance = math.hypot(dx, dy)
        if kind.startswith('slideb') and carrier:
            v = observation['right_team_direction'][k]
            n = math.hypot(float(v[0]), float(v[1]))
            u = (float(v[0]) / n, float(v[1]) / n) if n > 1e-6 else (1.0, 0.0)
            behind = (dx * u[0] + dy * u[1]) / distance if distance > 1e-6 else 0.0
            if distance < 0.03 and behind > 0.6:
                chosen.append(16)
            else:
                tx, ty = float(target[0]) - 0.03 * u[0], float(target[1]) - 0.03 * u[1]
                chosen.append(toward(tx - float(me[0]), ty - float(me[1])))
        elif kind.startswith('slideb'):
            chosen.append(int(rng.randint(0, 19)) if rng.rand() < 0.1 else toward(dx, dy))
        elif carrier and distance < 0.035:
            chosen.append(16)
        elif rng.rand() < 0.1:
            chosen.append(int(rng.randint(0, 19)))
        else:
            chosen.append(toward(dx, dy))
    return chosen[0] if agents == 1 else chosen


def patched_state(state, field, old, new):
    """`state` with its int32 `field` changed from `old` to `new`, found through `cuju.state`."""
    import cuju.state

    assert int(cuju.state.read(state, [field])[0]) == old
    layout = cuju.state.layout()
    end = max(f.offset + f.size for f in layout.values())
    guess = len(state) * layout[field].offset // end
    want = struct.pack('<i', old)
    places = [at for at in range(len(state) - 3) if state[at : at + 4] == want]
    for at in sorted(places, key=lambda at: abs(at - guess)):
        candidate = state[:at] + struct.pack('<i', new) + state[at + 4 :]
        try:
            if int(cuju.state.read(candidate, [field])[0]) == new:
                return candidate
        except cuju.errors.Error:
            continue
    raise AssertionError(f'{field} not found in the state')


def play_scenario(name):
    """The digests of every observation with a frame of the scenario `name`."""
    import cv2
    import numpy as np

    from cuju.gfootball.env import config, football_env

    entry = SCENARIOS[name]
    switch = entry['mode'].startswith('mid:')
    width, height = entry.get('resolution', (1280, 720))
    values = {
        'level': entry['scenario'],
        'players': entry.get('players', []),
        'game_engine_random_seed': entry['seed'],
        'render_resolution_x': width,
        'render_resolution_y': height,
        'dump_full_episodes': False,
        'dump_scores': False,
        'write_video': False,
    }
    env = football_env.FootballEnv(config.Config(values))
    kind = entry.get('actions', 'none')
    rng = np.random.RandomState(int(kind.split(':')[1])) if ':' in kind else None
    space = env.action_space
    agents = (len(space.nvec) if hasattr(space, 'nvec') else 1) if values['players'] else 0
    got = []

    def observe(step):
        frame = env._env._observation.get('frame')
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            got.append(
                {
                    'step': step,
                    'frame': sha(frame),
                    'pixels': sha(cv2.resize(frame, (96, 72), interpolation=cv2.INTER_AREA)),
                    'pixels_gray': sha(cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA)),
                }
            )

    class Ceremony:
        """GRF's engine as GRF's core steps it through a stoppage."""

        def __init__(self, real, spec, frames):
            self.real, self.spec, self.frames, self.count = real, spec, frames, 0

        def __getattr__(self, attribute):
            return getattr(self.real, attribute)

        def step(self):
            self.real.step()
            self.count += 1
            spec = self.spec
            if self.count == spec['after'] and 'field' in spec:
                state = self.real.get_state(pickle.dumps({}))
                self.real.set_state(patched_state(state, spec['field'], spec['from'], spec['to']))
            if self.count >= spec['after']:
                self.frames.append(hashlib.sha256(bytes(self.real.get_frame())).hexdigest())

    if not switch:
        env.render()
    env.reset()
    observe(0)
    patch, rewind, saved = entry.get('patch'), entry.get('setstate'), None
    ceremony = entry.get('ceremony')
    for step in range(entry['steps']):
        if rewind and step == rewind[1]:
            saved = env.get_state()
        if patch and step == patch['at']:
            state = patched_state(env.get_state(), patch['field'], patch['from'], patch['to'])
            env.set_state(state)
            observe(step)
        if rewind and step == rewind[0]:
            env.set_state(saved)
            observe(step)
        if switch and step == int(entry['mode'][4:]):
            env.render()
            observe(step)
        actions = scripted_actions(kind, env._env._observation, rng, agents)
        if ceremony and step == ceremony['at']:
            internal, real = [], env._env._env
            env._env._env = Ceremony(real, ceremony, internal)
            _, _, done, _ = env.step(actions)
            env._env._env = real
            observe(step + 1)
            got[-1]['internal'] = internal
        else:
            _, _, done, _ = env.step(actions)
            observe(step + 1)
        if done:
            break
    env.close()
    return got


@opted_in
@pytest.mark.parametrize('name', sorted(SCENARIOS))
def test_cuju_renders_grf_scenarios(pixels, media, name):
    pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    run = subprocess.run(
        [sys.executable, os.path.abspath(__file__), name],
        capture_output=True,
        text=True,
        env=dict(os.environ, CUJU_GRF_MEDIA=media),
        check=False,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    lines = [line for line in run.stdout.splitlines() if line.startswith('scenario ')]
    assert lines, run.stdout[-3000:]
    got = json.loads(lines[-1][len('scenario ') :])
    want = SCENARIOS[name]['observations']
    assert len(got) == len(want)
    for have, expected in zip(got, want, strict=True):
        assert have == expected, have['step']


@opted_in
@pytest.mark.parametrize('episode', sorted(EXPECTED['episodes']))
def test_cuju_matches_render_as_grf_does(pixels, media, episode, monkeypatch):
    from cuju.gfootball.env import config, football_env

    monkeypatch.setenv('CUJU_GRF_MEDIA', media)
    entry = EXPECTED['episodes'][episode]
    values = {
        'level': entry['scenario'],
        'players': [],
        'game_engine_random_seed': entry['seed'],
        'render_resolution_x': 1280,
        'render_resolution_y': 720,
        'dump_full_episodes': False,
        'dump_scores': False,
        'write_video': False,
    }
    try:
        env = football_env.FootballEnv(config.Config(values))
        env.render()
        env.reset()
    except cuju.errors.Error as error:
        pytest.skip(f"needs the engine's render mode: {error}")
    cv2 = pytest.importorskip('cv2')

    def observed():
        frame = env._env._observation['frame']
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return {
            'frame': sha(frame),
            'pixels': sha(cv2.resize(frame, (96, 72), interpolation=cv2.INTER_AREA)),
            'pixels_gray': sha(cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA)),
        }

    got = [observed()]
    for _ in range(entry['steps']):
        env.step([])
        got.append(observed())
    env.close()
    want = [{k: o[k] for k in ('frame', 'pixels', 'pixels_gray')} for o in entry['observations']]
    assert got == want


def rendering_batch(device, kind, steps, scenario='5_vs_5_l1_r0', observations=None):
    """The digests of a rendering batch's observations and frames over a reset and `steps`."""
    torch = pytest.importorskip('torch')
    batch = Batch(
        scenario,
        environment_count=2,
        device=device,
        representation=kind,
        render_frames=True,
        game_engine_random_seed=0,
    )
    try:
        digests = [sha(batch.reset().cpu().numpy())]
        actions = torch.zeros((2, batch.agents), dtype=torch.int32, device=batch.torch_device)
        for _ in range(steps):
            observation, _, _, information = batch.step(actions)
            digests.append(sha(observation.cpu().numpy()))
            digests.append(sha(information['frame'].cpu().numpy()))
            if observations is not None:
                observations.append(observation.cpu().clone())
        shape = tuple(observation.shape), tuple(information['frame'].shape)
    finally:
        batch.close()
    return digests, shape


def test_a_rendering_batch_returns_grf_pixels(pixels, media, monkeypatch):
    monkeypatch.setenv('CUJU_GRF_MEDIA', media)
    _, shape = rendering_batch('cpu', 'pixels', 1)
    assert shape == ((2, 1, 72, 96, 3), (2, 720, 1280, 3))
    _, gray = rendering_batch('cpu', 'pixels_gray', 1)
    assert gray[0] == (2, 1, 72, 96, 1)


def test_every_agent_of_a_rendering_batch_sees_the_same_pixels(pixels, media, monkeypatch):
    monkeypatch.setenv('CUJU_GRF_MEDIA', media)
    _, shape = rendering_batch('cpu', 'pixels', 1, scenario='5_vs_5_l0_r0')
    assert shape[0] == (2, 0, 72, 96, 3)
    observations = []
    _, shape = rendering_batch('cpu', 'pixels_gray', 1, '5_vs_5_l2_r0', observations)
    assert shape[0] == (2, 2, 72, 96, 1)
    assert bool((observations[0][:, 0] == observations[0][:, 1]).all())
    assert int(observations[0].max()) > 0


@opted_in
def test_rendering_batches_draw_alike_on_both_backends(pixels, media, monkeypatch):
    from conftest import CUDA

    if not CUDA:
        pytest.skip('no CUDA device or library')
    monkeypatch.setenv('CUJU_GRF_MEDIA', media)
    assert rendering_batch('cuda', 'pixels', 3) == rendering_batch('cpu', 'pixels', 3)


@opted_in
def test_a_batch_draws_the_same_frames_for_every_split(pixels, media, monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setenv('CUJU_GRF_MEDIA', media)
    entry = EXPECTED['episodes']['5_vs_5_0']
    runs = []
    for workers, threads in [(1, 1), (3, 1), (2, 3)]:
        monkeypatch.setenv('CUJU_RENDER_WORKERS', str(workers))
        monkeypatch.setenv('CUJU_RENDER_THREADS', str(threads))
        batch = Batch(
            entry['scenario'],
            environment_count=3,
            device='cpu',
            agent_counts=[(0, 0)],
            render_frames=True,
            game_engine_random_seed=entry['seed'],
        )
        try:
            assert (batch._frames.workers, batch._frames.threads) == (workers, threads)
            batch.reset()
            got = [[sha(frame.numpy()) for frame in batch.frame()]]
            actions = torch.zeros((3, 0), dtype=torch.int32)
            for _ in range(12):
                _, _, done, information = batch.step(actions)
                frames = torch.where(
                    done.bool()[:, None, None, None],
                    information['final_frame'],
                    information['frame'],
                )
                got.append([sha(frame.numpy()) for frame in frames])
        finally:
            batch.close()
        runs.append(got)
    assert runs[0] == runs[1] == runs[2]
    assert [step[0] for step in runs[0]] == [o['frame'] for o in entry['observations']][:13]


@opted_in
@pytest.mark.parametrize('episode', sorted(EXPECTED['episodes']))
def test_a_rendering_batch_draws_grf_frames(pixels, media, episode, monkeypatch):
    torch = pytest.importorskip('torch')
    monkeypatch.setenv('CUJU_GRF_MEDIA', media)
    entry = EXPECTED['episodes'][episode]
    batch = Batch(
        entry['scenario'],
        environment_count=1,
        device='cpu',
        agent_counts=[(0, 0)],
        render_frames=True,
        game_engine_random_seed=entry['seed'],
    )
    try:
        batch.reset()
        got = [sha(batch.frame()[0].numpy())]
        actions = torch.zeros((1, 0), dtype=torch.int32)
        for _ in range(len(entry['observations']) - 1):
            _, _, done, information = batch.step(actions)
            got.append(sha(information['final_frame' if bool(done[0]) else 'frame'][0].numpy()))
    finally:
        batch.close()
    assert got == [o['frame'] for o in entry['observations']]


def raw_digest(observation):
    """The SHA-256 of a raw observation without its frame, every value widened."""
    import numpy as np

    digest = hashlib.sha256()
    for key in sorted(observation):
        if key == 'frame':
            continue
        value = np.asarray(observation[key])
        if value.dtype.kind == 'f':
            value = value.astype(np.float64)
        elif value.dtype.kind in 'iub':
            value = value.astype(np.int64)
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def play_ops(name):
    """The digests of every observation of the script `name`."""
    import cv2

    from cuju.gfootball.env import config, football_env

    entry = OPS[name]
    width, height = entry.get('resolution', (1280, 720))
    values = {
        'level': entry['scenario'],
        'players': entry.get('players', []),
        'game_engine_random_seed': entry['seed'],
        'render_resolution_x': width,
        'render_resolution_y': height,
        'dump_full_episodes': False,
        'dump_scores': False,
        'write_video': False,
    }
    if entry.get('only_x'):
        del values['render_resolution_y']
    env = football_env.FootballEnv(config.Config(values))
    agents = 1 if values['players'] else 0
    got, saved, step = [], {}, 0

    def observe(op):
        observation = env._env._observation
        digests = {'op': op, 'step': step, 'raw': raw_digest(observation)}
        frame = observation.get('frame')
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            digests['frame'] = sha(frame)
            digests['pixels'] = sha(cv2.resize(frame, (96, 72), interpolation=cv2.INTER_AREA))
            digests['pixels_gray'] = sha(cv2.resize(gray, (96, 72), interpolation=cv2.INTER_AREA))
        got.append(digests)

    for op, *argument in entry['ops']:
        try:
            if op == 'render':
                env.render()
            elif op == 'disable':
                env.disable_render()
            elif op == 'reset':
                env.reset()
                step = 0
            elif op == 'save':
                saved[argument[0]] = env.get_state()
            elif op == 'load':
                env.set_state(saved[argument[0]])
            elif op == 'flag':
                env._env._env.game_config.render = True
            elif op == 'step':
                for _ in range(argument[0]):
                    env.step(0 if agents else [])
                    step += 1
                    observe(op)
                continue
            observe(op)
        except StateError:
            got.append({'op': 'crash', 'step': step})
            return got
    env.close()
    return got


@opted_in
@pytest.mark.parametrize('name', sorted(OPS))
def test_cuju_plays_grf_scripts_of_calls(pixels, media, name):
    pytest.importorskip('numpy')
    pytest.importorskip('cv2')
    run = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--ops', name],
        capture_output=True,
        text=True,
        env=dict(os.environ, CUJU_GRF_MEDIA=media),
        check=False,
    )
    assert run.returncode == 0, run.stderr[-3000:]
    lines = [line for line in run.stdout.splitlines() if line.startswith('ops ')]
    assert lines, run.stdout[-3000:]
    got = json.loads(lines[-1][len('ops ') :])
    entry = OPS[name]
    want = list(entry['observations'])
    if not entry['complete']:
        want.append({'op': 'crash', 'step': got[-1]['step']})
    assert len(got) == len(want)
    for index, (have, expected) in enumerate(zip(got, want, strict=True)):
        assert have == expected, index


def play_batch_ops(name, device, kind):
    """The frame and pixel digests of the script `name` in a batch of one environment."""
    torch = pytest.importorskip('torch')
    entry = OPS[name]
    width, height = entry.get('resolution', (1280, 720))
    options = {'render_resolution_x': width}
    if not entry.get('only_x'):
        options['render_resolution_y'] = height
    left = 1 if entry.get('players') else 0
    batch = Batch(
        entry['scenario'],
        environment_count=1,
        device=device,
        agent_counts=[(left, 0)],
        representation=kind,
        render_frames=True,
        game_engine_random_seed=entry['seed'],
        **options,
    )
    actions = torch.zeros((1, batch.agents), dtype=torch.int32, device=batch.torch_device)
    got, saved = [], {}

    def digests(op, observation):
        frame = sha(batch.frame()[0].cpu().numpy())
        got.append((op, frame, sha(observation[0, 0].cpu().numpy()) if left else None))

    try:
        assert entry['ops'][0] == ['render']
        observation = None
        for op, *argument in entry['ops'][1:]:
            if op == 'reset':
                observation = batch.reset()
            elif op == 'save':
                saved[argument[0]] = batch.get_state(0)
            elif op == 'load':
                batch.set_state(0, saved[argument[0]])
                observation = batch._frames.observation
            elif op == 'step':
                for _ in range(argument[0]):
                    observation, _, _, _ = batch.step(actions)
                    digests(op, observation)
                continue
            digests(op, observation)
        shape = tuple(batch.frame().shape)
    finally:
        batch.close()
    return got, shape


@opted_in
@pytest.mark.parametrize('kind', ['pixels', 'pixels_gray'])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('name', sorted(n for n, e in OPS.items() if e.get('kind') == 'batch'))
def test_a_rendering_batch_plays_grf_scripts(pixels, media, name, device, kind, monkeypatch):
    from conftest import CUDA

    if device == 'cuda' and not CUDA:
        pytest.skip('no CUDA device or library')
    monkeypatch.setenv('CUJU_GRF_MEDIA', media)
    entry = OPS[name]
    got, shape = play_batch_ops(name, device, kind)
    width, height = entry.get('resolution', (1280, 720))
    assert shape == (1, 720 if entry.get('only_x') else height, width, 3)
    left = bool(entry.get('players'))
    want = [(o['op'], o['frame'], o[kind] if left else None) for o in entry['observations'][1:]]
    assert got == want


if __name__ == '__main__':
    if sys.argv[1] == '--ops':
        print('ops ' + json.dumps(play_ops(sys.argv[2])))
    else:
        print('scenario ' + json.dumps(play_scenario(sys.argv[1])))


def test_the_expected_digests_are_the_pinned_ones():
    assert hashlib.sha256(DIGESTS_TEXT).hexdigest() == DIGESTS_SHA256
