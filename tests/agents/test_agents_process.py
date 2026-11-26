# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the process controller and the frames of client.py."""

import io
import os
import time

import numpy as np
import pytest
import torch

from cuju import agents
from cuju.agents import client
from cuju.agents._process import Process, source

KAGGLE = "def agent(obs):\n    print('noise on stdout')\n    return [obs['step'] % 19]\n"


def observation():
    return agents.Observation(
        data={
            'ball': np.zeros((2, 1, 3), dtype=np.float32),
            'active': np.zeros((2, 1), dtype=np.int32),
        },
        first=np.ones(2, dtype=bool),
        active=np.ones((2, 1), dtype=bool),
        left_players=np.full(2, 5),
        right_players=np.full(2, 5),
        previous=np.zeros((2, 1), dtype=np.int32),
        step=np.zeros(2, dtype=np.int64),
        side=np.zeros(2, dtype=np.int8),
    )


def test_frames_round_trip_every_dtype_and_refuse_pickles():
    arrays = {
        name: np.arange(6, dtype=name).reshape(2, 3) for name in client._DTYPES if name != 'bool'
    }
    arrays['bool'] = np.array([True, False])
    frame = client.encode('act', {'a': arrays, 'n': [1, 2.5, 'x', None, True]}, extra=7)
    found = client.read(io.BytesIO(frame))
    assert found['kind'] == 'act' and found['extra'] == 7
    for name, array in arrays.items():
        assert found['value']['a'][name].dtype == array.dtype
        assert (found['value']['a'][name] == array).all()
    with pytest.raises(client.FrameError):
        client.encode('act', object())
    with pytest.raises(client.FrameError, match='protocol'):
        client.read(io.BytesIO(frame.replace(b'"protocol":1', b'"protocol":9')))
    with pytest.raises(EOFError):
        client.read(io.BytesIO(frame[:20]))


def test_client_is_standalone_and_old_syntax():
    import ast

    text = source().decode()
    tree = ast.parse(text, feature_version=(3, 8))
    roots = {
        alias.name.split('.')[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split('.')[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert 'cuju' not in roots


def test_a_kaggle_agent_in_a_process_keeps_stdout_clean(tmp_path):
    (tmp_path / 'main.py').write_text(KAGGLE)
    controller = Process({'entry': 'kaggle', 'path': str(tmp_path)})
    try:
        assert controller.representation == 'raw'
        controller.start(agents.Spec(environments=2, agents=1))
        raw = observation()
        raw.data = {
            name: np.zeros((2, 1, *shape), dtype=dtype)
            for name, (shape, dtype) in {
                'active': ((), np.int32),
                'ball': ((3,), np.float32),
                'ball_direction': ((3,), np.float32),
                'ball_rotation': ((3,), np.float32),
                'ball_owned_player': ((), np.int32),
                'ball_owned_team': ((), np.int32),
                'designated': ((), np.int32),
                'game_mode': ((), np.int32),
                'steps_left': ((), np.int32),
                'score': ((2,), np.int32),
                'sticky_actions': ((10,), np.uint8),
                'left_team': ((11, 2), np.float32),
                'left_team_direction': ((11, 2), np.float32),
                'left_team_tired_factor': ((11,), np.float32),
                'left_team_yellow_card': ((11,), np.bool_),
                'left_team_active': ((11,), np.bool_),
                'left_team_roles': ((11,), np.int32),
                'right_team': ((11, 2), np.float32),
                'right_team_direction': ((11, 2), np.float32),
                'right_team_tired_factor': ((11,), np.float32),
                'right_team_yellow_card': ((11,), np.bool_),
                'right_team_active': ((11,), np.bool_),
                'right_team_roles': ((11,), np.int32),
            }.items()
        }
        answer = controller.act(raw)
        assert answer.actions.tolist() == [[0], [0]]
        raw.first[:] = False
        assert controller.act(raw).actions.tolist() == [[1], [1]]
    finally:
        controller.close()


def test_closing_ends_the_tree_and_a_crash_is_a_controller_error():
    controller = Process({'entry': 'cuju.agents.standard:Random'}, timeout=30)
    controller.start(agents.Spec(environments=2, agents=1))
    pid = controller.pid
    controller.close()
    time.sleep(0.2)
    with pytest.raises(OSError):
        os.kill(pid, 0)
    broken = Process({'entry': 'cuju.agents.standard:NoSuchThing'}, timeout=30)
    with pytest.raises(agents.ControllerError) as caught:
        assert broken.representation
    assert caught.value.reason == 'load'
    broken.close()


def test_a_silent_child_times_out():
    code = (
        'import time\nclass Slow:\n    representation = "raw"\n'
        '    def start(self, spec): pass\n    def act(self, o):\n        time.sleep(30)\n'
        '    def close(self): pass\n'
    )
    controller = Process({'entry': 'slowmod:Slow'}, timeout=1.0, env={'PYTHONPATH': ''}, cwd=None)
    folder = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'cuju_slowmod')
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'slowmod.py'), 'w') as handle:
        handle.write(code)
    controller.env['PYTHONPATH'] = folder
    controller.start(agents.Spec(environments=2, agents=1))
    pid = controller.pid
    with pytest.raises(agents.ControllerError) as caught:
        controller.act(observation())
    assert caught.value.reason == 'timeout'
    controller.close()
    time.sleep(0.2)
    with pytest.raises(OSError):
        os.kill(pid, 0)


def test_a_tensor_observation_crosses_as_numpy():
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    found = client.read(io.BytesIO(client.encode('act', {'x': value})))
    assert (found['value']['x'] == value.numpy()).all()
