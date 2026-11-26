# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks the constants of cuju.scene against the engine's own enumerations and values."""

import re
from pathlib import Path

import pytest

import cuju.scene as scene

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'include' / 'cuju'

pytestmark = pytest.mark.skipif(not SOURCE.is_dir(), reason='outside the source tree')


def snake(name):
    return re.sub(r'(?<=[a-z])(?=[A-Z])', '_', name).replace('__', '_').lower()


def enumeration(path, name, prefix):
    text = (SOURCE / path).read_text(encoding='utf-8')
    body = re.search(rf'enum {name}\s*(?::\s*\w+\s*)?{{(.*?)}}', text, re.S).group(1)
    names = [entry.split('=')[0].strip() for entry in body.split(',') if entry.strip()]
    return tuple(snake(entry.removeprefix(prefix)) for entry in names if not entry.endswith('SIZE'))


def number(name):
    text = (SOURCE / 'game_definitions.h').read_text(encoding='utf-8')
    return float(re.search(rf'{name} = ([0-9.]+)f?;', text).group(1))


def test_the_enumerations_equal_the_engines_in_order():
    enumerations = 'simulation/enumerations.h'
    assert enumeration(enumerations, 'e_FunctionType', 'e_FunctionType_') == scene.ANIMATIONS
    assert enumeration(enumerations, 'e_TouchType', 'e_TouchType_') == scene.TOUCHES
    assert enumeration(enumerations, 'e_PlayerRole', 'e_PlayerRole_') == scene.ROLES
    assert enumeration(enumerations, 'e_GameMode', 'e_GameMode_') == scene.GAME_MODES
    assert enumeration('simulation/referee.h', 'e_MatchPhase', 'e_MatchPhase_') == scene.HALVES


def test_the_pitch_and_the_goal_equal_the_engines_values():
    assert scene.PITCH == (number('PITCH_HALF_LENGTH'), number('PITCH_HALF_WIDTH'))
    assert scene.GOAL == (number('GOAL_HALF_WIDTH'), number('GOAL_HEIGHT'), number('GOAL_DEPTH'))
    assert scene.LINE_HALF_WIDTH == number('LINE_HALF_WIDTH')
    assert scene.FULL_PITCH == (60.0, 40.0)


def test_the_skeleton_lists_each_parent_before_its_children():
    assert len(scene.NODES) == len(scene.PARENTS) == 14
    assert scene.JOINTS == scene.NODES[1:]
    assert scene.PARENTS[0] == -1
    assert all(parent < node for node, parent in enumerate(scene.PARENTS) if node)
    assert len(set(scene.NODES)) == 14


def test_the_scene_layout_matches_the_header_of_the_engine():
    text = (SOURCE / 'scene' / 'scene_state.h').read_text(encoding='utf-8')
    assert f'SCENE_STATE_VERSION = {scene.VERSION};' in text
    assert 'SCENE_JOINTS = state::HUMANOID_NODES - 1' in text
    assert len(scene.JOINTS) == 13
