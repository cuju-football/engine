# Copyright 2026 The cuju Authors
# Copyright 2019 Google LLC & Bastiaan Konings
# SPDX-License-Identifier: Apache-2.0

from typing import NamedTuple


class Rectangle(NamedTuple):
    """Half of the length and half of the width of a rectangle, in meters."""

    half_length: float
    half_width: float


class Goal(NamedTuple):
    """A goal in meters, half of its width between the posts, its height and its depth."""

    half_width: float
    height: float
    depth: float


# GRF's pitch inside its lines, and the ground that the stadium lays around it.
PITCH = Rectangle(55.0, 36.0)
FULL_PITCH = Rectangle(60.0, 40.0)
GOAL = Goal(3.7, 2.5, 2.55)
LINE_HALF_WIDTH = 0.06

# GRF's skeleton of a player, in pre-order, so every parent comes before its children. The root is
# node 0, and the joints of a frame are the nodes from 1.
NODES = (
    'player',
    'body',
    'middle',
    'neck',
    'left_shoulder',
    'left_elbow',
    'right_shoulder',
    'right_elbow',
    'left_thigh',
    'left_knee',
    'left_ankle',
    'right_thigh',
    'right_knee',
    'right_ankle',
)
PARENTS = (-1, 0, 1, 2, 2, 4, 2, 6, 1, 8, 9, 1, 11, 12)
JOINTS = NODES[1:]

# The names of GRF's enumerations, in the order of their values.
TEAMS = ('left', 'right', 'official')
HALVES = ('pre_match', 'first_half', 'second_half')
ROLES = (
    'goalkeeper',
    'centre_back',
    'left_back',
    'right_back',
    'defensive_midfielder',
    'central_midfielder',
    'left_midfielder',
    'right_midfielder',
    'attacking_midfielder',
    'centre_forward',
)
GAME_MODES = ('normal', 'kick_off', 'goal_kick', 'free_kick', 'corner', 'throw_in', 'penalty')
TOUCHES = ('intentional_kicked', 'intentional_non_kicked', 'accidental', 'none')
FOULS = ('none', 'foul', 'yellow_card', 'red_card')
ANIMATIONS = (
    'none',
    'movement',
    'ball_control',
    'trap',
    'short_pass',
    'long_pass',
    'high_pass',
    'header',
    'shot',
    'deflect',
    'catch',
    'interfere',
    'trip',
    'sliding',
    'special',
)
CARDS = ('none', 'yellow', 'red')
