# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The engine functions of the CPU library and their C structures, loaded without PyTorch."""

from __future__ import annotations

import ctypes
import os
import threading

from .._library_path import _check_processor, library_path
from ..errors import LibraryError, MissingFileError

# `CUJU_ENGINE_VERSION` of the C API.
VERSION = 2
# The players that GRF's team holds at most.
TEAM_PLAYERS = 11


class Entry(ctypes.Structure):
    """`cuju_engine_entry`, a formation entry as `ScenarioConfig` stores it."""

    _fields_ = [
        ('position', ctypes.c_float * 3),
        ('start_position', ctypes.c_float * 3),
        ('role', ctypes.c_int),
        ('lazy', ctypes.c_int),
        ('controllable', ctypes.c_int),
    ]


class Scenario(ctypes.Structure):
    """`cuju_engine_scenario`, every field of GRF's `ScenarioConfig` and its cache."""

    _fields_ = [
        ('version', ctypes.c_int),
        ('ball_position', ctypes.c_float * 3),
        ('left_team', ctypes.POINTER(Entry)),
        ('left_size', ctypes.c_int),
        ('right_team', ctypes.POINTER(Entry)),
        ('right_size', ctypes.c_int),
        ('left_agents', ctypes.c_int),
        ('right_agents', ctypes.c_int),
        ('use_magnet', ctypes.c_int),
        ('offsides', ctypes.c_int),
        ('real_time', ctypes.c_int),
        ('game_engine_random_seed', ctypes.c_uint),
        ('reverse_team_processing', ctypes.c_int),
        ('left_team_difficulty', ctypes.c_float),
        ('right_team_difficulty', ctypes.c_float),
        ('deterministic', ctypes.c_int),
        ('end_episode_on_score', ctypes.c_int),
        ('end_episode_on_possession_change', ctypes.c_int),
        ('end_episode_on_out_of_play', ctypes.c_int),
        ('game_duration', ctypes.c_int),
        ('second_half', ctypes.c_int),
        ('control_all_players', ctypes.c_int),
        ('cache_computed', ctypes.c_int),
        ('cache_dynamic_player_selection', ctypes.c_int),
        ('cache_controllable_left', ctypes.c_int),
        ('cache_controllable_right', ctypes.c_int),
    ]


class Player(ctypes.Structure):
    """`cuju_engine_player`, one `PlayerInfo` with the raw vectors of GRF's `Position`."""

    _fields_ = [
        ('position', ctypes.c_float * 3),
        ('direction', ctypes.c_float * 3),
        ('tired_factor', ctypes.c_float),
        ('has_card', ctypes.c_int),
        ('is_active', ctypes.c_int),
        ('designated_player', ctypes.c_int),
        ('role', ctypes.c_int),
    ]


class Info(ctypes.Structure):
    """`cuju_engine_info`, GRF's `SharedInfo`."""

    _fields_ = [
        ('version', ctypes.c_int),
        ('ball_position', ctypes.c_float * 3),
        ('ball_direction', ctypes.c_float * 3),
        ('ball_rotation', ctypes.c_float * 3),
        ('left_size', ctypes.c_int),
        ('right_size', ctypes.c_int),
        ('left_team', Player * TEAM_PLAYERS),
        ('right_team', Player * TEAM_PLAYERS),
        ('left_controllers', ctypes.c_int * TEAM_PLAYERS),
        ('right_controllers', ctypes.c_int * TEAM_PLAYERS),
        ('left_goals', ctypes.c_int),
        ('right_goals', ctypes.c_int),
        ('game_mode', ctypes.c_int),
        ('is_in_play', ctypes.c_int),
        ('ball_owned_team', ctypes.c_int),
        ('ball_owned_player', ctypes.c_int),
        ('step', ctypes.c_int),
    ]


_LOCK = threading.Lock()
_LIBRARY = None


def library():
    """Returns the CPU library with the engine's signatures declared, loaded once."""
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is None:
            _LIBRARY = _load()
        return _LIBRARY


def _load():
    path = library_path('cpu')
    if not os.path.isfile(path):
        raise MissingFileError(
            f'expected the CPU library of cuju at {path}, which runs the engine of cuju.gfootball'
        )
    _check_processor(path)
    handle = ctypes.CDLL(path)
    pointer, integer = ctypes.c_void_p, ctypes.c_int
    size = ctypes.c_ulonglong
    signatures = {
        'cuju_last_error': ([], ctypes.c_char_p),
        'cuju_engine_version': ([], integer),
        'cuju_engine_create': ([ctypes.c_char_p], pointer),
        'cuju_engine_destroy': ([pointer], None),
        'cuju_engine_start': ([pointer], integer),
        'cuju_engine_reset': ([pointer, ctypes.POINTER(Scenario), integer, integer], integer),
        'cuju_engine_set_config': ([pointer, ctypes.POINTER(Scenario)], integer),
        'cuju_engine_config': ([pointer, ctypes.POINTER(Scenario)], integer),
        'cuju_engine_config_cache': ([pointer, ctypes.POINTER(ctypes.c_int)], integer),
        'cuju_engine_action': ([pointer, integer, integer, integer], integer),
        'cuju_engine_sticky': ([pointer, integer, integer, integer], integer),
        'cuju_engine_step': ([pointer, integer], integer),
        'cuju_engine_ticks': ([pointer, integer, integer], integer),
        'cuju_engine_finish_step': ([pointer], integer),
        'cuju_engine_info': ([pointer, integer, ctypes.POINTER(Info)], integer),
        'cuju_engine_game_state': ([pointer], integer),
        'cuju_engine_set_game_state': ([pointer, integer], integer),
        'cuju_engine_waiting_count': ([pointer], integer),
        'cuju_engine_set_waiting_count': ([pointer, integer], integer),
        'cuju_engine_get_state': (
            [
                pointer,
                ctypes.c_char_p,
                size,
                ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
                ctypes.POINTER(size),
            ],
            integer,
        ),
        'cuju_engine_set_state': (
            [pointer, ctypes.c_char_p, size, ctypes.POINTER(size), ctypes.POINTER(size)],
            integer,
        ),
        'cuju_engine_verify': ([pointer, pointer, ctypes.POINTER(ctypes.c_longlong)], integer),
        'cuju_engine_set_render': ([pointer, integer], integer),
        'cuju_engine_prepare_render': ([pointer], integer),
        'cuju_engine_render_state': (
            [pointer, pointer, size, ctypes.POINTER(ctypes.c_uint)],
            integer,
        ),
        'cuju_engine_render_start': ([pointer, pointer, size], integer),
        'cuju_engine_internal_render_state': ([pointer, size, integer], integer),
        'cuju_engine_render_state_version': ([], ctypes.c_uint),
        'cuju_engine_render_state_size': ([], size),
    }
    try:
        for name, (arguments, result) in signatures.items():
            function = getattr(handle, name)
            function.argtypes = arguments
            function.restype = result
    except AttributeError as error:
        raise LibraryError(
            f'the CPU library at {path} lacks the engine of cuju.gfootball, so it is older than '
            'this package; reinstall cuju'
        ) from error
    found = handle.cuju_engine_version()
    if found != VERSION:
        raise LibraryError(
            f'expected the engine version {VERSION} in the CPU library at {path}, found {found}'
        )
    return handle


def last_error():
    """Returns the message of the library's last failure on this thread."""
    return library().cuju_last_error().decode('utf-8', 'replace')
