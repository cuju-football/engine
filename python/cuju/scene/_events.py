# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NamedTuple

from ._frames import Frames


class Event(NamedTuple):
    """A change of the match between two frames, at the step of the later one."""

    step: int
    kind: str
    data: dict[str, Any]


KINDS = ('kickoff', 'restart', 'goal', 'foul', 'card', 'offside', 'halftime', 'fulltime')
_KICK_OFF = 1


def _entry(index: int) -> int | None:
    return None if index < 0 else int(index)


def between(before: Frames, after: Frames) -> list[list[Event]]:
    """Returns the events of each environment from one frame to the next.

    Both arguments hold one frame of each environment, and an environment whose episode changed
    between them has none, since the frames belong to different matches.

    Args:
        before: The frames of the environments at one step.
        after: The frames of the same environments at a later step.
    """
    first, second = before.data, after.data
    result: list[list[Event]] = []
    for index in range(first.shape[0]):
        was, now = first[index], second[index]
        events: list[Event] = []
        result.append(events)
        if was['episode'] != now['episode']:
            continue
        step = int(now['step'])
        if now['game_mode'] == _KICK_OFF and was['game_mode'] != _KICK_OFF:
            events.append(Event(step, 'kickoff', {}))
        if now['set_piece'] != 0 and (
            was['set_piece'] == 0 or was['set_piece'] != now['set_piece']
        ):
            events.append(Event(step, 'restart', {'set_piece': int(now['set_piece'])}))
        for team in range(2):
            for _ in range(int(now['score'][team]) - int(was['score'][team])):
                events.append(
                    Event(
                        step,
                        'goal',
                        {'team': team, 'scorer': _entry(now['goal'][1])},
                    )
                )
        if now['foul'][0] != 0 and (
            was['foul'][0] == 0 or tuple(was['foul']) != tuple(now['foul'])
        ):
            kind, offender, victim = (int(value) for value in now['foul'])
            events.append(
                Event(
                    step,
                    'foul',
                    {'kind': kind, 'offender': _entry(offender), 'victim': _entry(victim)},
                )
            )
        for player in range(now['players'].shape[0]):
            yellow, red = now['players'][player]['cards']
            old_yellow, old_red = was['players'][player]['cards']
            for kind, gained in ((1, yellow > old_yellow), (2, red > old_red)):
                if gained:
                    events.append(Event(step, 'card', {'player': player, 'kind': kind}))
        if now['offside_called'] and not was['offside_called']:
            events.append(Event(step, 'offside', {'player': _entry(now['offside_player'])}))
        if now['half'] == 2 and was['half'] != 2:
            events.append(Event(step, 'halftime', {}))
        if now['step'] >= now['duration'] and was['step'] < was['duration']:
            events.append(Event(step, 'fulltime', {}))
    return result
