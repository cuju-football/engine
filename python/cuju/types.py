# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0

"""The sets of strings that options of cuju take, as `Literal` aliases for type checkers."""

from typing import TYPE_CHECKING, Literal, Union

if TYPE_CHECKING:
    import torch

__all__ = [
    'ActionSet',
    'Assignment',
    'ControllerRepresentation',
    'Device',
    'Goalkeeper',
    'Reason',
    'RecordingEnd',
    'Representation',
    'Rewards',
    'Side',
    'Sides',
]

# What a batch gives, `raw` and the vectors and maps of GRF, and the frames of the renderer.
Representation = Literal['simple115', 'simple115v2', 'extracted', 'raw', 'pixels', 'pixels_gray']
# What a controller observes, which a match converts to from the raw observation.
ControllerRepresentation = Literal['raw', 'simple115', 'simple115v2', 'extracted']
# The terms of the reward, which always hold `scoring`.
Rewards = Literal['scoring', 'scoring,checkpoints']
ActionSet = Literal['default', 'v2', 'full']
# A name, or any string that torch takes such as `cuda:1`, or a torch device.
Device = Union[Literal['auto', 'cpu', 'cuda'], str, 'torch.device']
Side = Literal['left', 'right']
Sides = Literal['left', 'right', 'both']
Assignment = Literal['dynamic', 'fixed']
Goalkeeper = Literal['agent', 'builtin', 'scenario']
# Why a controller broke the contract.
Reason = Literal[
    'error', 'timeout', 'closed', 'protocol', 'shape', 'type', 'range', 'load', 'start', 'crash'
]
# How a recorded episode ended, running, terminated or by the time limit.
RecordingEnd = Literal[0, 1, 2]
