# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the tests of cuju.scene, with the reference joint math."""

import numpy as np
import torch

import cuju
import cuju.scene as scene
from cuju import _state

NODE_BYTES = 44
POSITION_STEP = 1 / 500
ROTATION_STEP = 1 / 32767


def make_batch(device='cpu', name='5_vs_5', environments=3, **options):
    options.setdefault('raw', True)
    return cuju.BatchEnvironment(
        name, environment_count=environments, device=device, seed=7, **options
    )


def random_actions(generator, batch):
    return torch.randint(0, 19, (batch.environment_count, batch.agents), generator=generator).int()


def play(batch, steps, seed=1):
    """Steps a batch with seeded random actions and returns what its raw outputs held each step."""
    generator = torch.Generator().manual_seed(seed)
    outputs = []
    for _ in range(steps):
        batch.step(random_actions(generator, batch))
        outputs.append(tuple(tensor.cpu().numpy().tobytes() for tensor in batch.raw().values()))
    return outputs


def _multiply(first, second):
    x1, y1, z1, w1 = np.moveaxis(first, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(second, -1, 0)
    return np.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        -1,
    )


def _rotate(quaternion, vector):
    part = quaternion[..., :3]
    first = np.cross(part, vector)
    second = np.cross(part, first)
    return vector + first * (2.0 * quaternion[..., 3:]) + second * 2.0


def world_transforms(blocks):
    """Returns `[humanoids, nodes, 7]`, a position and a rotation for every node, in float64.

    The math is independent of the engine's: each node of a humanoid comes after its parent, so one
    pass over the nodes gives every world transform from the local ones that a state holds.

    Args:
        blocks: `[humanoids, nodes * 44]` bytes of the nodes as a state holds them, each a position,
            a rotation, a scale and a local mode.
    """
    count = blocks.shape[0]
    nodes = len(scene.NODES)
    records = np.ascontiguousarray(blocks, np.uint8).reshape(count, nodes, NODE_BYTES)
    floats = records[..., :40].copy().view('<f4').astype(np.float64)
    modes = records[..., 40:44].copy().view('<i4')[..., 0]
    positions, rotations, scales = floats[..., 0:3], floats[..., 3:7], floats[..., 7:10]
    world = np.empty((count, nodes, 7))
    world_scale = np.empty((count, nodes, 3))
    for node, parent in enumerate(scene.PARENTS):
        world[:, node, 3:] = rotations[:, node]
        world_scale[:, node] = scales[:, node]
        world[:, node, :3] = positions[:, node]
        if parent < 0:
            continue
        relative = modes[:, node] == 0
        rotation = _multiply(world[:, parent, 3:], rotations[:, node])
        rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
        position = (
            _rotate(world[:, parent, 3:], world_scale[:, parent] * positions[:, node])
            + world[:, parent, :3]
        )
        world[relative, node, 3:] = rotation[relative]
        world_scale[relative, node] = (world_scale[:, parent] * scales[:, node])[relative]
        world[relative, node, :3] = position[relative]
    return world


def reference(state, room, frame='pitch'):
    """Returns the nodes of every humanoid of a state in the order of the entries of a scene.

    The result is `[entries, nodes, 7]` in float64, NaN where no humanoid fills the room, from the
    nodes of the state alone.

    Args:
        state: The bytes of a state, as `get_state` gives them.
        room: The room of a team among the entries, `Frames.team_players`.
        frame: `pitch` for the frame that the library loads the match into, or `rest` for the
            frame of the saved state, where the right team lies mirrored.
    """
    layout = _state.layout()
    reverse, first, left, right = (
        int(value.reshape(-1)[0])
        for value in _state.read(
            state,
            [
                'environment.scenario.reverseTeamProcessing',
                'match.firstTeam',
                'environment.scenario.leftTeamSize',
                'environment.scenario.rightTeamSize',
            ],
        )
    )
    first_team = first ^ reverse
    sizes = (left, right)
    order = [(first_team, index) for index in range(sizes[first_team])] + [
        (1 - first_team, index) for index in range(sizes[1 - first_team])
    ]

    def nodes_of(bases, frame):
        fields = []
        for base in bases:
            first_node = layout[f'{base}.nodes[0].position']
            fields.append(_state.Field(f'{base}.nodes', first_node.offset, 44 * 14, 'nodes'))
        blocks = np.frombuffer(_state.gather(state, fields, frame), np.uint8)
        return world_transforms(blocks.reshape(len(fields), -1))

    world = nodes_of(
        [f'match.players[{player}].base.humanoid' for player in range(left + right)], frame
    )
    out = np.full((2 * room + 3, 14, 7), np.nan)
    for player, (team, index) in enumerate(order):
        out[team * room + index] = world[player]
    # The library's pitch frame turns the officials around, which GRF's renderer does only when the
    # teams are reversed, so the officials come from the saved state, and only when it holds them
    # as the scene does.
    if not reverse:
        out[2 * room :] = nodes_of(
            [f'match.officials[{official}].humanoid' for official in range(3)], 'rest'
        )
    return out


def entry_nodes(frames):
    """Returns the root and the joints of frames as `[..., entries, nodes, 7]` in float64."""
    root = np.concatenate([frames.root_position, frames.root_rotation], -1)[..., None, :]
    joints = np.concatenate([frames.joints, frames.joint_rotations], -1)
    return np.concatenate([root, joints], -2).astype(np.float64)


def quantize(nodes):
    """Returns what a recording stores of nodes, in steps of 2 mm and of 1/32767."""
    scale = np.array([500] * 3 + [32767] * 4, np.float64)
    return np.clip(np.rint(nodes * scale), -32767, 32767)


def rotation_gap(first, second):
    """Returns how far two unit quaternions lie from the same rotation, as 1 less their dot."""
    return np.abs(1 - np.abs((first * second).sum(-1)))
