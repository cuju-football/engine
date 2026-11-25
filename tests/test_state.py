# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Checks that a state moved to another environment continues the same match."""

from conftest import actions_at, tensor_bytes
from cuju._batch import Batch


def test_state_round_trip(device):
    batch = Batch('5_vs_5_l4_r0', environment_count=2, device=device, rewards='scoring,checkpoints')
    batch.reset()
    for step in range(60):
        batch.step(actions_at(step, 2, 4, 19).to(device))
    batch.set_state(1, batch.get_state(0))
    # A state carries no generator, so the comparison stops where an episode ends.
    ended = False
    for step in range(60, 160):
        actions = actions_at(step, 1, 4, 19).repeat(2, 1).to(device)
        observation, reward, done, information = batch.step(actions)
        if bool(done[0]):
            ended = True
            observation = information['final_observation']
        assert tensor_bytes(observation[0]) == tensor_bytes(observation[1])
        assert tensor_bytes(reward[0]) == tensor_bytes(reward[1])
        if ended:
            break
    if not ended:
        assert batch.get_state(0) == batch.get_state(1)


def test_a_state_of_the_other_parity_plays_as_in_grf(device):
    # GRF: `11_vs_11_stochastic` swaps the teams' formations with the parity of the episode. A load
    # takes the state's order of the teams, while the players keep the formation entries of the
    # reset. The digest is of every field of the clean GRF build's match 115 steps after loading a
    # state of the second episode into the third, past a goal and its kickoff, without the padding
    # that GRF leaves unset in each radian.
    import hashlib

    import numpy as np
    import torch

    import cuju.state

    def batch(episode_number):
        made = Batch(
            '11_vs_11_stochastic',
            environment_count=1,
            device=device,
            agent_counts=[(0, 0)],
            episode_number=episode_number,
            game_engine_random_seed=2,
        )
        made.reset()
        return made

    actions = torch.zeros((1, 0), dtype=torch.int32, device=device)
    first = batch(1)
    for _ in range(100):
        first.step(actions)
    state = first.get_state(0)
    first.close()
    second = batch(2)
    second.set_state(0, state)
    for _ in range(115):
        second.step(actions)
    loaded = second.get_state(0)
    second.close()
    layout = cuju.state.layout()
    names = list(layout)
    digest = hashlib.sha256()
    for name, value in zip(names, cuju.state.read(loaded, names), strict=True):
        data = np.ascontiguousarray(value).tobytes()
        digest.update(data[:5] if layout[name].type == 'N7blunted6radianE' else data)
    assert digest.hexdigest() == '61019403dfa67543dcad95dc61e2ebd7f6cdcc18c50183a4e0cf96e27c2dd15a'
