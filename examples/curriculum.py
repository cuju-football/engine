# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Steps one batch per scenario in turn, as the stages of a curriculum, and reports each stage.

    python examples/curriculum.py [--device cuda] [--environments 64] [--rounds 300]

Every stage is its own `cuju.BatchEnvironment`, a drill, a five-a-side match and an 11-a-side match
here, with the agent counts that suit it. All of them give `simple115v2` observations of 115
numbers per agent and share one action set, so a single policy plays every stage, and the loop
below shows the shape of such a run with random actions. A round steps each batch once, and
a real curriculum would change which batches step as the earlier stages are solved.
"""

import argparse

import torch

import cuju

# The scenario, the agents of the left and of the right team, and a name for the report.
STAGES = [
    ('academy_empty_goal_close', 1, 0, 'drill'),
    ('5_vs_5', 4, 4, 'five a side, self-play'),
    ('11_vs_11_stochastic', 1, 0, 'eleven a side'),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--environments', type=int, default=64, help='per stage')
    parser.add_argument('--rounds', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    arguments = parser.parse_args()

    device = arguments.device
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    batches = [
        cuju.BatchEnvironment(
            scenario,
            environment_count=arguments.environments,
            device=device,
            seed=arguments.seed,
            left_agents=left,
            right_agents=right,
        )
        for scenario, left, right, _ in STAGES
    ]
    try:
        for batch, (scenario, _, _, name) in zip(batches, STAGES, strict=True):
            observation = batch.reset()
            print(f'{scenario} ({name}): observation {tuple(observation.shape)}')
        goals = torch.zeros(len(batches), dtype=torch.int64, device=device)
        episodes = torch.zeros(len(batches), dtype=torch.int64, device=device)
        for _ in range(arguments.rounds):
            for stage, batch in enumerate(batches):
                actions = torch.randint(
                    0,
                    batch.action_count,
                    (batch.environment_count, batch.agents),
                    dtype=torch.int32,
                    device=device,
                    generator=generator,
                )
                _, _, done, information = batch.step(actions)
                goals[stage] += (information['score_reward'] == 1).sum()
                episodes[stage] += done.sum()
        for stage, (scenario, _, _, _) in enumerate(STAGES):
            print(f'{scenario}: {int(goals[stage])} goals, {int(episodes[stage])} episodes ended')
    finally:
        for batch in batches:
            batch.close()


if __name__ == '__main__':
    main()
