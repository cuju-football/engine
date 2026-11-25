# Copyright 2026 The cuju Authors
# SPDX-License-Identifier: Apache-2.0
"""Runs one process per GPU, each stepping its own batch with random actions for a few seconds.

    python examples/one_process_per_gpu.py [--environments 4096] [--seconds 5] [--seed 0]

`torch.multiprocessing.spawn` starts a process for every visible GPU, and process `rank` creates
its `cuju.BatchEnvironment` on `cuda:{rank}` with the seed `base + rank`, so no two GPUs play the
same matches. A batch owns device memory and threads, so each process creates its own, and none
crosses a process boundary. Where PyTorch sees no GPU, one process plays on the CPU.
"""

import argparse
import time

import torch
import torch.multiprocessing

import cuju


def play(rank, arguments):
    """Steps the batch of process `rank` with random actions for `--seconds` and reports it."""
    device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'
    with cuju.BatchEnvironment(
        arguments.scenario,
        environment_count=arguments.environments,
        device=device,
        seed=arguments.seed + rank,
        left_agents=arguments.left_agents,
        right_agents=arguments.right_agents,
    ) as batch:
        batch.reset()
        generator = torch.Generator(device=device).manual_seed(arguments.seed + rank)
        shape = (batch.environment_count, batch.agents)
        steps = 0
        start = time.perf_counter()
        while time.perf_counter() - start < arguments.seconds:
            actions = torch.randint(
                0,
                batch.action_count,
                shape,
                dtype=torch.int32,
                device=device,
                generator=generator,
            )
            batch.step(actions)
            steps += 1
            if steps % 10 == 0 and device != 'cpu':
                # The step does not wait for the device, so the clock needs a pause now and then.
                torch.cuda.synchronize(device)
        if device != 'cpu':
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
    rate = steps * arguments.environments / elapsed
    print(f'process {rank} on {device}, seed {arguments.seed + rank}: {rate:,.0f} steps/s')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scenario', default='5_vs_5')
    parser.add_argument('--left-agents', type=int, default=4)
    parser.add_argument('--right-agents', type=int, default=0)
    parser.add_argument('--environments', type=int, default=4096, help='per process')
    parser.add_argument('--seconds', type=float, default=5.0)
    parser.add_argument('--seed', type=int, default=0, help='the base seed, plus the rank')
    arguments = parser.parse_args()

    processes = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(
        f'{processes} process{"es" * (processes > 1)}, {arguments.environments} environments each'
    )
    torch.multiprocessing.spawn(play, args=(arguments,), nprocs=processes)


if __name__ == '__main__':
    main()
