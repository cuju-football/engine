# Examples

Our **examples** can be found here as a notebook and three single-file Python programs. Each program runs
from the root of the repository, takes its settings from command-line flags, and draws every random
number from a seeded generator, so a run repeats exactly. `--help` lists the flags.

| Example | Shows | Command |
| --- | --- | --- |
| [cuju_quickstart.ipynb](cuju_quickstart.ipynb) | the first steps of a batch on Colab's free GPU | open it in Colab |
| [ppo.py](ppo.py) | a complete PPO trainer in the style of CleanRL, one network shared by every agent, with generalized advantage estimation, the clipped objective and value loss, and weights saved as a state dictionary | `python examples/ppo.py --device cuda --save ppo.pt` |
| [evaluate.py](evaluate.py) | fixed matches of saved weights, or of random actions, against the built-in AI, with wins, draws, losses, goals and a 95% interval on the win rate | `python examples/evaluate.py --weights ppo.pt --matches 1000` |
| [compare_with_grf.py](compare_with_grf.py) | the same matches in cuju and in an installed GRF, compared bit for bit at every step, which needs `gfootball` and plays on the CPU | `python examples/compare_with_grf.py --scenario 5_vs_5 --left 4` |

`ppo.py` and `evaluate.py` need PyTorch and cuju, and both play on the CPU unless given
`--device cuda`. `evaluate.py` loads the weights that `ppo.py` saves.

See [Getting started](../docs/getting-started.md) for the patterns behind these programs and the
[API reference](../docs/api.md) for every name that they use.
