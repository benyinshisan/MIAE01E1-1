# Paper, released code and project counting conventions

Author: 陈博文 (2260031814).

Audit date: 2026-11-15. This note distinguishes the paper, released code and
frozen project schedule. Seed 1 uses the project schedule recorded below.

## Paper wording

The [official paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf),
Methods and Extended Data Table 1, describes exploration and replay warm-up in
frames, target copying in parameter updates, and a uniform-random warm-up.
The project interprets frames as raw emulator frames and target periods as
gradient-update counts. The paper's stated 50M frames and roughly 38 days are
interpreted alongside the released code's decision and emulator-frame counters.

## Released implementation versus this experiment

The public [run_gpu driver](https://github.com/google-deepmind/dqn/blob/master/run_gpu)
sets action repeat to 4, warm-up to 50,000, exploration duration to 1,000,000,
target period to 10,000, and the training-loop budget to 50,000,000.
[train_agent.lua](https://github.com/google-deepmind/dqn/blob/master/dqn/train_agent.lua)
increments the loop once per agent interaction and prints a separate frame
count multiplied by action repeat. The two counters have distinct units.

| Schedule | Released implementation | Frozen project / recorded seed 1 |
|---|---|---|
| Warm-up | 50,000 perceived training states | 12,500 decisions = 50,000 nominal raw frames |
| Exploration decay | 1,000,000 perceived states after warm-up | 250,000 decisions from the beginning |
| Target copy | 10,000 perceived-state intervals | Every 10,000 gradient updates, about 40,000 decisions |
| Training budget | Driver requests 50M interaction-loop iterations | 2.5M decisions = 10M nominal raw frames |

The warm-up offset and target clock are directly visible in
[NeuralQLearner.perceive/eGreedy](https://github.com/google-deepmind/dqn/blob/master/dqn/NeuralQLearner.lua).
The project follows the paper table's target-update wording; the released code
uses perceived-state counts. Terminal/reset bookkeeping also affects exact ALE
counts. Budget comparisons therefore use the explicit units in the table above.

There is also a specific warm-up deviation: the frozen Trainer calls
`epsilon_at(step)` from step zero, so epsilon decreases from 1.0 toward 0.955
during the initial 12,500 decisions. The paper specifies strictly uniform random
warm-up. This difference is recorded as part of the architecture-based
experiment's data-collection schedule.

## What the audit establishes

Network dimensions, clipped TD derivatives and the independent evaluation
protocol have direct checks. Centered RMSProp and the sum-gradient convention
follow the released optimizer equations, as documented in
[full-implementation.md](full-implementation.md). The audit traces each choice
to its numerical specification and executable check.

The current result covers seed 1 under the project's reduced-budget contract.
A released-schedule comparison can be run as a separately configured experiment
with its own tests and records. The planned three-seed statistics and capacity
ablation build on the completed training and evaluation path.
