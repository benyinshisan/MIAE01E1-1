# Paper, released code and project counting conventions

Audit date: 2026-11-15. This note distinguishes three sources; they are not
interchangeable specifications. It does not change the frozen project schedule
or the configuration of the ongoing seed-1 experiment.

## Paper wording

The [official paper](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf),
Methods and Extended Data Table 1, describes exploration and replay warm-up in
frames, target copying in parameter updates, and a uniform-random warm-up.
The project interprets frames as raw emulator frames and target periods as
gradient-update counts. The paper's stated 50M frames and roughly 38 days should
not be converted to a raw-frame budget without also checking the released code.

## Released implementation versus this experiment

The public [run_gpu driver](https://github.com/google-deepmind/dqn/blob/master/run_gpu)
sets action repeat to 4, warm-up to 50,000, exploration duration to 1,000,000,
target period to 10,000, and the training-loop budget to 50,000,000.
[train_agent.lua](https://github.com/google-deepmind/dqn/blob/master/dqn/train_agent.lua)
increments the loop once per agent interaction and prints a separate frame
count multiplied by action repeat. Thus its counters are not all raw frames.

| Schedule | Released implementation | Frozen project / recorded seed 1 |
|---|---|---|
| Warm-up | 50,000 perceived training states | 12,500 decisions = 50,000 nominal raw frames |
| Exploration decay | 1,000,000 perceived states after warm-up | 250,000 decisions from the beginning |
| Target copy | 10,000 perceived-state intervals | Every 10,000 gradient updates, about 40,000 decisions |
| Training budget | Driver requests 50M interaction-loop iterations | 2.5M decisions = 10M nominal raw frames |

The warm-up offset and target clock are directly visible in
[NeuralQLearner.perceive/eGreedy](https://github.com/google-deepmind/dqn/blob/master/dqn/NeuralQLearner.lua).
Terminal/reset bookkeeping causes additional small differences, so these are
schedule comparisons rather than an assertion of exact ALE frame equivalence.
The project follows the paper table's target-update wording; it does not match
the released implementation's clock. A nominal 10M-versus-50M label must not be
presented as proof that the actual emulator experience differs by only fivefold.

There is also a specific warm-up deviation: the frozen Trainer calls
`epsilon_at(step)` from step zero, so epsilon decreases from 1.0 toward 0.955
during the initial 12,500 decisions. That is not the paper's strictly uniform
random warm-up. This behavior is retained for the requested architecture-based
experiment and explicitly reported, rather than silently changing its data
collection midway through the run.

## What the audit establishes

Network dimensions, clipped TD derivatives and the independent evaluation
protocol have direct checks. Centered RMSProp and the sum-gradient convention
follow the released optimizer equations, as documented in
[full-implementation.md](full-implementation.md). Library defaults alone do not
establish paper compliance.

The current result is a reduced-budget experiment under the project's explicit
contract. It is not a bitwise or schedule-identical reproduction of the released
system. Any future switch to its warm-up/exploration/target clocks needs a
separate configuration, tests and fresh experiment; the existing evidence must
remain labelled with the schedule actually run. One seed cannot establish the
planned three-seed or replay-capacity-ablation conclusions.
