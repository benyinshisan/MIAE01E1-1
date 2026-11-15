# Full implementation and verification

Author: 陈博文 (2260031814).

This integration branch starts from main `f0ce7a1`, preserving the merged
environment, buffer and Trainer work and the Issue #10 tests. It includes the
Evaluator change from PR #20 and completes Agent and Runner.

## Running

Use Python 3.12, install the PyTorch build appropriate to the machine, then
`python -m pip install -r requirements-test.txt` and `python -m pip install -e .`.
The conda environment file includes the same logging/plotting dependencies.

```powershell
python -m miae01_dqn.runner --smoke --seed 1 --device cuda
```

```powershell
python -m miae01_dqn.runner --seed 1 --device cuda --replay-capacity 200000 --total-agent-steps 2500000
```

Use `--device cpu` on a CPU machine. Optional `--config config.yaml` accepts
nested `env/algo/optim/eval/run` mappings or flat names. Explicit CLI values take
precedence. Unknown fields fail. `--log-dir` selects the output parent; no host
paths are embedded in source. `--run-name` is optional; automatic names contain
environment, seed, capacity and UTC timestamp. Existing output directories are
rejected so completed evidence cannot be overwritten accidentally.

Smoke is an explicit test profile: 5,000 steps, 250 warm-up steps, 5,000 replay
slots unless explicitly overridden, and 2 evaluation games unless overridden.
It exercises updates while keeping test memory small. It does **not** modify
the default experiment (2.5M steps, 12,500 warm-up steps, 200k slots, 30 games).

Each successful run writes TensorBoard events, `metrics.jsonl`, `config.json`,
`environment.json`, `evaluations.json`, `progress.json`, `result.json` and
`curves.png`. Partial failures write `failure.json`; result.json is only written
after training, final evaluation and plotting complete. Progress JSON includes
SPS, wall clock and process resident memory. The optional local checkpoint stores
model, optimizer and exploration RNG state. A resumable experiment additionally
requires replay-buffer and environment snapshots.

## Agent numerical specification

The TD target is `reward + gamma * (1-done) * max(target(next_obs))`.
Only the online action value receives gradients. The clipped TD derivative is
implemented with Huber loss; the returned loss is the batch mean, while backward
sums per-sample contributions to match the original Torch7 update convention.
Gradient norm clipping is disabled by default and only enabled by an explicit
configuration override. Target copying is controlled by Trainer's update count.

The optimizer tracks mean `m` and second moment `v` of the gradient:

```
m <- momentum * m + (1-momentum) * g
v <- alpha * v + (1-alpha) * g*g
parameter <- parameter - lr * g / sqrt(max(v-m*m, 0) + eps)
```

The existing `momentum=0.95` field denotes gradient-mean decay, not velocity
momentum. Epsilon is inside the square root. This distinction matters: blindly
using `torch.optim.RMSprop(momentum=.95)` changes the algorithm. Numerical tests
check multiple steps against an independent scalar calculation. Adam is an
explicit experimental option and must be reported as a deviation if used.

Reference: [DeepMind NeuralQLearner](https://github.com/google-deepmind/dqn/blob/master/dqn/NeuralQLearner.lua),
`qLearnMinibatch` and `getQUpdate`. The equations have an independent PyTorch
implementation in this repository. The project's frame/step/update conventions remain those frozen
in `architecture.md`. Paper parameters, project contracts and library defaults
are distinguished in the numerical implementation and its tests.

The separate [paper/code audit](paper-code-audit.md) records the released
implementation's different warm-up, exploration and target clocks, and the
project's non-uniform warm-up. The current run preserves the project contract;
the audit specifies the schedule used for each comparison.

## Integration fixes and preserved behavior

- Evaluator receives raw rewards; training retains sign clipping and raw-reward
  info. Its RNG does not consume the training exploration stream.
- The already-merged Trainer/ReplayBuffer truncation repair is retained: the
  final nonterminal observation survives the following reset push.
- Config parsing now understands Python's `float | None` form, including CLI
  strings for optional gradient clipping. Fractional integer overrides fail
  instead of silently truncating.
- `noop_max=0`, which the config permits, disables the no-op wrapper.
- CPU thread count is one; CUDA uses deterministic cuDNN algorithms with
  benchmarking disabled. Exact repeatability was measured on the same machine
  with the same package versions.

## Peer review

Tests cover config/network, Agent gradients/RNG/target copy, Trainer counters,
truncation, raw returns, evaluation isolation, and actual Runner artifacts.
Issues #9/#11/#12/#13 assign cross-verification to a non-implementer. This branch
supplies regression coverage and execution records for the team's independent
review and sign-off.

### Concurrent Agent PR

During the full experiment, PR #20 was merged as `e423765` and the team opened
[Agent PR #22](https://github.com/benyinshisan/MIAE01E1-1/pull/22), inspected at
head `73a54c2`. Its `torch.optim.RMSprop` uses uncentered second moments, velocity
momentum and epsilon outside the square root, with batch-mean gradients.
Those choices differ from the historical equations documented above. This
completion branch submits its independently implemented Agent for joint review
with PR #22. Integration review covers the numerical convention and shared API.

PR #22 also highlighted the allowed `eps=0` boundary: zero gradients must not
produce `0/0`. This branch uses the dtype's smallest positive normal value only
when epsilon is zero, with regression tests for both optimizer choices. The
paper-default `eps=0.01` computation and the running full experiment are unchanged.

The implementation tests and smoke runs establish the executable engineering
path. The full experiment supplies the recorded learning trajectory and final
30-game score for comparison with the paper's Pong baselines.
