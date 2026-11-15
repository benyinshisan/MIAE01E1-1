# Evaluator — Issue #7

`Evaluator` now implements the existing evaluation entry point. Each call owns
a new `EnvFactory.make_eval` environment and closes it even on failure. The
factory's evaluation chain omits both life-loss termination and reward clipping;
training still uses reward clipping. This fixes the previous evaluation chain,
which clipped the score before the evaluator could receive it.

The evaluation RNG controls epsilon-greedy selection independently of the
training agent. Only `act_greedy` is called, under `torch.inference_mode()`.
No training epsilon schedule, replay-buffer writes, updates, or target syncs
are performed. Successive episodes continue the environment's seeded no-op RNG;
they are not reset to the same seed on every episode. Recreating the evaluator
with the same seed reproduces the sequence of evaluation runs.

## Runner integration

```python
evaluator = Evaluator(
    agent, env_factory, cfg.eval, seed=cfg.run.seed,
    random_score=-20.7, human_score=9.3,  # Pong constants belong at the call site
    action_repeat=cfg.env.action_repeat,
)
result = evaluator.evaluate(at_frame=training_frames)
```

`scores` contains raw game returns. `std` is the population standard deviation
(`ddof=0`) across episodes. `at_frame` labels the training checkpoint; evaluation
frames do not advance it. Human normalization uses the injected baselines.
If no baselines were supplied, accessing `normalized` raises `ValueError`.

The constructor retains the previous positional arguments and adds optional
keyword arguments. `EvalResult` retains the original four named fields and adds
two trailing optional baseline fields. Four-argument construction and named
attribute access still work; consumers must use named attributes rather than
unpack the result into four variables. There are no tuple-unpacking consumers
in main at `11a1e4a`.

## Time limit and deliberate choices

Default evaluation uses 30 games, epsilon 0.05, and a maximum 18,000 ALE frames.
The ALE episode counter includes reset/no-op/FIRE frames. A complete repeated
action is only taken if it fits in the remaining budget; at most three frames
are left unused at the default repeat of four. This conservative cutoff avoids
changing action repeat for the last action. Reset overhead that already exhausts
the budget is rejected. Non-ALE `EnvHandle` implementations without a frame
counter use decision count times the supplied `action_repeat`.

`action_repeat` must match the environment. The measured ALE advance is checked,
so the common double-frame-skip configuration is rejected instead of silently
producing scores under the wrong protocol. A true termination or time-limit
truncation ends evaluation; life loss alone does not.

The random-policy acceptance test uses epsilon 1.0 with the project's four-frame
action repeat. This is the Issue #7 smoke criterion, not a recreation of the
paper's random baseline protocol, which selected random actions every six
frames. It does not demonstrate trained DQN performance.

## Verification

Run `python -m pytest tests/test_evaluator.py tests/test_evaluator_atari.py -q -s`.
The first file covers isolation, deterministic RNG, raw scores, termination and
truncation, frame budgets, error cleanup, statistics and baseline injection.
The second runs all 30 real Pong games and checks mean score in [-23.7, -17.7].
ALE and its bundled ROM must be installed; missing dependencies fail the test
rather than silently skipping acceptance. Install the repository runtime
requirements plus pytest, with a suitable PyTorch build as described in README.

Verified on Windows on 2026-11-15: Python 3.12.16, PyTorch 2.4.1+cu121,
Gymnasium 1.0.0, ALE 0.10.1, NumPy 2.5.4, OpenCV 5.0.0 and pytest 9.2.0.
Evaluator plus existing import tests: **22 passed**. The 30-game random-policy
mean was **-20.4**, population standard deviation **0.663325**, within the Issue's
acceptance interval. Seed: 2260031814. No training run was needed for this check.

The generic trained Agent remains another member's Issue #5. A random-policy
adapter permits evaluation acceptance before that implementation is merged.

## Paper and reference audit

Sources: [Mnih et al. (2015), Methods](https://doi.org/10.1038/nature14236)
([paper text mirror](https://device.report/m/ff0bef8f5b5cbeadaec0224ee95e854af59d78dce75de3234f802fe10cb620bc))
and [SB3 2.4.1 DQN source](https://stable-baselines3.readthedocs.io/en/v2.4.1/_modules/stable_baselines3/dqn/dqn.html).

- `network.py` matches the paper's 32/64/64 convolutions, kernels 8/4/3,
  strides 4/2/1, ReLU hidden layers, 512-unit dense layer and linear action head.
  Its `/255` conversion also satisfies the project's uint8 input contract.
  Structural similarity to NatureCNN is expected and does not establish
  whether source was copied. Existing source references should be retained.
- SB3 2.4.1 defaults include learning rate 1e-4, max gradient norm 10,
  and target sync counted in environment steps. Those are not the project's
  paper-spec settings. SB3's TD Huber loss and gradient-norm clipping are
  separate operations; they must not be conflated in the Agent implementation.
- The reward-clipping fix follows the paper's training-only clipping rule.
  Evaluation defaults follow its 30-game / epsilon 0.05 / five-minute procedure.
- New evaluator and tests were implemented against this repository's interfaces;
  no SB3 or CleanRL source was copied into these files. Existing third-party
  notices remain in place. A full audit of unimplemented Agent/Trainer/Runner
  behavior is separate from this PR.

These checks establish protocol and module behavior, not the paper's claim of
above-human trained performance. That requires the later training and evaluation
results with the project's documented scale deviations.
