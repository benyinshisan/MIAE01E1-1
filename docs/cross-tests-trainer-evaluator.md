# Issue #12 — independent Trainer and Evaluator verification (X-04)

Author: 2260031685 沈坤龙 (`@benyinshisan`).

Authorship note: Trainer (I-06, #6) was authored by 2260032112 黄俊皓 in PR #19
and Evaluator (I-07, #7) was authored by 2260031814 陈博文 in PR #20. Neither
module was owned by me (I authored I-01 `config` and I-03 `envs`; for full
transparency, I contributed review-fix commit `6b3f9dc` on PR #19 prior to
merge). Among the three team members, I am the sole non-author of both I-06
and I-07.

Base: `main` `bb24d71`. No production code is modified by this branch: the
change consists solely of `tests/test_trainer_evaluator_cross.py` and this
evidence document.

## Acceptance coverage

| Issue #12 requirement | Evidence |
|---|---|
| `trainer`: 5k-step smoke finishes without error and `episodes > 0` | `test_trainer_real_pong_5k_smoke_and_logging`: runs real `Trainer` with real `EnvFactory(EnvSpec())` (`ALE/Pong-v5`), real `QNetwork((4, 84, 84), 6)`, real `Agent`, and real `ReplayBuffer(5000)` for 5,000 agent steps (`20,000` frames); finishes with `episodes > 0`, `updates == 1187`, `len(buffer) == 5000`, finite `final_return_mean`, and all 5 documented logger tags (`episodic_return`, `epsilon`, `td_loss`, `q_values`, `SPS`) |
| `trainer`: fake env truncation replaces `next_obs` with `final_observation` | `test_trainer_truncation_uses_final_observation_and_preserves_in_real_buffer` & `test_trainer_plain_truncation_without_final_observation_resets_env`: verifies same-step autoreset stores `info["final_observation"]` (marker `77`) rather than reset frame (`99`), sets `done=False` and `truncated=True`, avoids redundant `env.reset()`, preserves marker `77` in a real `ReplayBuffer` across subsequent episode pushes, logs unclipped `raw_reward`, and falls back to `next_obs` + `env.reset()` when `final_observation` is absent |
| `trainer`: `updates` matches `step` conversion via `learn_every_steps` | `test_trainer_updates_and_target_sync_counts` (4 schedules across `action_repeat in {2, 4}`, aligned/unaligned `replay_start_steps`, and sub-interval budgets) plus `test_trainer_none_updates_not_counted_and_on_eval_uses_frames` (`None` returns from `agent.update()` do not increment `updates` or advance target syncs; `on_eval` triggers by frame milestones) |
| `trainer`: target sync count equals `updates // target_update_every_updates` | Verified across all 4 schedules (`22 // 5 == 4`, `18 // 3 == 6`, `3 // 10 == 0`, `13 // 4 == 3`) plus exact update indices `[k * target_update_every_updates]` and `None`-return skipping (`7 // 3 == 2` at updates `[3, 6]`) |
| `evaluator`: runs exactly 30 games with consistent return length and statistics | `test_evaluator_exact_30_episodes_and_consistent_statistics` & `test_evaluator_real_pong_30_games_preserves_agent_and_buffer`: default `EvalSpec()` runs exactly 30 episodes (`len(result.scores) == 30`, `reset_count == 30`), matches population `mean`/`std` and injected Pong human normalization (`-20.7 / 9.3`), closes the eval env, and never calls `make_train` |
| `evaluator`: agent parameters and buffer length remain unchanged | `test_evaluator_30_games_preserves_real_cnn_agent_and_buffer` & `test_evaluator_real_pong_30_games_preserves_agent_and_buffer`: real `Agent(QNetwork((4, 84, 84), 6), ReplayBuffer(...))` after real `update()` steps (populated `NatureRMSprop` state) retains bit-identical `policy`, `target`, `optimizer` tensors, `agent.rng` state, `len(buffer)`, `is_full`, zero `buffer.push` calls, and `torch.is_grad_enabled() == True` across 30 episodes |
| Acceptance: both truncation and counting error classes are caught | `test_truncation_assertion_catches_final_obs_and_done_bugs` (3 negative controls: storing autoreset frame instead of `final_observation`, collapsing `done = term or trunc`, omitting `truncated=True`) and `test_counting_assertion_catches_frame_step_and_sync_unit_bugs` (3 negative controls: waiting `replay_start_frames` steps instead of `replay_start_steps`, syncing every N steps instead of N updates, updating every step instead of every `learn_every_steps`) |

## What this suite adds beyond `tests/test_training_integration.py` and `tests/test_evaluator.py`

| Test | Additional coverage |
|---|---|
| `test_trainer_real_pong_5k_smoke_and_logging` | Full 5,000-step real `ALE/Pong-v5` training loop with real `QNetwork`, `Agent`, and `ReplayBuffer` directly through `Trainer` |
| `test_trainer_truncation_uses_final_observation_and_preserves_in_real_buffer` | End-to-end truncation check combining `Trainer` with a real `ReplayBuffer` to confirm `final_observation` survives next-episode ring writes |
| `test_trainer_plain_truncation_without_final_observation_resets_env` | Plain single-env fallback (no `final_observation` key) and post-truncation `env.reset()` count |
| `test_truncation_assertion_catches_final_obs_and_done_bugs` | Negative controls proving all three truncation error modes fail with `AssertionError` |
| `test_trainer_updates_and_target_sync_counts` | Multi-schedule matrix covering unaligned `replay_start_steps`, exact multiples, zero-sync budgets, and custom `action_repeat=2` |
| `test_trainer_none_updates_not_counted_and_on_eval_uses_frames` | `agent.update() -> None` exclusion from `updates`/syncs, frame-based `on_eval`, and invalid `log_every_steps` guard |
| `test_counting_assertion_catches_frame_step_and_sync_unit_bugs` | Negative controls proving frame/step confusion, step-vs-update sync confusion, and missing `learn_every_steps` gating all fail loudly |
| `test_evaluator_30_games_preserves_real_cnn_agent_and_buffer` | Real `QNetwork((4, 84, 84), 6)` + populated `NatureRMSprop` + push-guarded `ReplayBuffer` across 30 episodes |
| `test_evaluator_real_pong_30_games_preserves_agent_and_buffer` | Full 30-game real `ALE/Pong-v5` evaluation with a real `QNetwork` `Agent` verifying score bounds, Pong normalization, and state/buffer invariance |

## Observations and findings

No defects were found in `miae01_dqn/trainer.py` or `miae01_dqn/evaluator.py`.

## Reproduction and result

```bash
python -m pytest -v tests/test_trainer_evaluator_cross.py
python -m pytest -q -rs
```

Environment:

| Item | Value |
|---|---|
| Python | 3.12.16 |
| torch | 2.4.1+cpu (`cuda.is_available() == False`) |
| numpy / gymnasium / ale-py | 2.5.5 / 1.0.0 / 0.10.1 |
| opencv / matplotlib / pytest | 5.0.0 / 3.11.2 / 9.2.0 |
| OS / CPU / RAM | Linux 7.0.0 x86_64, AMD Ryzen 7 9700X (16 threads), 29 GiB |

Measured on 2026-11-23:
- `pytest -v tests/test_trainer_evaluator_cross.py`: **13 passed in 36.68 s** (includes 5k real Pong `Trainer` smoke and 30-game real Pong `Evaluator` run)
- Full suite `pytest -q -rs`: **82 passed in 53.41 s** (69 pre-existing + 13 new, 0 skipped, 0 xfailed)
