# Issue #11 — independent Agent verification (X-03)

Author: 2260031685 沈坤龙 (`@benyinshisan`).

Authorship note: on `main`, `miae01_dqn/agent.py` (including `NatureRMSprop`
and `Agent`) was implemented from scratch by 2260031814 陈博文 in PR #23
(`0a39511`, `595f8cd`, which closed #5). My earlier parallel PR #22
(`feat/issue-5-agent`) was closed unmerged and shares no implementation code
with `main`. This branch therefore satisfies the non-implementer rule for X-03.

Base: `main` `bb24d71`. No production code is modified by this branch: the
change consists solely of `tests/test_agent_cross.py` and this evidence
document.

## Acceptance coverage

| Issue #11 requirement | Evidence |
|---|---|
| `ε`: step=0 / midpoint / endpoint match frame conversion | `test_epsilon_frame_schedule_default_and_custom_repeat`: checks default `AlgoSpec()` (`action_repeat=4`, `eps_decay_frames=1,000,000`) at steps `0 → 1.0`, `62,500 → 0.775`, `125,000 → 0.55`, `250,000 → 0.1`, plus a custom `action_repeat=2` schedule and `eps_decay_frames=0` / negative-step guards |
| `ε`: stays locked at `eps_end` beyond decay | Verified at steps `250,001`, `300,000`, `1,000,000`, `2,500,000`, and `10,000,000` (all equal `0.1`) |
| Frame-vs-step unit bug is caught by tests | `test_epsilon_schedule_catches_frame_vs_step_unit_errors`: negative controls run the same `assert_frame_based_epsilon_schedule` helper against both a step-as-frame mutation (omitting `* action_repeat`) and an inverted-unit mutation (dividing by `action_repeat`); both fail with `AssertionError` |
| `sync_target()` makes `target` elementwise equal to `policy` | `test_sync_target_copies_all_cnn_parameters_without_aliasing`: runs multiple updates on a real `QNetwork((4, 84, 84), 6)`, confirms divergence before sync, verifies exact (`rtol=0, atol=0`) equality across all conv/FC tensors after `sync_target()`, verifies `eval` mode / `requires_grad=False`, and confirms a subsequent `update()` mutates `policy` without aliasing `target` |
| `update()` changes `policy`; insufficient buffer returns `None` without parameter change | `test_update_returns_none_when_buffer_insufficient_and_updates_when_ready`: on a real `QNetwork((4, 84, 84), 6)` with `batch_size=4`, checks `len(buffer) in {0, 1, 2, 3}` all return `None` with empty optimizer state and bit-identical weights; once `len(buffer) == 4`, `update()` returns a finite positive float and mutates `policy` while leaving `target` unchanged |
| TD error clipping (`|TD| > 1` gradient contract) | `test_td_error_clipping_gradient_and_loss_contract` (`delta in {-10, +10, -500, -0.4}`), `test_td_clipping_distinguishes_clipped_from_unclipped_and_batch_sum`, `test_grad_clip_norm_disabled_by_default_and_active_when_set`, `test_target_bootstrap_for_nonterminal_and_mask_for_terminal`: checks clipped derivative `±1` for `|TD| > 1`, linear derivative for `|TD| <= 1`, saturation invariance (`delta=-500` matches `delta=-10`), zero gradient on unselected actions, batch-summed backward vs batch-mean returned loss, `td_error_clip=0` contrast, optional `grad_clip_norm`, and terminal masking |
| Action selection: `eps=0` always argmax; `eps=1` approximately uniform | `test_action_selection_eps0_greedy_and_eps1_uniform_over_six_actions` & `test_real_qnetwork_greedy_action_and_state_dict_isolation`: 200 draws at `eps=0` always match argmax and `act_greedy` without consuming `agent.rng`; 6,000 draws at `eps=1` across 6 actions pass both a chi-square uniformity check (`df=5`, `chi2 < 25`) and per-action `[850, 1150]` bounds; invalid epsilon raises `ValueError` |

## What this suite adds beyond the author's `tests/test_agent.py`

| Test | Additional coverage |
|---|---|
| `test_epsilon_schedule_catches_frame_vs_step_unit_errors` | Negative controls proving the schedule assertion catches both omitted `* action_repeat` and inverted division |
| `test_update_returns_none_when_buffer_insufficient_and_updates_when_ready` | Real `QNetwork((4, 84, 84), 6)` with `uint8` frames tested across every `0..batch_size` boundary |
| `test_sync_target_copies_all_cnn_parameters_without_aliasing` | Real 5-layer CNN parameter-by-parameter copy plus post-sync non-aliasing check |
| `test_td_error_clipping_gradient_and_loss_contract` | Both signs (`+10`, `-10`), saturation at `500`, interior quadratic regime (`0.4`), and zero gradient on unselected actions |
| `test_td_clipping_distinguishes_clipped_from_unclipped_and_batch_sum` | Direct contrast between `td_error_clip=1.0` and `td_error_clip=0.0` on a multi-sample batch, pinning both summed backward and mean returned loss |
| `test_grad_clip_norm_disabled_by_default_and_active_when_set` | Verifies `grad_clip_norm=None` leaves norm `4.0` intact while `grad_clip_norm=0.5` clips L2 norm to `0.5` |
| `test_action_selection_eps0_greedy_and_eps1_uniform_over_six_actions` | 6-action chi-square uniformity test (matching Atari Pong's 6 discrete actions), RNG preservation on `act_greedy`, and out-of-range epsilon rejection |
| `test_real_qnetwork_greedy_action_and_state_dict_isolation` | Real CNN `q_values`/`act_greedy` alignment and `state_dict()` deep-copy isolation |

## Observations and findings

No defects were found in `miae01_dqn/agent.py`.

1. **Batch reduction convention.** `update()` calls `losses.sum().backward()`
   and returns `float(losses.detach().mean().item())`, matching the original
   Torch7 convention documented in `docs/full-implementation.md`. For a batch
   of size `B` with all `|delta| > 1`, the clipped per-sample derivative
   sums to `±B` before `NatureRMSprop` scaling, while the logged scalar loss
   remains independent of `B`.
2. **Centered RMSProp.** `NatureRMSprop` updates `mean` with `momentum=0.95`,
   `square` with `alpha=0.95`, and places `eps=0.01` inside the square root
   with `clamp_min_(0)` protection, verified across 4 multi-element steps
   against an independent NumPy calculation (`atol=1e-7`).

## Reproduction and result

```bash
python -m pytest -v tests/test_agent_cross.py
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
- `pytest -v tests/test_agent_cross.py`: **14 passed in 0.84 s**
- Full suite `pytest -q -rs`: **83 passed in 19.27 s** (69 pre-existing + 14 new, 0 skipped, 0 xfailed)
