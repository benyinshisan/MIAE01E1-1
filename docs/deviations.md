# Deviations from the paper

Every difference between our reproduction and Mnih et al. (2015). This table is the source
for the report's **Reproduction Setup** section, criterion
`a complete table of deviations from the paper (item, original setting, setting used and reason)`.

Paper values come from the Methods section and Extended Data Table 1 (verified against the
official PDF; see the W-05 fact sheet held by the team).

Three groups follow: **(A)** scale and hardware, **(B)** assumptions where the paper is silent,
**(C)** implementation provenance. Items that were checked and found *unchanged* are listed at
the end so the table is verifiably complete.

---

## A. Scale and hardware

| # | Item | Paper | Ours | Reason and expected effect |
|---|---|---|---|---|
| A1 | Games | 49 Atari 2600 games, one network per game | **1 game (Pong)** | No NVIDIA GPU available; 49 × 3 seeds is infeasible. The generality claim cannot be tested — it is declared as a limitation, not as a result. |
| A2 | Training length | 50,000,000 frames (≈38 days of game experience) | **10,000,000 frames** (2.5M agent steps × action repeat 4) | At ~273 agent steps/s on CPU this is ≈2.3 h per seed. Expect slower learning / worse final score than the paper. |
| A3 | Replay capacity | 1,000,000 most recent frames | **200,000** | 1M frames × (4, 84, 84) uint8 ≈ 28 GB; the machine has 31.5 GB total and ~10 GB free. Expected effect: less diverse sampling, likely lower final score. |
| A4 | Compute | GPU; 50M frames reported as 38 days of game experience | **CPU only** (no CUDA device present) | Determines A2. Also makes wall-clock comparisons meaningless; we report frames, not hours. |
| A5 | Parallelism | — (one agent per game) | 1 environment, sequential runs | The trainer assumes a single env; parallel envs would change the replay layout. |

## B. Points where the paper is silent (our assumptions)

| # | Item | Paper | Ours | Reason |
|---|---|---|---|---|
| B1 | Evaluation interval during training | **Not stated** for the main 49-game results (only the ablation captions mention 250,000 training frames) | **Every 1,000,000 frames** | Must be declared in the report's Limitations: the Paper-vs-Ours comparison therefore depends on an assumption about the evaluation schedule. |
| B2 | Which checkpoint is reported | **Not stated** for the main results (only the ablation captions say "the highest average episode score is reported") | **Final model** | Same limitation. We do *not* select the best checkpoint, so our number is not comparable to a best-of-training number. |
| B3 | Random seeds | Not stated | **Seeds 1, 2, 3** | Report statistics as mean ± s.d. across three seeds, as the course requires. |
| B4 | Human baseline | Measured by a professional tester (~20 episodes after ~2 h practice) | **Paper's reported value (9.3 for Pong)** | We do not re-measure a human. The normalized score therefore inherits the paper's human baseline. |

## C. Implementation provenance

| # | Item | Paper | Ours | Reason |
|---|---|---|---|---|
| C1 | Starting code | Original Torch7 implementation, never publicly released; the "Code availability" link now redirects to a DeepMind Workspace sign-in | **Written from scratch**; `cleanrl` @ `fe8d8a0` used as reference implementation | The course lists cleanrl as the reference. Atari wrapper and replay-buffer design are adapted from cleanrl / Stable-Baselines3 (MIT); see `THIRD_PARTY_NOTICES.md`. |
| C2 | Code structure | Single research script | 8-module package (`miae01_dqn/`) | Three-person parallel work; module boundaries documented in `docs/architecture.md`. No effect on algorithm behaviour. |

## D. Settings confirmed identical to the paper (no deviation)

| # | Item | Value used | Source |
|---|---|---|---|
| D1 | Optimizer | RMSProp, lr 2.5e-4, momentum 0.95, squared-gradient momentum 0.95, ε 0.01 | EDT1 |
| D2 | Loss | TD error clipped to [−1, 1]; **no** gradient-norm clipping | Methods |
| D3 | Discount γ | 0.99 | EDT1 |
| D4 | Minibatch size | 32 | EDT1 |
| D5 | Replay start size | 50,000 frames (= 12,500 agent steps) | EDT1 |
| D6 | Update frequency | Every 4 agent actions | EDT1 |
| D7 | Target sync | Every **10,000 parameter updates** (≈160,000 frames) | EDT1 |
| D8 | Exploration | ε 1.0 → 0.1 linearly over 1,000,000 frames, then fixed at 0.1 | EDT1 |
| D9 | Action repeat | 4 | EDT1 |
| D10 | Input | 4 stacked 84×84 grayscale frames; max over last two raw frames; Y channel | Methods |
| D11 | Reward clipping | [−1, +1] during training; raw scores at evaluation | Methods |
| D12 | Evaluation protocol | 30 episodes per game, ≤5 min each, random no-op starts, ε = 0.05 | Methods |

---

## How to use this table

- The report's **Setup** section should reference this file rather than restate it.
- Anything in group **A** or **B** that plausibly explains a score gap must be revisited in the
  report's **Analysis of Discrepancies**; the replay-capacity ablation (group A3) is the planned
  controlled experiment for that purpose.
- When the implementation lands, replace "expected effect" remarks with measured values.
