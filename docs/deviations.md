# Deviations from the paper

Declared differences between our reproduction and Mnih et al. (2015). This table is the source
for the report's **Reproduction Setup** section, criterion
`a complete table of deviations from the paper (item, original setting, setting used and reason)`.

Paper values come from the Methods section and Extended Data Table 1. The
integration audit also compares the released implementation: see
[paper-code-audit.md](paper-code-audit.md) for counting differences and warm-up.

Three groups follow: **(A)** scale and hardware, **(B)** assumptions where the paper is silent,
**(C)** implementation provenance. Items that were checked and found *unchanged* are listed at
the end so the table is verifiably complete.

---

## A. Scale and hardware

| # | Item | Paper | Ours | Reason and expected effect |
|---|---|---|---|---|
| A1 | Games | 49 Atari 2600 games, one network per game | **1 game (Pong)** | Project compute/time budget; the current comparison covers Pong. |
| A2 | Training length | Paper reports 50M frames; released driver requests 50M interaction-loop iterations with repeat 4 | **10M nominal raw frames** (2.5M agent decisions) | Reduced budget; the two sources use different counting conventions. See the audit before quoting a budget ratio. Actual wall times are in `validation.md`. |
| A3 | Replay capacity | 1,000,000 stored recent observations | **200,000** | One million full (4, 84, 84) uint8 stacks would need about 28 GB before runtime overhead. Reduced capacity changes replay diversity. |
| A4 | Compute | GPU | **RTX 3080 Laptop GPU** for the full integration run; CPU for clean-environment checks | Supersedes the original developer machine's CPU-only description. Per-run metadata records the actual hardware. |
| A5 | Parallelism | — (one agent per game) | 1 environment, sequential runs | The trainer assumes a single env; parallel envs would change the replay layout. |

## B. Points where the paper is silent (our assumptions)

| # | Item | Paper | Ours | Reason |
|---|---|---|---|---|
| B1 | Evaluation interval during training | Main-result interval unspecified; ablation captions give 250,000 training frames | **Every 1,000,000 project frames** | Project evaluation cadence, recorded in each run configuration. |
| B2 | Which checkpoint is reported | Main-result selection unspecified; ablations report the highest evaluation mean | **Final model** | The reported score corresponds to the final scheduled evaluation. |
| B3 | Random seeds | Not stated | **Seed 1 in this delivery; seeds 1, 2, 3 planned** | Current standard deviations describe 30 games within one seed; the planned comparison also measures variation across seeds. |
| B4 | Human baseline | Measured by a professional tester (~20 episodes after ~2 h practice) | **Paper's reported value (9.3 for Pong)** | We do not re-measure a human. The normalized score therefore inherits the paper's human baseline. |

## C. Implementation provenance

| # | Item | Paper | Ours | Reason |
|---|---|---|---|---|
| C1 | Starting code | Original Torch7 implementation available at google-deepmind/dqn | **Written from scratch**; `cleanrl` @ `fe8d8a0` used as reference implementation | The course lists cleanrl as the reference. Atari wrapper and replay-buffer design are adapted from cleanrl / Stable-Baselines3 (MIT); see `THIRD_PARTY_NOTICES.md`. |
| C2 | Code structure | Torch7 research implementation | 8-module package (`miae01_dqn/`) | Three-person parallel work; module boundaries documented in `docs/architecture.md`. No effect on algorithm behaviour. |

## D. Paper parameters under the project's stated interpretation

The numeric values below follow the project's counting convention. The
released-code comparison for D5/D7/D8 is detailed in `paper-code-audit.md`.

| # | Item | Value used | Source |
|---|---|---|---|
| D1 | Optimizer | RMSProp, lr 2.5e-4, momentum 0.95, squared-gradient momentum 0.95, ε 0.01 | EDT1 |
| D2 | Loss | TD error clipped to [−1, 1]; **no** gradient-norm clipping | Methods |
| D3 | Discount γ | 0.99 | EDT1 |
| D4 | Minibatch size | 32 | EDT1 |
| D5 | Replay start size | 50,000 nominal raw frames (= 12,500 agent steps); epsilon decays during warm-up, a deviation from strictly uniform random collection | EDT1 / project contract; audit note |
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


## Integration update (2026-11-15)

The integration experiment uses an RTX 3080 Laptop GPU; per-run environment metadata records its hardware and timings. Scale is one game, 200k replay capacity and 10M nominal raw frames. The centered RMSProp equations, reduction convention and smoke profile are documented in [full-implementation.md](full-implementation.md). Observed scores are reported in [validation.md](validation.md).
