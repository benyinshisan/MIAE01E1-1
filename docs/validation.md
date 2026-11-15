# Validation evidence

## Automated regression suite

The full implementation has executable coverage for all eight modules. The
independent-author requirement of the cross-test Issues still needs team review;
see `full-implementation.md`. Test execution alone is not independent sign-off.

On 2026-11-15, `python -m pytest -q` completed with **57 passed** in 34.12 seconds,
with no skipped or expected-failure tests, on the machine listed below.

## Issue #14: two 5,000-step smoke runs

Machine: Windows, NVIDIA RTX 3080 Laptop GPU, Python 3.12.16,
torch 2.4.1+cu121, Gymnasium 1.0.0, ALE 0.10.1, NumPy 2.5.4.
Run configuration and full metadata are stored beside each result.

| Evidence | smoke_seed1_a | smoke_seed1_b |
|---|---:|---:|
| Seed / steps / frames | 1 / 5,000 / 20,000 | 1 / 5,000 / 20,000 |
| Gradient updates | 1,187 | 1,187 |
| Completed training episodes | 5 | 5 |
| Training wall time, seconds | 16.6424 | 16.5522 |
| Final logged SPS | 304.4 | 286.9 |
| Process RSS, GiB | 1.15 | 1.15 |
| Mean of observed training episode returns | -20.2 | -20.2 |
| Final two-game evaluation mean | -21.0 | -21.0 |

Online and target model tensors matched **exactly** between the two runs, as
did episode/update counters and evaluation results. Wall clocks naturally differ.
Both runs produced events, configuration, results and a learning curve. Model
checkpoints used for this local comparison are excluded from Git.

These are smoke-profile runs with reduced warm-up, capacity and evaluation
count; the scores do not represent a trained DQN result.

## Issue #15: first full seed

The user explicitly requested completion on the local GPU. The full run uses
seed 1, 200,000 replay capacity, 2,500,000 agent steps and 30-game evaluation
every 1,000,000 training frames. Source implementation commit: `0a39511`.

Full-run result and measured wall time will be recorded after the run finishes.
No success or score conclusion is inferred from partial progress.
