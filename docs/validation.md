# Validation evidence

## Automated regression suite

The full implementation has executable coverage for all eight modules. The
independent-author requirement of the cross-test Issues still needs team review;
see `full-implementation.md`. Test execution alone is not independent sign-off.

On 2026-11-15, `python -m pytest -q` completed with **57 passed** in 34.12 seconds,
with no skipped or expected-failure tests, on the machine listed below.

After the Windows JSON-lock regression was added, a **new conda environment**
was created directly from `environment.yml`, followed by the documented CPU
PyTorch install and `pip install -e .`. With no shared dependency directory on
PYTHONPATH, the complete suite passed **58 tests in 35.13 seconds**. Verified
versions: Python 3.12.16, torch 2.4.1+cpu, NumPy 2.5.4, Gymnasium 1.0.0. This
provides the clean-install evidence requested by Issue #13.

After adding the zero-epsilon boundary tests identified while reviewing the
concurrent Agent PR #22, the same clean environment passed **60 tests in
34.83 seconds**, again without skips or expected failures. The paper-default
optimizer calculation is unchanged.

The new environment also completed `--smoke --seed 1 --device cpu`: 5,000 steps,
1,187 updates, 5 completed episodes, training wall time 69.17 seconds, final
logged 69.2 SPS, RSS 0.43 GiB, and all required output files. This CPU result was
measured while the separate full GPU run was active; it is not a dedicated
hardware throughput benchmark.

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

The initial attempt (`seed1_cap200k_full`) reached about 504k steps before a
Windows file-reader lock prevented replacing progress.json. Its partial logs,
two evaluations and failure report are retained as debugging evidence, not as
a completed experiment. Commit `d3b95ba` adds bounded retry for atomic JSON
replacement, with a regression test that simulates the lock (6 Runner tests
passed). The full seed was restarted from initialization as
`seed1_cap200k_complete`, using unchanged training hyperparameters.

Full-run result and measured wall time will be recorded after the restarted run
finishes. No success or score conclusion is inferred from partial progress.
