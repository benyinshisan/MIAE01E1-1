# Issue #13 — independent runner and end-to-end verification

Author: 2260032112 黄俊皓, who implemented neither the runner nor its CLI
(Issues #4 and #6 are mine; the runner is #8, 陈博文). This satisfies the
"written by a non-implementer" rule for X-05.

Base: main `7a73983`. No production code is changed by this branch: the change
is one test file, this document and one recorded smoke run.

## Acceptance coverage

| Issue #13 requirement | Evidence |
|---|---|
| `build_config` rejects unknown parameters; `run_name` is reversible | CLI/`build_config` tests; `ValueError: unknown field 'bogus'` and `ValueError: run_name must identify a child directory` reproduced by hand; the run directory name is parsed back into `(env, seed, capacity)` and compared with `config.json` |
| A finished run leaves `tfevents` / `config.json` / `result.json` | 5,000-step `--smoke` run recorded in `runs/ALE-Pong-v5__seed1__cap5000__20261121T054802905418Z/`; the automated test repeats the same artifact check at a 400-step budget |
| Whole-repository grep: no drive-letter paths, no hard-coded environment paths | `git grep "D:/"` and `git grep "C:\"` return exit 1 (no hits); the automated scan covers every tracked `*.py|*.yml|*.toml` |
| A clean conda environment follows the README | New environment created from `environment.yml` on a machine that never had the project installed; full suite and CLI smoke run below |
| `--seed` / `--replay-capacity` overrides take effect | CLI test plus a hand run: `--seed 7 --replay-capacity 4096` produced `ALE-Pong-v5__seed7__cap4096__…` with `config.json` showing `seed=7`, `replay_capacity=4096` |

## What the author's runner tests do not cover

`tests/test_config_network_runner.py` calls `build_config` and `run_experiment`
directly. The nine tests added here exercise what a user actually types:

| Test | What it pins |
|---|---|
| `test_cli_run_writes_every_documented_artifact` | `main()` → `build_config` → `run_experiment` end to end; all five documented artifacts plus `events.out.tfevents.*`; `agent_steps`/`frames`/evaluation length in `result.json` |
| `test_run_directory_name_is_reversible` | the directory name parsed back into `(env, seed, capacity)` matches `config.json` |
| `test_cli_overrides_reach_the_config` | `--seed`, `--replay-capacity`, `--log-dir` reach the resolved config and the name |
| `test_build_config_rejects_unknown_fields_and_bad_run_name` | typo'd keys and `run_name=".."` fail loudly |
| `test_smoke_profile_matches_the_documented_budget` | `--smoke` is the explicit profile 5,000 steps / 1,000 warm-up frames / 5,000 capacity / 2 games |
| `test_same_seed_reproduces_scalars_and_evaluation` | two independent runs with seed 11 reproduce every non-SPS scalar, the evaluation score vector and its mean |
| `test_cuda_request_is_rejected_without_cuda` | `--device cuda` on a CPU build fails before any run directory exists |
| `test_no_machine_specific_paths_in_tracked_sources` | repository-wide scan of tracked code/config |
| `test_committed_run_configs_use_relative_paths` | committed `runs/*/config.json` never embeds an absolute log directory |

The YAML `--config` route is covered because the CLI exposes no flag for the
evaluation budget (see finding F1), so the tests shrink evaluation through a
small YAML file.

## Clarifications

**Drive-path scan.** A naive `[A-Za-z]:[\\/]` also matches `https://` (the `s:/`)
and the escaped newline in `'algo:\n'`. The test therefore requires the drive
letter to start a token:
`(?<![A-Za-z0-9+.\-])[A-Za-z]:[\\/]`. With that correction the scan is clean;
without it, `environment.yml` and the author's own test file are false
positives.

**Literal grep.** The issue's wording is satisfied literally as well:
`git grep -nI "D:/"` and `git grep -nI "C:\"` both return no matches (exit 1)
over the whole repository.

**`--smoke` budget.** 5,000 agent steps, 1,000 warm-up frames, 5,000 replay
capacity and 2 evaluation games are engineering choices, not paper values. They
are pinned by a test so the shortcut cannot drift silently.

## Findings

| # | Finding | Action |
|---|---|---|
| F1 | The CLI exposes no `--eval-episodes` / `--max-episode-frames` flag; the evaluation budget is only reachable through `--config`. `--smoke` is the sole CLI shortcut. | Low priority; recorded for #29's runbook. No code change here. |
| F2 | The pinned `torch==2.4.1` CUDA wheels contain kernels up to `sm_90`; this machine's RTX 5060 Ti is `sm_120`, so verification used the CPU wheel (`torch.cuda.is_available()` is `False`). README's "CUDA machine" line should mention the fallback. | Refer to #29; not a defect in the runner. |
| F3 | Committed run configurations do use the relative `runs` log directory; the absolute path appears only in `result.json.run_dir` when a caller passes an absolute `--log-dir` (as this verification did outside the repository). | No action. |

## Reproduction and result

```bash
conda env create -f environment.yml        # name: miae01-dqn
conda activate miae01-dqn
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m pip install -r requirements-test.txt
python -m pytest -q                        # 69 passed
python -m miae01_dqn.runner --smoke --seed 1 --device cpu
```

Clean environment and machine:

| Item | Value |
|---|---|
| Python | 3.12.16 |
| torch | 2.4.1+cpu (`cuda.is_available() == False`) |
| numpy / gymnasium / ale-py | 2.5.4 / 1.0.0 / 0.10.1 |
| opencv / matplotlib / pytest | 5.0.0 (headless) / 3.11.2 / 9.2.0 |
| OS / CPU / RAM | Windows, Ryzen 7 7800X3D, 15.2 GB |

Automated suite on 2026-11-21: **69 passed in 35.90 s** (60 pre-existing tests
from `main` plus 9 new ones), with no skips and no expected failures; a second
run with `-rs` confirmed **69 passed in 36.80 s**.

5,000-step smoke run (`runs/ALE-Pong-v5__seed1__cap5000__20261121T054802905418Z`):

| Evidence | Value |
|---|---:|
| Agent steps / nominal frames | 5,000 / 20,000 |
| Gradient updates | 1,187 |
| Completed episodes | 5 |
| Training-loop wall time | 38.51 s |
| Whole-run wall time | 41.25 s |
| Logged SPS (final) | 125.0 |
| Process RSS | 447.7 MiB |
| Final training-episode return mean | −20.2 |
| Two-game evaluation mean | −21.0 (std 0.0), normalized −1.0 |
| Scalar records in `metrics.jsonl` | 27 |

## Test scope

This file verifies the runner surface: CLI parsing, configuration resolution,
artifact layout, determinism and path hygiene. It does not re-derive the
evaluator's 30-game protocol, the agent's TD maths or a full 2.5M-step seed —
those belong to X-03, X-04 and #15 respectively. The two "same seed" runs use
small budgets; they pin reproducibility of the wiring, not of a long training
curve.
