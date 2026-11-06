# DQN Replication — Human-level Control through Deep Reinforcement Learning

Replication of **Mnih et al., "Human-level control through deep reinforcement learning", *Nature* 518:529–533 (2015)**, for the course *MIAE01 — Programming and Tools for Machine Learning* (MUST, AY 2026–2027).

We reproduce the paper's central claim on a **reduced scale** (one game, 10M frames instead of 49 games, 50M frames) and report a Paper-vs-Ours comparison together with the deviations that explain the gap.

---

## Status

> **This repository is in the implementation phase.** The interface contract and environment are complete; the modules themselves are being written.

| Part | State |
|---|---|
| Environment (`environment.yml`, `requirements.txt`) | ✅ done, verified on Python 3.12 |
| Interface contract (`prototype/interfaces.py`) | ✅ done |
| Architecture & invariants (`docs/architecture.md`) | ✅ done |
| Module implementation (`miae01_dqn/`) | 🚧 in progress — see Issues |
| Smoke run / main experiment | ⏳ after implementation |

Run commands in this README describe the **target interface**. They become usable as the corresponding issues close; anything not yet runnable is marked `[pending]`.

---

## 1. What we reproduce

**Claim tested (one claim):**

> A single architecture, with the same hyperparameters and only raw pixels as input, reaches above-human performance on Pong.

**Reference numbers** (paper, Extended Data Table 2, p. 11):

| Metric | Value |
|---|---|
| DQN | **18.9** (±1.3) |
| Professional human | **9.3** |
| Random policy | **−20.7** |
| Human-normalized | 132.0% |

Normalization: `100 × (ours − random) / (human − random)`.

**Verdict rule we pre-registered:** above 9.3 → *supported*; positive but below 9.3 → *partially supported*; fails to learn → *not supported*.

**Not reproduced:** generality across 49 games (hardware-limited; see §8 Deviations).

**Extension (one factor):** replay capacity ablation at 50k / 100k / 200k, used as evidence when explaining the gap to the paper.

---

## 2. Quick start

### Option A — conda (local, CPU)

```bash
conda env create -f environment.yml
conda activate miae01-dqn
```

### Option B — Colab (free tier) / any Python 3.12 environment

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` deliberately **does not install PyTorch**: Colab ships a CUDA build. Locally, `environment.yml` pins `torch==2.4.1`.

Verified combination: Python 3.12.16, `gymnasium==1.0.0`, `ale-py==0.10.1`, `numpy==2.5.4`.

---

## 3. Data and ROMs — nothing to download

The Atari ROMs are **bundled inside the `ale-py` wheel** (`ale_py/roms`, 108 ROM files, 672 KB total), so no `AutoROM`, no ROM download step and no licensed binary is stored in this repository.

Environment ids: we use **`ALE/Pong-v5`** and must pass **`frameskip=1`** explicitly — the v5 default is 4, which would compound with our `MaxAndSkip(4)` wrapper into 16 frames per agent action.

---

## 4. Run commands

All commands are run from the repository root. `[pending]` marks commands that require modules still under construction.

```bash
# Smoke test: 5,000 agent steps, ~1 minute, writes logs to runs/
[pending] python -m miae01_dqn.runner --smoke

# Main experiment: Pong, 3 seeds, 2.5M agent steps each
[pending] python -m miae01_dqn.runner --env-id ALE/Pong-v5 --seeds 1 2 3 \
              --total-agent-steps 2500000 --replay-capacity 200000

# Extension: replay-capacity ablation
[pending] python -m miae01_dqn.runner --env-id ALE/Pong-v5 --seeds 1 2 3 \
              --total-agent-steps 2500000 --replay-capacity 50000
[pending] python -m miae01_dqn.runner --env-id ALE/Pong-v5 --seeds 1 2 3 \
              --total-agent-steps 2500000 --replay-capacity 100000

# Evaluation of a finished run
[pending] python -m miae01_dqn.evaluator --run-dir runs/<run_name> --episodes 30
```

Seed handling: every run is identified by `env_id`, `seed` and the key hyperparameters, and the seed is propagated to the environment, NumPy, `random`, PyTorch and cuDNN.

---

## 5. Expected outputs and runtimes

Each run writes to `runs/<env_id>__<seed>__<tag>__<timestamp>/`:

| Artifact | Content |
|---|---|
| `events.out.tfevents.*` | TensorBoard scalars: `episodic_return`, `epsilon`, `td_loss`, `q_values`, `SPS` |
| `config.json` | full resolved configuration for that run |
| `result.json` | final evaluation: per-episode scores, mean, std, normalized score |

Measured throughput on the development machine (CPU-only, no CUDA device available): **≈273 agent steps/s**.
At that rate a 2.5M-step run takes **≈2.3 h per seed**; three seeds ≈ 7 h. A 5k-step smoke run takes under a minute.

Memory: the replay buffer stores one `uint8` array of shape `(capacity, 4, 84, 84)`, i.e. **≈5.6 GB at 200k**. The paper's 1M capacity would need ≈28 GB and is not feasible here.

---

## 6. Repository layout

```text
.
├── prototype/
│   └── interfaces.py        # the contract: signatures only, no implementation
├── docs/
│   └── architecture.md      # module boundaries, invariants, paper-default table
├── environment.yml          # conda environment (local, CPU)
├── requirements.txt         # pip-only environment (Colab)
├── .gitignore
└── .gitattributes

# planned, added as issues close:
# miae01_dqn/   config.py envs.py buffer.py network.py agent.py trainer.py evaluator.py runner.py
# tests/        cross-validation tests (written by a member who did not implement the module)
# reference/    upstream files kept for comparison, with their original licenses
```

Module dependency direction (enforced):

```text
runner → trainer → agent → network
  │         │         └──→ buffer
  │         └──→ envs ──→ config
  └──→ evaluator ─→ agent, envs
```

---

## 7. Paper configuration (defaults)

Values are taken from the paper (Methods and Extended Data Table 1) and are the defaults of `RunConfig`.

| Parameter | Paper value |
|---|---|
| Optimizer | RMSProp, lr 2.5e-4, momentum 0.95, squared-gradient momentum 0.95, ε 0.01 |
| Loss | TD error **clipped to [−1, 1]** (no gradient-norm clipping) |
| Discount γ | 0.99 |
| Minibatch | 32 |
| Replay capacity | 1,000,000 frames |
| Replay start size | 50,000 frames |
| Update frequency | every 4 agent actions |
| Target sync | every **10,000 parameter updates** (≈160,000 frames) |
| Exploration | ε 1.0 → 0.1 linearly over the first **1,000,000 frames**, then fixed 0.1 |
| Action repeat | 4 |
| Frame stack | 4 × 84×84 grayscale |
| Reward clipping | [−1, +1] during training; raw scores at evaluation |
| Evaluation | 30 episodes, ≤5 min each, random no-op starts, ε = 0.05 |

> Three units are easy to confuse: **frames**, **agent steps** and **parameter updates**. The target-sync interval is counted in parameter updates; the ε schedule is counted in frames.

---

## 8. Deviations from the paper

| Item | Paper | Ours | Reason |
|---|---|---|---|
| Games | 49 | 1 (Pong) | CPU-only machine; 49 × 3 seeds is not feasible |
| Training budget | 50M frames | 10M frames (2.5M agent steps) | ~2.3 h/seed at 273 steps/s |
| Replay capacity | 1,000,000 | 200,000 | memory: 1M would need ≈28 GB |
| Hardware | GPU (reported as 38 days of game experience) | CPU only, no CUDA device | available hardware |
| Evaluation interval | not stated for the main results | every 1M frames | **our assumption**, declared in the report's Limitations |
| Reported checkpoint | not stated for the main results | final model | **our assumption**, same reason |
| Optimizer / loss | RMSProp / TD error clipped | same | no deviation |

---

## 9. Reproducibility notes

- **Package versions are pinned** in `environment.yml` and `requirements.txt`.
- **No hard-coded paths**: all paths are derived from `RunConfig.log_dir` and relative paths; the repository is checked for absolute paths in the cross-validation tests.
- **Seeds are fixed** per run and recorded in `config.json`.
- **Device is configurable** (`--device cpu|cuda`); results on CPU and GPU are not expected to be bit-identical (cuDNN non-determinism).
- Colab free tier has 12.7 GB RAM; 200k replay ≈ 5.6 GB, so runs fit but with little headroom.

---

## 10. Starting code, credits and license

The paper's original implementation was never publicly released: the "Code availability" note (Methods, p. 6) points to `sites.google.com/a/deepmind.com/dqn`, which now redirects to a DeepMind Google Workspace sign-in and is not publicly accessible. The course therefore lists a reference implementation.

| Component | Source | Commit / version | License |
|---|---|---|---|
| Reference implementation | [`vwxyzjn/cleanrl`](https://github.com/vwxyzjn/cleanrl) | `fe8d8a03c41a7ef5b523e2e354bd01c363e786bb` | MIT |
| Atari wrappers, replay buffer | `stable-baselines3` (cleanrl's own `atari_wrappers.py` and `buffers.py` are adapted from SB3) | see `THIRD_PARTY_NOTICES.md` | MIT |
| Algorithm specification | Mnih et al., *Nature* 518:529–533 (2015) | DOI `10.1038/nature14236` | — |

Full license texts and per-file provenance: **`THIRD_PARTY_NOTICES.md`**.

**Files modified / adapted from upstream** (updated as implementation proceeds):

| Our file | Derived from | What we changed |
|---|---|---|
| `miae01_dqn/envs.py` | cleanrl `atari_wrappers.py` (adapted from SB3) | wrapped into an `EnvFactory`; evaluation env disables life-loss truncation; `frameskip=1` |
| `miae01_dqn/buffer.py` | cleanrl `buffers.py` (adapted from SB3) | reduced to a single `uint8` ring array; removed `n_envs` and timeout branches |
| `miae01_dqn/agent.py` | paper, Methods | written for this replication: RMSProp, TD-error clipping, target sync counted in updates |
| `miae01_dqn/trainer.py` | paper, Algorithm 1 | written for this replication: step-driven loop, truncation handling |
| `miae01_dqn/{config,network,evaluator,runner}.py` | paper | written for this replication |

---

## 11. Team and contributions

| Member | Student ID | Contributions |
|---|---|---|
| | | |
| | | |
| | | |

> Fill in before the final submission; must be consistent with the git history.

---

## 12. AI usage statement

AI assistance was used for: repository scaffolding and environment setup, code review of the reference implementation, drafting of this README and the architecture document, and analysis of training logs. All experimental results reported in the final report are produced by the code in this repository and are traceable to the logs committed here.

---

## 13. Acknowledgements

Course material and repository template by Prof. Xie Xuan (MUST). Reference implementation by the cleanrl authors; Atari wrappers and replay-buffer design by the Stable-Baselines3 authors.
