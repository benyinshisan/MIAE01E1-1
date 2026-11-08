# DQN Replication — Human-level Control through Deep Reinforcement Learning

Replication of **Mnih et al., "Human-level control through deep reinforcement learning", *Nature* 518:529–533 (2015)**, for the course *MIAE01 — Programming and Tools for Machine Learning* (MUST, AY 2026–2027).

We reproduce the paper's central claim on a **reduced scale** (one game, 10M frames instead of 49 games, 50M frames) and report a Paper-vs-Ours comparison together with the deviations that explain the gap.

---

## Status

> **This repository is in the implementation phase.** The package skeleton and the environment are complete; the modules themselves are being written.

| Part | State |
|---|---|
| Environment — conda (`environment.yml`) | ✅ verified from a clean environment (Python 3.12.16, Windows, CPU build) |
| Environment — Colab (`requirements.txt`) | ⚠️ prepared, **not verified** |
| Package skeleton (`miae01_dqn/`, `pyproject.toml`) | ✅ done — importable, `pytest` green |
| Architecture & invariants (`docs/architecture.md`) | ✅ done |
| Module implementation (`miae01_dqn/`) | 🚧 in progress — see Issues |
| Smoke run / main experiment | ⏳ after implementation |

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

**Not reproduced:** generality across 49 games (hardware-limited; every deviation from the paper, with the reason, is listed in [`docs/deviations.md`](docs/deviations.md)).

**Extension (one factor):** replay capacity ablation at 50k / 100k / 200k, used as evidence when explaining the gap to the paper.

---

## 2. Quick start

### Option A — conda (the only verified path)

```bash
conda env create -f environment.yml
conda activate miae01-dqn
```

`environment.yml` intentionally does **not** install PyTorch, because the right wheel depends on
the hardware. Install it once, with the build that matches your machine:

```bash
# CPU-only machine (~200 MB) — this is what we develop on
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

# CUDA machine (~2.4 GB)
pip install torch==2.4.1
```

Verified combination: Python 3.12.16, `gymnasium==1.0.0`, `ale-py==0.10.1`, `numpy==2.5.4`, `torch==2.4.1+cpu`.

### Install the package

This step makes `import miae01_dqn` and `python -m miae01_dqn.runner` work from any directory:

```bash
python -m pip install -e .
python -m pytest -q          # expected: 3 passed
```

### Editing the code after installation

`pip install -e .` is an *editable* install: the entry in `site-packages` is a link to this
source tree, so changes to `.py` files take effect on the next run — **no reinstall needed**.
Reinstall only when `pyproject.toml` changes (metadata, package layout).

To confirm you are running the editable copy rather than a stale installed one:

```bash
python -c "import miae01_dqn; print(miae01_dqn.__file__)"
# the path must be inside this repository, not under site-packages/
```

### Colab — prepared, **not verified**

> The Colab path has **not** been tested. Only the conda path above is verified.
> Treat the commands below as a starting point, not a guarantee.

```python
!git clone https://github.com/benyinshisan/MIAE01E1-1.git
%cd MIAE01E1-1
!pip install -q -r requirements.txt
!pip install -q -e .
!python -m pytest -q
```

`requirements.txt` omits PyTorch because Colab already ships a CUDA build.

---

## 3. Expected outputs and runtimes

When the implementation lands, each run will write to `runs/<env_id>__<seed>__<tag>__<timestamp>/`:

| Artifact | Content |
|---|---|
| `events.out.tfevents.*` | TensorBoard scalars: `episodic_return`, `epsilon`, `td_loss`, `q_values`, `SPS` |
| `config.json` | full resolved configuration for that run |
| `result.json` | final evaluation: per-episode scores, mean, std, normalized score |

Measured throughput on the development machine (CPU-only, no CUDA device available): **≈273 agent steps/s**.
At that rate a 2.5M-step run would take **≈2.3 h per seed**; three seeds ≈ 7 h. A 5k-step smoke run takes under a minute.

Memory: the replay buffer stores one `uint8` array of shape `(capacity, 4, 84, 84)`, i.e. **≈5.6 GB at 200k**. The paper's 1M capacity would need ≈28 GB and is not feasible here.

---

## 4. Repository layout

```text
.
├── miae01_dqn/              # the package (8 modules, one per interface unit)
│   ├── __init__.py          # public exports + __version__
│   ├── config.py            # EnvSpec / OptimSpec / AlgoSpec / EvalSpec / RunSpec / RunConfig
│   ├── envs.py              # EnvFactory, EnvHandle
│   ├── buffer.py            # ReplayBuffer, TransitionBatch (numpy only)
│   ├── network.py           # QNetwork (paper CNN)
│   ├── agent.py             # Agent (act / update / sync_target)
│   ├── trainer.py           # Trainer, TrainResult (step-driven loop)
│   ├── evaluator.py         # Evaluator, EvalResult (30-episode protocol)
│   └── runner.py            # Logger, build_config, run_experiment, CLI entry
├── tests/                   # cross-validation tests (written by a non-implementer)
├── docs/
│   ├── architecture.md      # module boundaries, invariants, paper-default table
│   └── deviations.md        # every deviation from the paper, with reasons
├── pyproject.toml           # editable install + pytest config
├── environment.yml          # conda environment (single source of truth)
├── requirements.txt         # pip-only mirror (any Python 3.12 environment)
├── THIRD_PARTY_NOTICES.md   # upstream licences
├── .gitignore
└── .gitattributes
```

Module dependency direction (enforced):

```text
runner → trainer → agent → network
  │         │         └──→ buffer
  │         └──→ envs ──→ config
  └──→ evaluator ─→ agent, envs
```

---

## 5. Reproducibility notes

- **Package versions are pinned** in `environment.yml` and `requirements.txt`; PyTorch is installed
  separately because the build depends on the hardware.
- **No hard-coded paths**: all paths are derived from `RunConfig.log_dir` and relative paths; the
  repository is checked for absolute paths in the cross-validation tests.
- **Seeds are fixed** per run and recorded in `config.json`.
- **Device is configurable** (`--device cpu|cuda`); results on CPU and GPU are not expected to be
  bit-identical (cuDNN non-determinism).
- Atari ROMs are bundled inside the `ale-py` wheel, so no ROM download step is required.

---

## 6. Starting code, credits and license

The paper's original implementation was never publicly released: the "Code availability" note
(Methods, p. 6) points to `sites.google.com/a/deepmind.com/dqn`, which now redirects to a
DeepMind Google Workspace sign-in and is not publicly accessible. The course therefore lists a
reference implementation.

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
